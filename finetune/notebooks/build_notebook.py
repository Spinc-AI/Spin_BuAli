"""Generate the Kaggle fine-tuning notebook.

Same rule as `benchmark/notebooks/build_notebook.py`: **the notebook has no
logic of its own.** Everything it runs lives in `finetune/*.py`, covered by
`finetune/tests/`. This file's only job is to clone the repo, install what
GPU-dependent code needs (torch/transformers/peft/datasets are not installed
in the environment that WROTE this pipeline, which is exactly why the real
proof this works has to happen here, on Kaggle, not in that environment), and
call the training/evaluation scripts as subprocesses -- so their own
`argparse` interface is the only contract between this notebook and them.

    python build_notebook.py
"""
import json
import pathlib

REPO_URL = "https://github.com/Spinc-AI/Spin_BuAli.git"
BRANCH = "audio-finetune"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ══════════════════════════════════════════════════════════════════════════
md(f"""
# Spin BuAli — Gemma 4 domain fine-tuning

LoRA fine-tunes Gemma 4 (the encoder-free, audio-capable "Unified" variant --
`gemma-4-e4b`/`gemma-4-12b` in the benchmark's own multimodal roster) on this
project's own dictations, trained against the **exact prompt shape
production uses**: `core_llm/model.py`'s `GemmaAudioModel` chat template
(system + user-with-audio + assistant), not a separate transcription-only
API. That is the whole reason Gemma was chosen over the other audio-capable
models in the roster for this — see `finetune/README.md`'s Research section
for the full citation trail and the one published precedent this recipe is
built from.

**Before running:** Settings → Accelerator **GPU T4 ×2**, Internet **On**.
This notebook does not need the `spin-buali-dataset` Kaggle Dataset attached
if you instead attach a public smoke-test dataset first (recommended on a
first run — see cell 5).

**What "proven to work" does and does not mean here** — this pipeline has
never executed a real training loop; the environment it was written in has
no GPU and no `torch` installed at all. Every non-GPU-dependent piece is
genuinely tested (`finetune/tests/`), and both scripts were run for real
against a fixture dataset, reaching exactly the point where `torch`/
`transformers`/`datasets` are needed before failing. Cell 6 below
(`--dry-run`) is what closes that gap on real hardware, in minutes, before
any training cell runs. Read `finetune/README.md` before trusting a number
out of this notebook.
""")

# ── 1. Hardware ──────────────────────────────────────────────────────────
md("## 1 — Check the hardware")
code("""
!nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
""")

# ── 2. Get the code ──────────────────────────────────────────────────────
md(f"""
## 2 — Get the code

Clones `{REPO_URL}` (branch `{BRANCH}`) into `/kaggle/working/Spin_BuAli`. If
it is already there, pulls instead — re-running this cell after a fix
upstream is enough to pick it up.
""")
code(f'''
import pathlib, subprocess, sys

REPO_DIR = pathlib.Path("/kaggle/working/Spin_BuAli")

if REPO_DIR.exists():
    print("repo already present -- pulling the latest commit")
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "{BRANCH}"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/{BRANCH}"], check=True)
else:
    subprocess.run(["git", "clone", "--branch", "{BRANCH}", "--depth", "1",
                    "{REPO_URL}", str(REPO_DIR)], check=True)

FINETUNE_DIR = REPO_DIR / "finetune"
commit = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, check=True).stdout.strip()
print(f"\\nrunning commit {{commit}}")
''')

# ── 3. Install ───────────────────────────────────────────────────────────
md("""
## 3 — Install dependencies

Same `transformers>=5.5.0`, `--no-deps` reasoning as the benchmark notebook
(see its own cell 2 for why) -- this repo pins that version for a reason
unrelated to fine-tuning specifically, and the two notebooks should not
silently diverge on it. `datasets`, `peft`, `bitsandbytes` and `jiwer` are
this notebook's own additions.
""")
code("""
!pip install -q --no-deps --upgrade "transformers>=5.5.0" tokenizers huggingface-hub safetensors
!pip install -q python-dotenv sentencepiece bitsandbytes accelerate
!pip install -q datasets peft jiwer
""")
code('''
import numpy, torch, transformers

print(f"transformers {transformers.__version__}")
print(f"torch        {torch.__version__}")
print(f"numpy        {numpy.__version__}")

_major, _minor = (int(x) for x in transformers.__version__.split(".")[:2])
if (_major, _minor) < (5, 5):
    raise SystemExit(
        f"transformers {transformers.__version__} is too old -- restart the "
        "kernel and re-run the install cell above before continuing.")
print("\\nversions OK")
''')

# ── 4. HF auth ───────────────────────────────────────────────────────────
md("""
## 4 — Hugging Face authentication

Gemma 4 is a gated checkpoint. Typed live, not written into this notebook --
same reasoning as the benchmark notebook's own cell 4.
""")
code('''
import getpass
import os

from huggingface_hub import login, whoami


def resolve_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        secret = UserSecretsClient().get_secret("HF_TOKEN")
        if secret:
            return secret.strip(), "Kaggle secret HF_TOKEN"
    except Exception:
        pass
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip(), f"${name}"
    typed = getpass.getpass("Hugging Face token (hidden while typing, not saved anywhere): ")
    return (typed.strip(), "typed just now") if typed.strip() else (None, None)


_token, _source = resolve_hf_token()
if _token:
    login(token=_token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"] = _token
    print(f"signed in as {whoami().get('name', 'unknown')}   (token from {_source})")
else:
    print("No token given. Gemma 4 will fail to download without one.")
''')

# ── 5. Dataset ───────────────────────────────────────────────────────────
md("""
## 5 — Dataset

**Recommended for a first run: a public smoke-test dataset, not the private
`spin-buali-dataset`.** Only nine labelled clips exist there today — nowhere
near enough to expect a real fine-tune to generalize from. Proving the
*code* works (data loads, a batch collates, loss goes down) against a public
dataset first is a five-minute check; discovering a wiring bug three epochs
into the real, small, private dataset is not.

`LABELS_CSV` below should point at a `labels.csv` in the shape
`benchmark/dataset.py`'s `from_csv` expects (`asset_id,audio,report` columns,
paths relative to the CSV) -- either the real one from the attached
`spin-buali-dataset` Kaggle Dataset, or one you build from a public ASR
dataset's `audio`/`text` columns for the smoke test.
""")
code("""
LABELS_CSV = None   # e.g. "/kaggle/input/spin-buali-dataset/Spin_BuAli_DataSet/Small_Demo/labels.csv"

if LABELS_CSV is None:
    raise SystemExit("set LABELS_CSV above before running the cells below")

import pandas as pd
labels = pd.read_csv(LABELS_CSV)
print(f"{len(labels)} row(s) in {LABELS_CSV}")
labels.head()
""")

# ── 6. Dry run ───────────────────────────────────────────────────────────
md("""
## 6 — Dry run: prove the pipeline before spending GPU-hours on it

Loads data, builds one real batch, runs one forward+backward step, and
exits. Same reason `runner.run_one`'s `_release_leaked_vram()` exists in the
benchmark: a mistake should cost seconds, not an interrupted training run
partway through.
""")
code("""
import subprocess

subprocess.run(["python", "-m", "gemma.train_lora",
                "--labels-csv", LABELS_CSV, "--dry-run"],
               cwd=str(FINETUNE_DIR), check=True)
""")

# ── 7. Train ─────────────────────────────────────────────────────────────
md("""
## 7 — Train

Every flag defaults to the recipe documented in `gemma/train_lora.py`'s own
module docstring and `finetune/README.md`; override anything here by adding
`--flag value` to the list below. `--per-device-batch-size` stays at its
default of 1 unless you have verified `GemmaChatCollator`'s audio-feature
batching works above that -- see its own docstring.
""")
code("""
subprocess.run(["python", "-m", "gemma.train_lora",
                "--labels-csv", LABELS_CSV,
                "--output-dir", "/kaggle/working/gemma-buali-lora"],
               cwd=str(FINETUNE_DIR), check=True)
""")

# ── 8. Evaluate ──────────────────────────────────────────────────────────
md("""
## 8 — Baseline vs fine-tuned WER

`final_text` (parsed out of the model's own JSON reply) against the
reference report, base checkpoint first, then with the trained adapter
merged in.
""")
code("""
subprocess.run(["python", "-m", "gemma.evaluate", "--labels-csv", LABELS_CSV],
               cwd=str(FINETUNE_DIR), check=True)
""")
code("""
subprocess.run(["python", "-m", "gemma.evaluate", "--labels-csv", LABELS_CSV,
                "--adapter-dir", "/kaggle/working/gemma-buali-lora"],
               cwd=str(FINETUNE_DIR), check=True)
""")

# ── 9. After training ────────────────────────────────────────────────────
md("""
## 9 — Score the fine-tuned checkpoint through the real benchmark

WER alone does not say whether the report-level metrics the benchmark
actually cares about (medical term F1, negation/laterality error rate) moved
too. To find out, merge the LoRA adapter into the base weights
(`PeftModel.merge_and_unload()`, as `gemma/evaluate.py` already does) and
save that; then set `GEMMA_E4B_MODEL_ID=<merged checkpoint path>` before
`benchmark/notebooks/kaggle_dual_t4.ipynb`'s cell 3 -- `core_llm/config.py`
reads that environment variable as `GemmaAudioModel`'s checkpoint for the
`gemma-4-e4b` key. Run that notebook unchanged after that; nothing else
about it needs to know a checkpoint was fine-tuned.
""")


def main():
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = pathlib.Path(__file__).parent / "kaggle_finetune.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {out}\n  {len(cells)} cells ({code_cells} code), {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
