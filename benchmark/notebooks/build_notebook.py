"""Generate the Kaggle notebook.

Rewritten from scratch around one rule: **the notebook has no logic of its
own.** Every class, function and metric lives in a `.py` file in this repo,
covered by `benchmark/tests/`. The notebook's job is to `git clone` the repo,
import those files, and call them -- one run, one cell, one CSV.

That makes bug-fixing a two-step loop instead of a re-upload:

    1. Fix the bug in the .py file here, run the tests, commit, push.
    2. In the notebook, re-run the "update the code" cell (a `git pull`) and
       re-run whichever run cell was wrong. Nothing else needs to change.

And it makes a run resumable at the granularity that matters: **one cell is
one (STT, LLM, pipeline) combination**, and it writes its own CSV before the
cell finishes. Stop the session after cell 6 of 10 and the other four are
still just four cells away -- not a loop that redoes the first six to get
there, because there is no loop.

    python build_notebook.py
"""
import json
import pathlib

REPO_URL = "https://github.com/parsafaramarzi/Spin_BuAli.git"
BRANCH = "benchmark-selfcontained"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ══════════════════════════════════════════════════════════════════════════
md(f"""
# Spin BuAli — radiology pipeline benchmark

Runs Persian radiology dictations through **speech recognition → language
model** and scores the finished report against the radiologist's signed one.

**Every run is its own cell**, and each one writes its own CSV before the cell
finishes. Stop the session whenever you like — nothing already run needs to be
redone, because nothing here is a loop.

**All the logic lives in the repo, not in this notebook.** Cell 2 clones it;
every cell after that imports from it. Fixing a bug means fixing the `.py`
file, pushing it, and re-running cell 2 — never re-uploading this file.

**Before running:** Settings → Accelerator **GPU T4 ×2**, Internet **On**, and
attach the `spin-buali-dataset` dataset under *Add Input*.
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
it is already there — because this is not the first cell run this session —
it pulls instead, so re-running this cell after a code fix upstream is enough
to pick it up.
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


# Only benchmark/ goes on the path directly. Its own bridge.py reaches
# evaluation/, controller/ and stt/app/ itself, appending each to the END of
# sys.path rather than the front -- three of those folders each have their own
# config.py, and inserting all of them up front here would make the wrong one
# win depending on loop order, exactly the bug this split avoids.
sys.path.insert(0, str(REPO_DIR / "benchmark"))

commit = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, check=True).stdout.strip()
print(f"\\nrunning commit {{commit}}")
''')

code("""
# Everything the benchmark needs beyond what Kaggle ships. bitsandbytes and
# accelerate are for the quantized (tier B) language models.
!pip install -q python-dotenv sentencepiece bitsandbytes accelerate
""")

# ── 3. Imports ───────────────────────────────────────────────────────────
md("""
## 3 — Imports and hardware placement

`tiers.py` decides, for the language models, the highest precision this
hardware can hold: fp16 if it fits, otherwise int8, otherwise 4-bit. A model
placed at anything other than fp16 is tier B — its score includes whatever the
compression cost.
""")

code("""
import warnings
warnings.filterwarnings("ignore")

import bridge, dataset, kaggle_dataset, leaderboard, llm, pipeline
import plan, report_structure, runner, scoring, tiers, transcribe

import pandas as pd

usable_gb, cards = tiers.usable_vram()
print(f"{cards} GPU(s), ~{usable_gb:.1f} GB usable each "
      f"(~{usable_gb * max(cards, 1):.1f} GB if a model is sharded)\\n")

placements = tiers.plan_placements(plan.LLM_PARAMS, usable_gb, cards)
print(f"{'model':18} {'tier':5} {'placement'}")
for p in placements:
    print(f"  {p.model:18} {p.tier:5} {p.describe()}")
""")

# ── 4. HF auth ───────────────────────────────────────────────────────────
md("""
## 4 — Hugging Face authentication

Several checkpoints are gated: HF serves them only to an account that has
accepted the model's licence on its page. Without a token, or without having
accepted the licence, the load fails with *"Cannot access gated repo"* — which
is easy to mistake for a bug here.

Preferred: **Add-ons → Secrets**, a secret named `HF_TOKEN`. It is never saved
in this notebook. Get a token (read scope is enough) at
huggingface.co/settings/tokens.
""")

code('''
HF_TOKEN = ""      # leave empty to use the Kaggle secret

import os
from huggingface_hub import login, model_info, whoami


def resolve_hf_token():
    if not HF_TOKEN:
        try:
            from kaggle_secrets import UserSecretsClient
            secret = UserSecretsClient().get_secret("HF_TOKEN")
            if secret:
                return secret.strip(), "Kaggle secret HF_TOKEN"
        except Exception:
            pass
    if HF_TOKEN:
        return HF_TOKEN.strip(), "the HF_TOKEN variable in this cell"
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip(), f"${name}"
    return None, None


_token, _source = resolve_hf_token()
if _token:
    login(token=_token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"] = _token
    name = whoami().get("name", "unknown")
    print(f"signed in as {name}   (token from {_source})")
else:
    print("No token found. Ungated models still download; gated ones will not.")
    print("Add one under Add-ons -> Secrets as HF_TOKEN, or set HF_TOKEN above.")
''')

code('''
# Checked once, here, instead of discovering each blocked checkpoint hours
# apart inside a run cell.
def check_access(repo_ids):
    reachable, blocked = [], []
    for repo in sorted(set(repo_ids)):
        try:
            model_info(repo, token=_token or None)
            reachable.append(repo)
        except Exception as error:
            text = str(error).lower()
            if "gated" in text or "awaiting" in text:
                reason = "gated -- accept the licence on its model page"
            elif "401" in text or "not found" in text:
                reason = "not found, or private to another account"
            else:
                reason = type(error).__name__
            blocked.append((repo, reason))
    return reachable, blocked


stt_repos = [spec["model_id"] for spec in bridge.model_registry().values()]
llm_repos = [llm._hugging_face_id(key) for key in plan.LLM_PARAMS]

reachable, blocked = check_access(stt_repos + llm_repos)
print(f"{len(reachable)} of {len(stt_repos) + len(llm_repos)} checkpoints reachable")
for repo, reason in blocked:
    print(f"  BLOCKED  {repo}")
    print(f"           {reason}   ->   https://huggingface.co/{repo}")
if blocked:
    print("\\nAccept the licence on each page above, signed in as the same account,")
    print("then re-run this cell.")
''')

# ── 5. Dataset ───────────────────────────────────────────────────────────
md("""
## 5 — Dataset

Searched at any depth under `/kaggle/input`, because how deep `labels.csv`
sits depends on how the dataset was zipped. If this cell says the file was not
found, it prints the mounted tree — paste the path it shows into
`LABELS_OVERRIDE` below and re-run.
""")

code("""
LABELS_OVERRIDE = None   # e.g. "/kaggle/input/spin-buali-dataset/Small_Demo/labels.csv"

try:
    LABELS = kaggle_dataset.resolve(override=LABELS_OVERRIDE)
except FileNotFoundError as error:
    raise SystemExit(str(error))

DATA_DIR = LABELS.parent
labels = pd.read_csv(LABELS)

print(f"labels : {LABELS}")
print(f"folder : {DATA_DIR}")
print(f"audio  : {kaggle_dataset.count_audio(DATA_DIR)} file(s)")
print(f"rows   : {len(labels)}")
labels[["asset_id", "audio", "report"]].head()
""")

code("""
clips = dataset.from_csv(LABELS)
census = dataset.describe(clips)
assert not census["missing_audio"], census["missing_audio"]
print(f"{census['items']} recording(s), {census['labelled']} labelled")

for item in clips:
    audio, sr = dataset.load_audio(item.audio)
    print(f"  {item.asset_id:12} {len(audio) / sr:6.1f}s")
""")

# ── 6. Config ────────────────────────────────────────────────────────────
md("""
## 6 — Run configuration

One place to change the language model, the pipeline, or whether the
benchmark-only report-structure addendum is used, without editing every run
cell below.

**The addendum is not part of `controller/prompts.py`.** It tells the model
the section order this dataset's reference reports use, because this
benchmark scores the combined STT+LLM output against one house template, not
free text -- see `report_structure.py`. Set it to `None` to score the
controller's prompt exactly as production sends it.
""")

code("""
LLM_KEY = "aya-expanse-8b"     # any key in plan.LLM_PARAMS -- see cell 3's placement table
PIPELINE = "separate"          # separate | multimodal | hybrid
LANGUAGE = "fa"
DEVICES = ["cuda:0"]           # STT stays on one card so the LLM has the other free

STRUCTURE_GUIDE = report_structure.GUIDE   # or None to score the bare controller prompt

_placement = next(p for p in placements if p.model == LLM_KEY)
PRECISION, CARDS = _placement.precision, _placement.cards
print(f"{LLM_KEY}: tier {_placement.tier}, {_placement.describe()}")

RESULTS_DIR = pathlib.Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
""")

# ── 7. Runs ──────────────────────────────────────────────────────────────
md("""
## 7 — Runs: the top 5 STT engines, solo

`runner.TOP5_STT` is the five models named in `docs/STT_Models.pdf`, in that
document's own rank order. Each is its own cell.

**To add a run:** copy a cell, change `stt_key` (or pass `None` for
`multimodal`) and `label`. **To try a different LLM or pipeline:** change
`LLM_KEY` / `PIPELINE` in the cell above and re-run — no cell below needs
editing, since they all read those variables.
""")

for index, stt_key in enumerate(
        ["seamless", "seamless-medium", "whisper", "mms-fl102", "whisper-vhdm"], start=1):
    checkpoint_comment = {
        "seamless": "facebook/seamless-m4t-v2-large -- WER 0.107 in the PDF",
        "seamless-medium": "facebook/hf-seamless-m4t-medium -- WER 0.134",
        "whisper": "nezamisafa/whisper-persian-v4 -- WER 0.137",
        "mms-fl102": "facebook/mms-1b-fl102 -- WER 0.146, in-domain/optimistic, see the PDF",
        "whisper-vhdm": "vhdm/whisper-large-fa-v1 -- WER 0.150",
    }[stt_key]
    md(f"### 7.{index} — `{stt_key}`\n\n{checkpoint_comment}")
    code(f'''
df_{index:02d} = runner.run_one(
    "{stt_key}", LLM_KEY, PIPELINE, clips,
    language=LANGUAGE, devices=DEVICES, precision=PRECISION, cards=CARDS,
    structure_guide=STRUCTURE_GUIDE, results_dir=RESULTS_DIR,
    label="{index:02d}_{stt_key}__" + LLM_KEY + "__" + PIPELINE,
)
df_{index:02d}[df_{index:02d}["asset_id"] != "SUMMARY"][
    ["asset_id", "wer", "medical_term_f1", "negation_errors",
     "laterality_errors", "number_errors", "requires_medical_review"]]
''')

# ── 8. Master ────────────────────────────────────────────────────────────
md("""
## 8 — Master results

Merges the SUMMARY row of every `results__*.csv` on disk — whichever cells
above have actually been run — sorted by corpus WER. Safe to run after any
subset of the cells above, and safe to re-run after more of them finish.
""")

code("""
master = runner.build_master(RESULTS_DIR)
master
""")

md("""
**Read it in this order.** Corpus WER is templated-text noise as much as
signal on this dataset — a wrong patient's report can score a better WER than
a correctly reworded one. Negation, laterality, number and unit error rates
are what actually separate a usable configuration from one that changes what
the report means.
""")

# ── 9. Save ──────────────────────────────────────────────────────────────
md("""
## 9 — Save

Every run already wrote its own CSV as its cell finished. This only lists
them. Download the folder before ending the session — Kaggle does not keep
`/kaggle/working` between sessions unless this notebook version is saved with
its output.

To resume later without redoing a finished run: re-attach the downloaded CSVs
into `/kaggle/working/results` (or a fresh Kaggle Dataset) before running
`build_master`, and only re-run the run cells you have not done yet.
""")

code("""
for path in sorted(RESULTS_DIR.glob("*.csv")):
    print(f"  {path.name:52} {path.stat().st_size / 1024:8.1f} KB")
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
    out = pathlib.Path(__file__).parent / "kaggle_dual_t4.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {out}\n  {len(cells)} cells ({code_cells} code), {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
