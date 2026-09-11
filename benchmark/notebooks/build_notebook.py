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

REPO_URL = "https://github.com/Spinc-AI/Spin_BuAli.git"
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

**Typed live, not written into this notebook.** This cell prompts for the
token with a masked field — nothing is echoed back, and nothing here prints
it — so it never ends up in this cell's source or its saved output, even if
the notebook is public. Re-run this cell each session; there is nothing to
carry over because nothing was saved.

If this notebook is your own and stays private, **Add-ons → Secrets** (a
secret named `HF_TOKEN`) is one step less per session — this cell tries that
first and only prompts if no secret is set. A public notebook should rely on
the prompt, not a secret attached to the notebook.

Get a token (read scope is enough) at huggingface.co/settings/tokens.
""")

code('''
import getpass
import os
from huggingface_hub import login, model_info, whoami


def resolve_hf_token():
    """A Kaggle secret if one is set, otherwise a live masked prompt.

    Neither path writes the token anywhere this notebook file can carry it:
    a secret lives in Kaggle's own store, not in the notebook, and getpass
    does not echo what is typed and nothing here prints it back.
    """
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
    name = whoami().get("name", "unknown")
    print(f"signed in as {name}   (token from {_source})")
else:
    print("No token given. Ungated models still download; gated ones will not.")
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

Shared settings every run cell below reads -- windowing, the report-structure
addendum, the token cap, and where results land. The STT and LLM rosters
themselves are fixed (`runner.TOP3_STT`, `runner.TOP3_LLM`,
`runner.MULTIMODAL_LLM`); see cell 7 for why each list holds what it holds.

**The addendum is not part of `controller/prompts.py`.** It tells the model
the section order this dataset's reference reports use, because this
benchmark scores the combined STT+LLM output against one house template, not
free text -- see `report_structure.py`. Set it to `None` to score the
controller's prompt exactly as production sends it.
""")

code("""
LANGUAGE = "fa"
DEVICES = ["cuda:0"]           # STT stays on one card so the LLM has the other free

# How the recording is windowed before it reaches the STT model. adaptive
# listens to the audio and snaps cuts to quiet moments; the -vad variants
# chunk within detected speech regions so a long pause becomes a boundary
# instead of something a window spends itself on. None means fixed windows.
PREPROCESSING = "adaptive"     # None | "fixed" | "uniform" | "adaptive" | "adaptive-vad"
print("preprocessing options:", {**plan.PREPROCESSING, None: "fixed windows, no chunking module"})

STRUCTURE_GUIDE = report_structure.GUIDE   # or None to score the bare controller prompt

# The `separate` pipeline asks for three full fields (raw_transcript,
# corrected_transcript, final_text) -- easy to overrun the 1536-token default
# on a real report. A generation cut off mid-JSON shows up as "no JSON object
# found" or a JSONDecodeError, and only on the longer clips, since the model
# never reached the closing brace. Raise this if a run shows that pattern.
MAX_NEW_TOKENS = None          # None uses settings.LLM_MAX_NEW_TOKENS (1536); try 3072 if truncating

RESULTS_DIR = pathlib.Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def placement_for(llm_key):
    \"\"\"This LLM's precision/card count from cell 3's placement table --
    looked up per run cell instead of once, since every cell here can name a
    different LLM.\"\"\"
    p = next(p for p in placements if p.model == llm_key)
    return p.precision, p.cards
""")

# ── 7. Runs ──────────────────────────────────────────────────────────────
md(f"""
## 7 — Runs: 3 STT engines x 3 LLMs (`separate`), plus 3 LLMs (`multimodal`)

Two rosters, picked for a reason:

* **`runner.TOP3_STT`** -- the three lowest-WER engines in `docs/STT_Models.pdf`.
* **`runner.TOP3_LLM`** -- the three *lightest* LLMs by parameter count, used
  for the `separate` pipeline (text only, so audio capability doesn't matter):
  {", ".join(f"`{k}`" for k in ["medgemma-1.5-4b", "phi-4-multimodal", "gemma-4-e4b"])}.
* **`runner.MULTIMODAL_LLM`** -- the three lightest **audio-capable** LLMs,
  used for `multimodal` (the LLM hears the recording directly, so a text-only
  model like `medgemma-1.5-4b` cannot run here at all --
  `gemma-4-12b` takes its place):
  {", ".join(f"`{k}`" for k in ["phi-4-multimodal", "gemma-4-e4b", "gemma-4-12b"])}.

3 STT x 3 LLM = 9 `separate` runs, + 3 `multimodal` runs (one per audio-capable
LLM, no STT stage) = **12 runs, 12 cells**.

**To add a run:** copy a cell and change its `stt_key`/`llm_key`/`pipeline`/
`label`. Every cell is independent -- stopping the session after any of them
loses nothing.

**Speech recognition is cached.** Running several LLMs against the same STT
engine transcribes once -- the cache key is `(preprocessing, stt_key)` only,
so it doesn't care which LLM or pipeline asked for it. A cell that reused a
cached transcript prints `-- cached, skipping transcription` and its CSV's
`stt_cached` column is `True`. The cache lives in `RESULTS_DIR/transcripts/`;
delete a file there to force that one pair to be redone, or pass
`use_cache=False` to force a cell to redo it regardless.
""")

_stt_comment = {
    "seamless": "facebook/seamless-m4t-v2-large -- WER 0.107 in the PDF",
    "seamless-medium": "facebook/hf-seamless-m4t-medium -- WER 0.134",
    "whisper": "nezamisafa/whisper-persian-v4 -- WER 0.137",
}
_llm_comment = {
    "medgemma-1.5-4b": "google/medgemma-1.5-4b-it -- 4.3B, text only",
    "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct -- 5.6B, audio-capable",
    "gemma-4-e4b": "google/gemma-4-E4B-it -- 7.85B, audio-capable",
    "gemma-4-12b": "google/gemma-4-12B-it -- 12B, audio-capable",
}

_top3_stt = ["seamless", "seamless-medium", "whisper"]
_top3_llm = ["medgemma-1.5-4b", "phi-4-multimodal", "gemma-4-e4b"]
_multimodal_llm = ["phi-4-multimodal", "gemma-4-e4b", "gemma-4-12b"]

_index = 0
for stt_key in _top3_stt:
    for llm_key in _top3_llm:
        _index += 1
        md(f"### 7.{_index} — `separate`: `{stt_key}` + `{llm_key}`\n\n"
           f"{_stt_comment[stt_key]}  \n{_llm_comment[llm_key]}")
        code(f'''
PRECISION, CARDS = placement_for("{llm_key}")
df_{_index:02d} = runner.run_one(
    "{stt_key}", "{llm_key}", "separate", clips,
    language=LANGUAGE, devices=DEVICES, precision=PRECISION, cards=CARDS,
    preprocessing=PREPROCESSING, structure_guide=STRUCTURE_GUIDE, results_dir=RESULTS_DIR,
    max_new_tokens=MAX_NEW_TOKENS,
    label="{_index:02d}_{stt_key}__{llm_key}__separate__" + str(PREPROCESSING),
)
df_{_index:02d}[df_{_index:02d}["asset_id"] != "SUMMARY"][
    ["asset_id", "wer", "medical_term_f1", "negation_errors",
     "laterality_errors", "number_errors", "requires_medical_review"]]
''')

for llm_key in _multimodal_llm:
    _index += 1
    md(f"### 7.{_index} — `multimodal`: `{llm_key}` (no STT stage)\n\n{_llm_comment[llm_key]}")
    code(f'''
PRECISION, CARDS = placement_for("{llm_key}")
df_{_index:02d} = runner.run_one(
    None, "{llm_key}", "multimodal", clips,
    language=LANGUAGE, devices=DEVICES, precision=PRECISION, cards=CARDS,
    preprocessing=PREPROCESSING, structure_guide=STRUCTURE_GUIDE, results_dir=RESULTS_DIR,
    max_new_tokens=MAX_NEW_TOKENS,
    label="{_index:02d}_multimodal__{llm_key}__" + str(PREPROCESSING),
)
df_{_index:02d}[df_{_index:02d}["asset_id"] != "SUMMARY"][
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
