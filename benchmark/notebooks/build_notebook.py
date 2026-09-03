"""Generate the self-contained Kaggle notebook from the module sources.

    python build_notebook.py

The notebook it writes needs nothing but itself: no repo to clone, no dataset
to attach, no `sys.path` pointing anywhere. Paste it into Kaggle and run.

It gets there by *embedding* the modules rather than duplicating them. Each
source file becomes one cell that writes it back to disk in the same three-folder
layout the repo uses, and the notebook then imports it unchanged. So the code
that runs on Kaggle is character-for-character the code the tests cover -- the
notebook is a build artifact, not a second implementation.

Re-run this after changing anything under `benchmark/`, `evaluation/` or
`stt/app/`. `tests/test_notebook.py` fails if you forget.
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "kaggle_dual_t4.ipynb"

# (source in the repo, destination inside the reconstructed tree, section).
# The destinations mirror the repo exactly, which is what lets every module keep
# its own imports: `settings.REPO_ROOT` resolves to the reconstructed root, and
# `bridge.py` finds `evaluation/` and `stt/` beside it just as it does locally.
SOURCES = [
    ("evaluation/config.py", "evaluation/config.py", "evaluation"),
    ("evaluation/text_normalizer.py", "evaluation/text_normalizer.py", "evaluation"),
    ("evaluation/extractors.py", "evaluation/extractors.py", "evaluation"),
    ("evaluation/general_metrics.py", "evaluation/general_metrics.py", "evaluation"),
    ("evaluation/semantic_metrics.py", "evaluation/semantic_metrics.py", "evaluation"),
    ("evaluation/medical_metrics.py", "evaluation/medical_metrics.py", "evaluation"),
    ("evaluation/evaluate_results.py", "evaluation/evaluate_results.py", "evaluation"),
    ("evaluation/clinical_terms.json", "evaluation/clinical_terms.json", "evaluation"),
    ("controller/prompts.py", "controller/prompts.py", "controller"),
    ("controller/report_template.json", "controller/report_template.json", "controller"),
    ("stt/app/config.py", "stt/app/config.py", "stt"),
    ("stt/app/model.py", "stt/app/model.py", "stt"),
    ("benchmark/settings.py", "benchmark/settings.py", "benchmark"),
    ("benchmark/bridge.py", "benchmark/bridge.py", "benchmark"),
    ("benchmark/dataset.py", "benchmark/dataset.py", "benchmark"),
    ("benchmark/transcribe.py", "benchmark/transcribe.py", "benchmark"),
    ("benchmark/scoring.py", "benchmark/scoring.py", "benchmark"),
    ("benchmark/leaderboard.py", "benchmark/leaderboard.py", "benchmark"),
    ("benchmark/llm.py", "benchmark/llm.py", "benchmark"),
    ("benchmark/pipeline.py", "benchmark/pipeline.py", "benchmark"),
    ("benchmark/tiers.py", "benchmark/tiers.py", "benchmark"),
    ("benchmark/ledger.py", "benchmark/ledger.py", "benchmark"),
    ("benchmark/plan.py", "benchmark/plan.py", "benchmark"),
    ("benchmark/session.py", "benchmark/session.py", "benchmark"),
    ("benchmark/campaign.py", "benchmark/campaign.py", "benchmark"),
    ("benchmark/run_benchmark.py", "benchmark/run_benchmark.py", "benchmark"),
]

SECTION_NOTES = {
    "evaluation": """### The metrics

Straight from `evaluation/` — the same functions the live scoring service runs.
Not a reimplementation: if a metric changes there, this notebook changes with
it, so the benchmark can never quietly disagree with production about which
model is better.""",
    "controller": """### The prompts

From `controller/prompts.py`, unchanged. The prompt *is* the experiment — a
benchmark that phrased the instruction its own way would be ranking a system
nobody ships, and the difference would not show up anywhere in the results.""",
    "stt": """### The models

From `stt/app/` — one class per architecture, and a registry naming every
model. Adding a model is one registry entry; nothing below names a model.""",
    "benchmark": """### The harness

From `benchmark/` — windowing, per-GPU scheduling, scoring, and the campaign
machinery: which models this hardware can hold, and how a stopped session picks
up where it left off.""",
}

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


def embed(source_path, destination):
    """One cell that writes one module back to disk, verbatim.

    A raw triple-quoted string keeps the source readable in the notebook -- you
    can see what is about to run, which is the whole point of shipping it as one
    file. Single quotes, not double: every module here opens with a \"\"\"
    docstring. `check_sources()` enforces that nothing in the tree breaks it.
    """
    text = (REPO / source_path).read_text(encoding="utf-8")
    code("write(\"%s\", r'''\n%s''')" % (destination, text))


def check_sources():
    """Refuse to build something that would be silently mis-embedded.

    The delimiter is r'''...''' precisely because every module in this project
    opens with a \"\"\" docstring: embedding with double quotes ends the raw
    string at the first one and spills source into the notebook as live code.
    """
    for source_path, _, _ in SOURCES:
        text = (REPO / source_path).read_text(encoding="utf-8")
        if "'''" in text:
            raise SystemExit(f"{source_path}: contains ''', which would end the embedding")
        if text.rstrip("\n").endswith("\\"):
            raise SystemExit(f"{source_path}: ends with a backslash, breaks a raw string")


# --- The notebook ----------------------------------------------------------
md("""
# Spin BuAli — STT benchmark (Kaggle, dual T4)

Runs a batch of radiology dictations through several Persian STT models and
ranks them on accuracy, clinical correctness, speed and memory.

**Self-contained.** No repo to clone, no dataset to attach. Set the accelerator
to **GPU T4 ×2** and Internet to **On** (the model weights come from Hugging
Face), then Run All.

**Before you edit a cell in section 2:** those cells are generated from the
project's source files and are covered by its test suite. Change the repo and
re-run `benchmark/notebooks/build_notebook.py`, rather than patching here — an
edit made in the notebook is lost on the next regeneration.
""")

md("## 1. Check the hardware")

code("""
!nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
""")

code("""
import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.1f} GB")

assert torch.cuda.device_count() >= 1, "no GPU — set Accelerator to GPU T4 x2"
if torch.cuda.device_count() == 1:
    print("\\nonly one GPU visible: the run will work but take about twice as long")
""")

code("""
# Kaggle already ships torch, transformers, soundfile and pandas. This adds only
# what the benchmark needs on top, and stays quiet when they are present.
!pip install -q python-dotenv sentencepiece
""")

md("""
## 2. The code

The next cells write the benchmark's modules to disk and put them on the import
path. They are the project's real source files, embedded — read them, but change
them in the repo.
""")

code("""
import pathlib, sys

SRC = pathlib.Path("/kaggle/working/buali_src")
if not pathlib.Path("/kaggle/working").is_dir():
    SRC = pathlib.Path.cwd() / "buali_src"   # so the notebook also runs locally


def write(relative_path, text):
    path = SRC / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# stt/ is a package; the other two are plain directories of modules.
(SRC / "stt" / "app").mkdir(parents=True, exist_ok=True)
(SRC / "stt" / "app" / "__init__.py").write_text("", encoding="utf-8")

# Only benchmark/ goes on the path here. It reaches the other two itself, the
# same way it does in the repo — one place decides, and it is a source file.
sys.path.insert(0, str(SRC / "benchmark"))
print("source tree:", SRC)
""")

check_sources()

current_section = None
for source_path, destination, section in SOURCES:
    if section != current_section:
        md(SECTION_NOTES[section])
        current_section = section
    embed(source_path, destination)

code("""
import bridge, dataset, leaderboard, ledger, llm, scoring, session, transcribe

print("modules loaded")
print("metrics version:", bridge.METRICS_VERSION)
print("models available:")
for key, spec in bridge.model_registry().items():
    print(f"   {key:24} {spec['model_id']}")
""")

md("""
## 3. Point at the dataset

The dataset is its own Kaggle Dataset — attach it with **Add Input**, and it
mounts read-only under `/kaggle/input/`. The cell below finds it by looking for
a `labels.csv`, so the slug does not have to match.

Its layout, one folder per dataset:

```
Small_Demo/
├── DPM89130.MP3 ...     the dictations
└── labels.csv           asset_id, audio, image, report
```
""")

code("""
def find_labels():
    \"\"\"The dataset's labels.csv, wherever Kaggle mounted it.\"\"\"
    roots = [pathlib.Path("/kaggle/input"), pathlib.Path.cwd(), REPO.parent]
    for root in roots:
        if not root.exists():
            continue
        for found in sorted(root.glob("*/*/labels.csv")) + sorted(root.glob("*/labels.csv")):
            return found
    raise SystemExit("no labels.csv found — attach the dataset with Add Input")


LABELS = find_labels()
print("dataset:", LABELS.parent)

# /kaggle/working is the writable half of the runtime, and the only directory
# Kaggle keeps when the session ends. Results must go here.
OUT_DIR = pathlib.Path("/kaggle/working/benchmark_results")
""")

md("## 4. Load it")

code("""
items = dataset.from_csv(LABELS)
census = dataset.describe(items)

assert items, f"no recordings in {LABELS}"
assert not census["missing_audio"], census["missing_audio"][:5]
leaderboard.check_writable(OUT_DIR)   # prove it before spending the GPU time

for item in items[:5]:
    audio, sr = dataset.load_audio(item.audio)
    print(f"  {item.asset_id:20} {len(audio)/sr:7.1f}s  labelled={item.labelled}")
census
""")

md("""
## 5. Plan the campaign

Every configuration is written down before anything runs, and each model is
placed in the highest precision these cards can hold:

| Tier | | |
|---|---|---|
| **A** | native | fits unquantized — the number means what it says |
| **B** | quantized | only as far as it had to be, int8 before nf4 |
| **C** | deferred | does not fit here; planned, not run |

Tier C is the *same models* as tier B at full precision. It stays in the plan so
the gap is visible: run one on hardware that can hold it and the difference
against tier B is the quantization penalty.

Trim the lists below for a first pass — the full matrix is thousands of runs.
""")

code("""
import campaign, plan as plan_module, tiers

usable, cards = tiers.usable_vram()
print(f"{cards} GPU(s), ~{usable:.1f} GB usable each")
print(f"~{usable * cards:.1f} GB if a model is sharded across both")

runs = plan_module.build(
    stt_models=["whisper", "whisper-large-v3-turbo", "seamless"],
    llm_models=["aya-expanse-8b"],          # add more once one has worked
    usable_gb=usable, cards=cards,
    pipelines=("separate",),
    preprocessing=("adaptive",),
    max_slots=1,
    cloud=[],                                # e.g. ["gemini:gemini-2.5-pro"]
)

ledger.write_json(OUT_DIR / "plan.json", runs)
ledger.write_csv(OUT_DIR / "plan.csv", plan_module.to_rows(runs))
print(plan_module.summarize(runs))
for run in runs[:8]:
    print(f"  {run['tier']}  {run['preprocessing']:13} {run['pipeline']:11} "
          f"stt={'+'.join(run['stt_models']) or '-':24} llm={run['llm_model']:18} "
          f"{run['placement']}")
""")

md("""
### Dry run first

Stub models, no weights. Exercises decoding, windowing, stitching, the LLM
stage, scoring and the CSV writing in seconds — so a wrong path costs you that,
not a 10 GB download followed by a failure.
""")

code("""
import session

dry = session.work_through(
    runs[:2], items, OUT_DIR / "_dryrun",
    model_factory=transcribe.dry_run_factory,
    llm_factory=llm.dry_run_factory)
print("plumbing OK —", len(dry["performed"]), "run(s) written end to end")
""")

md("""
## 6. Work through it

**Stop this session whenever you like.** A run is finished when its CSV exists,
so nothing in flight is ever lost — there is nothing in flight. Run this cell
again next session and it continues from what is on disk.

`max_minutes` stops *between* runs on purpose: better to leave forty minutes
unused than start a run Kaggle kills at minute thirty-nine having written
nothing. Kaggle's hard cap is twelve hours.
""")

code("""
report = session.work_through(
    runs, items, OUT_DIR,
    tier="A",                       # finish A before starting B
    budget=ledger.Budget(minutes=300),
    on_run=lambda run, outcome: print(
        f"  [{outcome['status']:7}] {run['run_id']}  {run['tier']}  "
        f"{run['preprocessing']:13} {'+'.join(run['stt_models']) or '-':22} "
        f"{outcome.get('seconds', 0):.0f}s", flush=True),
)

print()
print(f"{len(report['performed'])} run(s) this session, "
      f"{report['remaining']} still pending")
if report["stopped_because"]:
    print("stopped:", report["stopped_because"])
report["status"]
""")

md("""
## 7. Combine

Rebuilds the leaderboard from whatever is on disk — so it always reflects the
runs that actually completed, across however many sessions it took.
""")

code("""
print(session.combine(OUT_DIR))

import pandas as pd

board = pd.read_csv(OUT_DIR / "leaderboard.csv")
board
""")



md("""
## 8. Read the results

The leaderboard is one row per run. **`WER` is the corpus WER** — total edits
over total reference words — not the mean of the per-report WERs, which would
weight a one-line finding like a multi-minute study.

Do not rank on it. These reports are templated enough that a *different
patient's* report can score a better WER than a correctly reworded one, and
flipping left for right moves it by 0.009. Rank on the clinical columns:
negation, laterality, number, unit, critical omissions, term F1.
""")

code("""
import json

columns = [c for c in ["model", "tier", "precision", "preprocessing", "pipeline",
                       "reports", "corpus_wer", "corpus_cer", "chrf_mean",
                       "medical_term_f1", "negation_error_rate",
                       "laterality_error_rate", "number_error_rate",
                       "unit_error_rate", "critical_omission_rate",
                       "repetition_score_p95", "review_rate", "pct_catastrophic"]
           if c in board.columns]
board[columns].sort_values("corpus_wer")
""")

md("""
## 9. What went wrong

An aggregate says a configuration is 12% wrong; it never says *which* 12%.
These are the recordings to read before trusting any of the numbers above.
""")

code("""
reports = pd.read_csv(OUT_DIR / "all_reports.csv")
print(f"{len(reports)} scored report(s) across "
      f"{reports['run_id'].nunique()} run(s)")
print()

worst = reports.sort_values("wer", ascending=False).head(15)
worst[[c for c in ["asset_id", "run_id", "llm_model", "stt_models", "wer",
                   "medical_term_f1", "n_laterality_errors", "review_reasons"]
       if c in worst.columns]]
""")

code("""
from collections import Counter

flagged = reports[reports["requires_medical_review"].astype(str).str.lower() == "true"]
print(f"{len(flagged)} of {len(reports)} reports need a human look")
print()
Counter(reason for reasons in flagged.get("review_reasons", [])
        for reason in str(reasons).split(";") if reason).most_common()
""")

md("""
## 10. Read one recording side by side

Critical errors are the ones that change what a report means — a flipped
negation, a wrong side, a wrong number.
""")

code("""
ASSET = reports.iloc[0]["asset_id"]
rows = reports[reports["asset_id"] == ASSET]

print("REFERENCE")
print(rows.iloc[0]["reference"])
print()
for _, row in rows.iterrows():
    print(f"--- {row.get('llm_model')} / {row.get('stt_models')}  "
          f"WER={row['wer']:.3f}  F1={row.get('medical_term_f1')}")
    print(row["hypothesis"])
    print()
""")

md("""
## 11. Save

Everything is already on disk — each run wrote its CSV as it finished. This
just lists it.
""")

code("""
for path in sorted(OUT_DIR.rglob("*")):
    if path.is_file():
        print(f"  {str(path.relative_to(OUT_DIR)):44} {path.stat().st_size / 1024:8.1f} KB")
""")

md("""
| File | What it is |
|---|---|
| `plan.csv` | every configuration, its tier and placement |
| `runs/<run_id>.csv` | one run's per-recording scores — written as it finished |
| `summaries/<run_id>.json` | that run's aggregate, plus the configuration |
| `leaderboard.csv` | one row per completed run |
| `all_reports.csv` | every scored report from every run |
| `transcripts/` | the STT cache — reused by later runs, so keep it |

Download the folder from the output panel before the session ends. Next session:
attach the dataset again, re-run the cells, and section 6 picks up from the runs
already recorded — **keep `transcripts/` and the STT stage is skipped entirely.**
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
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"wrote {OUT} — {len(cells)} cells, {len(SOURCES)} embedded files, {size:.0f} KB")


if __name__ == "__main__":
    main()
