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
    ("stt/app/config.py", "stt/app/config.py", "stt"),
    ("stt/app/model.py", "stt/app/model.py", "stt"),
    ("benchmark/settings.py", "benchmark/settings.py", "benchmark"),
    ("benchmark/bridge.py", "benchmark/bridge.py", "benchmark"),
    ("benchmark/dataset.py", "benchmark/dataset.py", "benchmark"),
    ("benchmark/transcribe.py", "benchmark/transcribe.py", "benchmark"),
    ("benchmark/scoring.py", "benchmark/scoring.py", "benchmark"),
    ("benchmark/leaderboard.py", "benchmark/leaderboard.py", "benchmark"),
    ("benchmark/run_benchmark.py", "benchmark/run_benchmark.py", "benchmark"),
]

SECTION_NOTES = {
    "evaluation": """### The metrics

Straight from `evaluation/` — the same functions the live scoring service runs.
Not a reimplementation: if a metric changes there, this notebook changes with
it, so the benchmark can never quietly disagree with production about which
model is better.""",
    "stt": """### The models

From `stt/app/` — one class per architecture, and a registry naming every
model. Adding a model is one registry entry; nothing below names a model.""",
    "benchmark": """### The harness

From `benchmark/` — windowing, per-GPU scheduling, scoring and the leaderboard.""",
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
import bridge, dataset, leaderboard, run_benchmark, scoring, transcribe

print("modules loaded")
print("metrics version:", bridge.METRICS_VERSION)
print("models available:")
for key, spec in bridge.model_registry().items():
    print(f"   {key:24} {spec['model_id']}")
""")

md("""
## 3. Configure the run

`AUDIO_DIR` holds the recordings. `TRUTH_DIR` holds `<same stem>.txt` ground
truth files — leave it `None` for a first pass over unlabelled audio, which
still produces transcripts and timings.

`MODELS` are keys from the registry printed above.
""")

code("""
AUDIO_DIR = "/kaggle/input/buali-audio/audio"     # <- your recordings
TRUTH_DIR = "/kaggle/input/buali-audio/truth"     # <- or None if unlabelled
MODELS    = ["whisper", "whisper-large-v3-turbo", "seamless"]
LANGUAGE  = "fa"

# /kaggle/working is the writable half of the runtime, and the only directory
# Kaggle keeps when the session ends.
OUT_DIR   = pathlib.Path("/kaggle/working/benchmark_results")

# Whisper's encoder is fixed at 30 seconds; every model gets the same windows so
# a long-audio penalty is never mistaken for a model being worse.
WINDOW_SEC, OVERLAP_SEC = 28.0, 3.0
""")

md("## 4. Load the dataset")

code("""
items = dataset.from_directory(AUDIO_DIR, TRUTH_DIR)
census = dataset.describe(items)
census
""")

code("""
assert items, "no audio found — check AUDIO_DIR"
assert not census["missing_audio"], census["missing_audio"][:5]

# Prove the results can be written before spending the GPU time, not after.
leaderboard.check_writable(OUT_DIR)

for item in items[:5]:
    audio, sr = dataset.load_audio(item.audio)
    print(f"  {item.asset_id:20} {len(audio)/sr:7.1f}s  labelled={item.labelled}")
""")

md("""
### Dry run first

A stub model, no weights. Exercises decoding, windowing, stitching, scoring and
writing in seconds — so a wrong path costs you that, and not a 10 GB download
followed by a failure.
""")

code("""
_, dry_results, _ = run_benchmark.run(
    items[:3], models=["dry-run"], model_factory=transcribe.dry_run_factory,
    window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC)
print(f"plumbing OK — {len(dry_results)} labelled recording(s) scored end to end")
""")

md("""
## 5. Run

One model at a time, replicated across both GPUs with the batch split between
them: throughput doubles and no number changes. Peak memory stays one model's
worth however many are being compared.

Roughly *(audio hours) × (models) × RTF*. Progress prints per recording.
""")

code("""
for device in transcribe.describe_devices():
    print(f"  {device['device']:9} {device['name']} ({device['total_vram_gb']} GB)")

runs, results, summary = run_benchmark.run(
    items,
    models=MODELS,
    language=LANGUAGE,
    window_sec=WINDOW_SEC,
    overlap_sec=OVERLAP_SEC,
    on_progress=lambda t: print(f"  [{t.model}] {t.asset_id}: "
                                f"{t.error or f'{t.real_time_factor:.2f}x real time'}",
                                flush=True),
)
print("\\ndone")
""")

md("""
## 6. Optional — the two embedding metrics

BERTScore and semantic similarity load a second model, so they are off by
default. This adds them for the whole batch in one pass; skip it if you only
need WER and the clinical metrics.
""")

code("""
# !pip install -q bert-score sentence-transformers   # uncomment on first run

usable, reason = bridge.semantic_available()
print("semantic extras:", "available" if usable else f"not installed — {reason}")

if usable:
    scoring.add_semantic(results, summary)
    print("added:", list(summary["models"][0]["semantic"]))
""")

md("""
## 7. The batch leaderboard

One row per model over the whole batch, accuracy and cost side by side on
purpose: a model that wins on WER while running at four times real time on a T4
has not won anything deployable.

**`WER` is the corpus WER** — total edits over total reference words — not the
mean of the per-report WERs, which would weight a one-line finding exactly like
a multi-minute study. `p50`/`p90` beside it describe the spread.

`latin` / `latin(ref)` are script contamination: the share of letters that are
not Arabic-script, in the transcript and in the ground truth. **Read the gap,
not the number** — this corpus code-switches English radiology terms on purpose,
so a correct transcript is "contaminated" too.
""")

code("""
import pandas as pd

board = leaderboard.to_dataframe(summary)
board[["model", "reports", "corpus_wer", "corpus_cer", "wer_p50", "wer_p90",
       "chrf_mean", "medical_term_f1", "negation_error_rate",
       "laterality_error_rate", "number_error_rate", "unit_error_rate",
       "critical_omission_rate", "unsupported_addition_rate",
       "repetition_score_p95", "script_contamination_mean",
       "reference_script_contamination_mean", "insertion_rate", "review_rate",
       "pct_catastrophic", "real_time_factor", "throughput", "peak_vram_gb"]]
""")

code("""
from IPython.display import Markdown

Markdown(leaderboard.to_markdown(summary))
""")

md("""
### WER by recording length

Long audio is where windowing and stitching can go wrong, and where a model with
a short attention span quietly degrades. One overall WER hides both.
""")

code("""
pd.DataFrame({
    bucket["model"]: {name: figures["wer"]
                      for name, figures in bucket["by_duration"].items()}
    for bucket in summary["models"]
}).T
""")

code("""
# Everything summarize() produced for one model. The table above is a readable
# subset; this is the full set.
import json

print(json.dumps(summary["models"][0], ensure_ascii=False, indent=2)[:3000])
""")

md("""
## 8. What went wrong

An aggregate says a model is 12% wrong; it never says *which* 12%. These are the
recordings to read before trusting any of the numbers above.
""")

code("""
pd.DataFrame(leaderboard.worst(results, count=15))
""")

code("""
from collections import Counter

flagged = [r for r in results if r["requires_medical_review"]]
print(f"{len(flagged)} of {len(results)} scored reports need review\\n")
Counter(reason for r in flagged for reason in r["review_reasons"]).most_common()
""")

md("""
## 9. Read one recording side by side

Set `ASSET` to anything from the table above. Critical errors are the ones that
change what a report means — a flipped negation, a wrong side, a wrong number.
""")

code("""
assert results, "nothing was scored — this run had no ground truth"
ASSET = results[0]["asset_id"]

reference = next(i.reference for i in items if i.asset_id == ASSET)
print(f"REFERENCE\\n{reference}\\n")

for result in [r for r in results if r["asset_id"] == ASSET]:
    said = next(t.text for run in runs for t in run.transcripts
                if t.asset_id == ASSET and t.model == result["model"])
    print(f"--- {result['model']}  WER={result['general']['wer']:.3f} "
          f"F1={result['clinical_metrics']['medical_term_f1']:.3f}")
    print(said)
    for error in result["critical_errors"]:
        print(f"    ! {error['type']}: {error['reference']} -> {error['prediction']}")
    print()
""")

md("## 10. Save")

code("""
leaderboard.write(OUT_DIR, results, summary, runs)

for path in sorted(OUT_DIR.iterdir()):
    print(f"  {path.name:20} {path.stat().st_size / 1024:9.1f} KB")
""")

md("""
| File | What it is |
|---|---|
| `leaderboard.csv` | one row per model — the table above |
| `leaderboard.md` | the same, as markdown |
| `per_report.csv` | one row per recording per model, with reference and hypothesis |
| `summary.json` | the full nested batch summary |
| `results.json` | the full per-report scores |
| `transcripts.json` | what every model said, including unlabelled recordings |

Both CSVs are UTF-8 with a BOM, so Excel reads the Persian correctly.

`transcripts.json` covers recordings with no ground truth too — those drafts are
the starting point for the next labelling round.
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
