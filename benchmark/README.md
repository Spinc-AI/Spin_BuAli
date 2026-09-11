# Benchmark

Runs Persian radiology dictations through **speech recognition → language
model** and scores the finished report against the radiologist's signed one,
using the same metrics `evaluation/` uses in production.

It answers one question: **which combination should we ship, and can the
hardware run it?** So every configuration is scored for clinical correctness
*and* measured for cost in the same run.

## The notebook has no logic of its own

`notebooks/kaggle_dual_t4.ipynb` clones this repo and imports from it. It does
not embed, define, or duplicate a single function — every class and metric it
calls lives in a `.py` file here, covered by `tests/`.

That makes fixing a bug a two-step loop instead of a re-upload:

1. Fix the `.py` file in this folder, run the tests, commit, push.
2. On Kaggle, re-run the "get the code" cell (a `git pull`) and re-run
   whichever run cell was affected. The rest of the notebook is untouched.

**One cell is one (STT, LLM, pipeline) run.** Not a loop over several — a
loop would put more than one run behind a single cell, and stopping the
session mid-loop would lose whichever run was in flight. Each cell writes its
own CSV before it finishes, so a session can end after any cell with nothing
lost, and resuming means running the cells not yet done — never redoing the
ones that already wrote their file.

## Files

| File | Responsibility |
|---|---|
| `runner.py` | `run_one()` — the one function every notebook cell calls; `build_master()` merges their CSVs |
| `report_structure.py` | The benchmark-only section-order addendum (see below) |
| `kaggle_dataset.py` | Finds the attached dataset's `labels.csv` at any depth |
| `pipeline.py` | Transcripts → report, using `controller/prompts.py` verbatim |
| `llm.py` | Loads a language model at its tier's precision; cloud models too |
| `transcribe.py` | Windowing, stitching, STT scheduling |
| `dataset.py` | Reads a dataset folder's `labels.csv`, decodes audio |
| `scoring.py` | Calls `evaluation/`'s real metrics — never reimplements them |
| `leaderboard.py` | Flattens a scored report into one CSV row |
| `tiers.py` | Which precision (fp16 / int8 / nf4) this hardware can hold a model at |
| `bridge.py` | **The only file that imports `evaluation/`, `stt/` and `controller/`** |
| `notebooks/build_notebook.py` | Generates the notebook from these sources |

`plan.py`, `ledger.py`, `session.py`, `campaign.py` are a separate, older path:
an automated multi-tier sweep driven from the command line
(`run_benchmark.py plan/work/status/combine`), still tested, still usable for
a large batch run outside a notebook. **The notebook does not use them** —
it calls `runner.run_one()` directly, one cell at a time, by design.

### About `bridge.py`

Every service in this repo is import-isolated and talks to its neighbours
over HTTP. A benchmark cannot be: it needs weights in-process and the metrics
called directly. That coupling is confined to one file. Delete `bridge.py`
and nothing else here knows the other modules exist.

`evaluation/`, `controller/` and `stt/app/` each have their own `config.py`.
`bridge.py` reaches each of them by *appending* its directory to `sys.path`
at the moment it is needed, never by inserting all three up front — insert
all three and whichever the loop processes last silently wins the name,
which is why the notebook's clone cell puts only `benchmark/` on the path
directly and lets `bridge.py` do the rest.

## Run it on Kaggle

Attach the `spin-buali-dataset` dataset (*Add Input*), set **Accelerator → GPU
T4 ×2** and **Internet → On**, then run the cells in order:

1. **Get the code** — clones this repo (or pulls, if already cloned this
   session).
2. **Hugging Face auth** — several checkpoints are gated; this signs in and
   checks which ones this account can actually reach, once, up front.
3. **Dataset** — finds `labels.csv` under `/kaggle/input` at any depth. If it
   is not found, it prints the mounted tree so the real path is visible
   instead of guessed at.
4. **Config** — `LLM_KEY`, `PIPELINE`, `STRUCTURE_GUIDE` in one place; every
   run cell reads these, so trying a different model means editing one cell,
   not five.
5. **Runs** — one cell per STT engine (the top 3 from `docs/STT_Models.pdf`).
   Copy a cell to add another engine, LLM, or pipeline.
6. **Master** — merges every `results__*.csv` present into one sorted table.
   Safe to run after any subset of the run cells.

## The top 3 STT engines

`runner.TOP3_STT`, in the rank order `docs/STT_Models.pdf` measured on FLEURS
fa_ir:

| Key | Checkpoint | WER (PDF) |
|---|---|---|
| `seamless` | facebook/seamless-m4t-v2-large | 0.107 |
| `seamless-medium` | facebook/hf-seamless-m4t-medium | 0.134 |
| `whisper` | nezamisafa/whisper-persian-v4 | 0.137 |

The other seven registered models (mms-fl102, whisper-vhdm, stock Whisper
below large-v3, wav2vec2, mms-1b-all, whisper-halakoo, whisper-large-v3,
whisper-large-v3-turbo) are not run here because that PDF already showed they
lose on this language.

## The report-structure addendum

**Not part of `controller/prompts.py`.** The reference reports in this
dataset all follow one house template — every one of them states the same
organs in the same order, and two sentences appear close to verbatim in all
nine. This benchmark measures the *combined* STT+LLM output against that
fixed structure, so `report_structure.GUIDE` tells the model what order to
use, appended after the controller's own JSON template.

It is opt-in (`STRUCTURE_GUIDE = None` scores the bare controller prompt) and
it is additive: `pipeline.build_report`'s test suite asserts the controller's
own prompt text always appears first and unmodified, whether or not a guide
is attached.

Keeping it out of `controller/prompts.py` is deliberate: that file is
production's prompt, used for every customer's reports, most of which do not
share this one's template. Baking a Kaggle dataset's house style into it
would fix this benchmark and quietly bias production.

## Output, per run

```
results__<label>.csv    one row per recording, plus a trailing SUMMARY row
results_master.csv      every run's SUMMARY row, merged, sorted by corpus WER
```

Mirrors the shape of the FLEURS benchmark's older per-model CSVs: per-clip
rows first, one aggregate row appended last, so a spreadsheet opened cold
shows both without a second file.

**Read the master table in this order**, not by `corpus_wer` alone. These
reference reports are templated enough that a *different patient's* report
can score a better WER than a correctly reworded one, and flipping left for
right — the error that sends a surgeon to the wrong kidney — barely moves it.
Negation, laterality, number and unit error rates are what actually separate
a usable configuration from one that changes what the report means.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Nothing loads weights or touches a GPU — models are injected, so `runner.py`,
`pipeline.py`, `llm.py` and the generated notebook are all tested on CPU in a
few seconds. `tests/test_notebook.py` runs the notebook's own cells against a
live checkout of this repo and a simulated Kaggle mount, dry-run models
standing in for real ones — it proves the wiring, not the models.
