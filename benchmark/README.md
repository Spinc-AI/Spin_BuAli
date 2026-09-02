# Benchmark

Runs a batch of recordings through several STT models and ranks them with the
same metrics the live evaluation service uses.

It answers one question: **which model should we ship, and can the hardware run
it?** So every model is scored for accuracy *and* measured for speed and memory
in the same run — a model that wins on word error rate while running at four
times real time on a T4 has not won anything a clinic can deploy.

Unlike the five services, this is offline tooling. It is not a server, has no
port, and is never in the request path.

## What it does

```
recordings ──▶ decode ──▶ 28s windows ──▶ model ──▶ stitch ──▶ transcript
                                            │                      │
                                    speed / VRAM              evaluation/
                                            │                      │
                                            └──────▶ leaderboard ◀─┘
```

1. **Pair** each recording with its ground truth (or note that it has none yet).
2. **Window** long audio into overlapping 28-second slices.
3. **Transcribe** with one model at a time, replicated across every GPU.
4. **Score** each transcript by calling `evaluation/`.
5. **Rank** the models, and list the recordings that went worst.

## Why it looks like this

**Windowing is not optional.** Whisper's encoder takes a fixed 30-second
window. Hand it a four-minute dictation and it transcribes the first thirty
seconds and stops — which shows up as a catastrophic word error rate that looks
like a bad model and is actually a bad harness. Every model gets the same
windows, so the comparison stays about the models.

**One model at a time, one replica per GPU.** Both T4s run the same model over
different halves of the batch. Throughput doubles and no number changes. The
alternative — a different model on each GPU — would have them competing for
bandwidth and make every latency figure meaningless.

**No metric is implemented here.** Everything comes from
[`evaluation/`](../evaluation/README.md). A benchmark with its own WER would
eventually disagree with production about which model is better, with no way to
tell which one was wrong.

**A failure is a result.** A model that OOMs on one recording gets that
recording scored as an empty output — not dropped. Dropping it would flatter
the model that crashed.

## Files

| File | Responsibility |
|---|---|
| `dataset.py` | Find the recordings, pair them with ground truth, decode audio |
| `transcribe.py` | Windowing, stitching, per-GPU scheduling, speed and VRAM |
| `scoring.py` | Call `evaluation/` and join the scores to the speed figures |
| `leaderboard.py` | Rank the models; write `summary.json`, `results.json`, `transcripts.json` |
| `run_benchmark.py` | Command-line entry point; `run()` is what the notebook calls |
| `settings.py` | Windowing, devices, defaults — from the environment or `.env` |
| `bridge.py` | **The only file that imports `evaluation/` and `stt/`** |
| `notebooks/build_notebook.py` | Generates the Kaggle notebook from these sources |
| `notebooks/kaggle_dual_t4.ipynb` | **Generated** — the self-contained Kaggle notebook |

### About `bridge.py`

Every service in this repo is import-isolated and talks to its neighbours over
HTTP. A benchmark cannot be: it needs model weights in-process (no HTTP round
trip per window) and the metrics called directly (no server to stand up on a
Kaggle runtime). That coupling is real, so it lives in one file. Delete
`bridge.py` and nothing else in this folder knows the other modules exist.

`settings.py` is called that, and not `config.py`, because `bridge.py` puts
`evaluation/` on `sys.path` and `evaluation/config.py` would otherwise win the
name.

## Run

```bash
pip install -r requirements.txt
python run_benchmark.py --audio recordings/ --truth labels/ --models whisper,seamless
```

Or with a manifest, when the files are not laid out by stem:

```bash
python run_benchmark.py --manifest dataset/manifest.json --out results/run-01
```

```json
[
  {"asset_id": "DPM89130", "audio": "audio/DPM89130.mp3", "reference": "truth/DPM89130.txt"},
  {"asset_id": "DPM89131", "audio": "audio/DPM89131.mp3"}
]
```

`reference` may be inline text or a path, and may be omitted entirely — see
*Unlabelled recordings* below.

To see what can be benchmarked:

```bash
python run_benchmark.py --list-models
```

The models come from `stt/app/config.py`. Adding one there is enough to make it
benchmarkable; nothing in this folder lists models by name.

### On Kaggle

[`notebooks/kaggle_dual_t4.ipynb`](notebooks/kaggle_dual_t4.ipynb) is
**self-contained**. Upload it, set the accelerator to **GPU T4 ×2** and Internet
to **On** (the weights come from Hugging Face), and Run All. Nothing to clone,
no dataset to attach.

It is also **generated**, not hand-written. `build_notebook.py` embeds each of
the seventeen source files into a cell that writes it back to disk in the same
three-folder layout the repo uses; the notebook then imports it unchanged. So
the code that runs on Kaggle is character-for-character the code the tests
cover — a self-contained notebook without the usual price of one, which is a
second implementation that quietly drifts from the first.

**After changing anything under `benchmark/`, `evaluation/` or `stt/app/`:**

```bash
python benchmark/notebooks/build_notebook.py
```

`tests/test_notebook.py` fails if you forget, and separately reconstructs the
tree in a temp directory with this repo off `sys.path` and runs a benchmark
there — so "self-contained" is a tested claim, not an intention.

Set the accelerator to **GPU T4 ×2**. The notebook checks that both GPUs are
visible before it starts, since a session that quietly fell back to one takes
twice as long to tell you. It also runs a stub-model dry run over three
recordings before loading any weights, so a wrong path costs seconds.

**Results go to `/kaggle/working`** — the writable half of the runtime, and the
only directory Kaggle keeps when the session ends.

## Unlabelled recordings

Ground truth arrives slower than audio does. A recording with no reference is
still transcribed, timed and written to `transcripts.json` — it is only left
out of the scores. Those drafts are the input to the next labelling round
rather than a by-product of this one.

## Output

```
results/
├── leaderboard.md      the ranked table
├── leaderboard.csv     the same, one row per model
├── per_report.csv      one row per recording per model, with the text
├── summary.json        per-model corpus totals, rates, distribution and speed
├── results.json        one scored row per recording per model
└── transcripts.json    what each model actually said, with timings
```

Both CSVs are UTF-8 **with a BOM**, because Excel reads Persian as mojibake
without one.

The ranking is on **corpus WER** — total edits over total reference words —
not the mean of the per-report WERs, which would weight a one-line finding
exactly like a multi-minute study.

Aggregate rates are computed from **summed counts**, never by averaging
per-report rates — `evaluation/`'s rule, and it applies here for the same
reason: one negation error in a short report becomes a 100% rate and drowns out
everything else.

`summary.json` also carries **WER by recording length** (`<30s`, `30s–2m`,
`>2m`). Long audio is where windowing and stitching can go wrong and where a
model with a short attention span degrades; a single overall WER hides both.

## Metrics

All of them come from `evaluation/`; see
[its README](../evaluation/README.md#metrics) for the full list and what each
one means. Both levels are carried through in full:

* **Per report** → `results.json` — every general, clinical and semantic metric
  `evaluate()` returns, plus `requires_medical_review` and its reasons.
* **Per model across the batch** → `summary.json` — the WER distribution, every
  clinical rate recomputed from summed counts, and mean/tail figures for the
  text metrics.

The leaderboard shows a readable subset of that; `summary.json` has all of it.

The benchmark adds three metrics of its own, which are about the hardware
rather than the text:

| Metric | Meaning |
|---|---|
| Real-time factor | Seconds of compute per second of audio. Below 1.0 is faster than real time. |
| Throughput | The same number read forward: hours of audio per hour of wall clock. |
| Peak VRAM | What one GPU had to hold, in GB. Not the sum across replicas. |
| Load seconds | Cold-start time for the weights — a per-session cost, not a per-request one. |
| WER by duration | Corpus WER within `<30s`, `30s–2m` and `>2m` buckets. |

The leaderboard's `latin` / `latin(ref)` pair is script contamination — the
share of letters that are not Arabic-script, in the transcript and in the
ground truth. **Read the gap, not the number.** This corpus code-switches
English radiology terms on purpose, so a correct transcript is "contaminated"
too; a model actually drifting into English shows up as `latin` pulling away
from `latin(ref)`.

The two embedding metrics (BERTScore, semantic similarity) are off by default
because they load a second model; add `--semantic` to include them. They are
computed for the whole batch in one pass — scoring report-by-report would mean
a batch of one against a model that takes longer to load than to run.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Nothing loads weights, touches a GPU or opens a socket — the model is injected,
so the windowing, stitching, scheduling and failure paths are all tested on
CPU in under a second.
