# Benchmark

One notebook. Runs a batch of recordings through several STT models and ranks
them on accuracy, clinical correctness, speed and memory.

```
notebooks/kaggle_dual_t4.ipynb
```

It answers one question: **which model should we ship, and can the hardware run
it?** Every model is scored for accuracy *and* measured for cost in the same
run, because a model that wins on word error rate while running at four times
real time on a T4 has not won anything a clinic can deploy.

## Run it

Upload the notebook to Kaggle, set **Accelerator → GPU T4 ×2** and
**Internet → On** (the model weights come from Hugging Face), then Run All.

Nothing to clone, no dataset to attach, no setup beyond what the notebook does
itself. Point `AUDIO_DIR` at your recordings and `TRUTH_DIR` at the matching
`<stem>.txt` ground truth files, and go.

There is a dry-run cell before the real run: a stub model, no weights, seconds
instead of a download. Run it first — a wrong path then costs you seconds rather
than a GPU session.

## What it does

```
recordings ──▶ decode ──▶ 28s windows ──▶ model ──▶ stitch ──▶ transcript
                                            │                      │
                                    speed / VRAM               metrics
                                            │                      │
                                            └──────▶ leaderboard ◀─┘
```

**Windowing is not optional.** Whisper's encoder takes a fixed 30-second window.
Hand it a four-minute dictation and it transcribes the first thirty seconds and
stops — which shows up as a catastrophic word error rate that looks like a bad
model and is actually a bad harness. Every model gets the same windows.

**One model at a time, one replica per GPU.** Both T4s run the same model over
different halves of the batch: throughput doubles and not one number changes.
Peak VRAM stays one model's worth however many are being compared.

**A failure is a result.** A model that runs out of memory on one recording has
that recording scored as an empty output, not dropped — dropping it would
flatter the model that crashed.

**Unlabelled recordings still run.** Ground truth arrives slower than audio
does. A recording with no reference is transcribed, timed and written to
`transcripts.json`; it is only left out of the scores. Those drafts are the
input to the next labelling round.

## Metrics

Per recording, and again across the whole batch. The full list and what each one
means is in [`evaluation/README.md`](../evaluation/README.md#metrics) — the
notebook runs those exact functions.

The headline is **corpus WER**: total edits over total reference words, not the
mean of per-report WERs, which would weight a one-line finding exactly like a
multi-minute study. Alongside it: CER, chrF, the clinical error rates (negation,
laterality, number, unit, critical omission, unsupported addition), the failure
detectors (repetition, hallucination ratio), and WER broken out by recording
length.

Three metrics are the notebook's own, about the hardware rather than the text:

| Metric | Meaning |
|---|---|
| Real-time factor | Seconds of compute per second of audio. Below 1.0 is faster than real time. |
| Throughput | The same number read forward: hours of audio per hour of wall clock. |
| Peak VRAM | What one GPU had to hold. Not the sum across replicas. |

BERTScore and semantic similarity are opt-in — they load a second model — and
run over the whole batch in one pass when enabled.

One caution on **script contamination**: it counts letters that are not
Arabic-script, so it is reported beside the reference's own figure. Read the
gap, not the number. This corpus code-switches English radiology terms on
purpose, so a *correct* transcript is "contaminated" too.

## Output

Written to `/kaggle/working/benchmark_results`, the only directory Kaggle keeps
when a session ends.

```
leaderboard.csv     one row per model
leaderboard.md      the same, as markdown
per_report.csv      one row per recording per model, with reference and hypothesis
summary.json        the full nested batch summary
results.json        the full per-report scores
transcripts.json    what every model said, including unlabelled recordings
```

Both CSVs are UTF-8 with a BOM, so Excel reads the Persian correctly.

## About the code in section 2

The notebook's second section writes out the modules it needs and imports them.
Those cells are not hand-written glue — they are this project's real source
files, embedded verbatim: the metrics from `evaluation/`, the model classes from
`stt/app/`, and the harness itself. That is what lets the notebook stand alone
without becoming a second implementation that drifts from the live scoring
service.

The source was pruned from this folder once the notebook was generated, and
lives in git history. Commit **`523fadd`** holds the full module tree, the
generator that produced this notebook, and 60 tests:

```bash
git checkout 523fadd -- benchmark/
```

**Editing the notebook directly works, but it is a one-way door.** There is no
build step in this folder any more and no tests behind it, so a change made in a
cell is a change made in JSON, unverified, and it will not reach
`evaluation/` — where the same metric code still runs in production. For
anything beyond a constant: restore the source above, edit there, re-run
`notebooks/build_notebook.py`, and prune again.
