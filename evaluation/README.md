# Evaluation Service

Scores a transcript against a radiologist-verified reference. Two texts in,
general and clinical metrics out as JSON.

It holds no state and takes only text, so it can be called at any point after
transcription, and old reports can be re-scored when the metrics change.

## Files

The pipeline runs in this order:

| File | Responsibility |
|---|---|
| `text_normalizer.py` | Unify digits, letter forms, ZWNJ, and spelled-out numbers |
| `extractors.py` | Pull out measurements, negation, laterality, and clinical concepts |
| `general_metrics.py` | Text-level metrics: WER, CER, chrF, and the failure-shape signals |
| `medical_metrics.py` | Align the entities, then compute every metric (`evaluate()`) |
| `semantic_metrics.py` | The two optional embedding metrics — loaded only if asked for |
| `clinical_terms.json` | The clinical concept vocabulary |
| `evaluation_schema.json` | JSON Schema for the output |
| `main.py` | HTTP service (default port `8002`) |
| `evaluate_results.py` | Batch scoring from the command line, no server |

## Metrics

Everything named in [`docs/metrics_summary.md`](../docs/metrics_summary.md),
and where each one is computed.

### Per report — returned by `POST /evaluate`

| Metric | Status |
|---|---|
| WER | ✅ |
| CER | ✅ |
| chrF | ✅ |
| Sub / Ins / Del counts | ✅ |
| Hallucination ratio | ✅ output length over reference length |
| Repetition score | ✅ the loop detector |
| Punctuation F1 | ✅ computed on raw text, before normalisation |
| Script contamination | ✅ reported **with** the reference's own figure — see below |
| Negation error rate | ✅ |
| Laterality error rate | ✅ |
| Number error rate | ✅ |
| Unit error rate | ✅ |
| Medical term precision / recall / F1 | ✅ |
| Critical omission rate | ✅ |
| Unsupported addition rate | ✅ |
| `requires_medical_review` | ✅ with `review_reasons` |
| Edit burden (post-edit WER) | ✅ the same number as WER, read as editing effort when the reference is a post-edit |

### Per report, opt-in — `"include_semantic": true`

| Metric | Status |
|---|---|
| BERTScore (ParsBERT) | ✅ optional |
| Semantic similarity | ✅ optional |

Off by default because they are the only metrics here that load a model.
`GET /` reports whether the extras are installed; asking for them without the
extras returns `503`, not a silent omission.

For many reports at once use `semantic_metrics.compute_batch(pairs)`, which
loads each model once and scores every pair in one forward pass. `compute()` is
that function with a batch of one — fine for a live request, ruinous in a loop.

```bash
pip install -r requirements-semantic.txt
```

### Across a batch — `evaluate_results.py`

Every per-report metric above also has a corpus-level form here. Nothing is
per-report only.

**Corpus totals** — the headline numbers

| Metric | Status |
|---|---|
| Corpus WER | ✅ total edits over total reference words |
| Corpus CER | ✅ total character edits over total characters |
| Substitution / insertion / deletion rate | ✅ they sum to the corpus WER |

Corpus WER is **not** the mean of the per-report WERs. A mean weights a one-line
finding exactly like a multi-minute study, so a model that fails on short
reports looks worse than it is and one that fails on long reports looks better.
The insertion rate is broken out separately because in a medical transcript an
insertion is the fabrication signal, not just noise.

**Distribution around it**

| Metric | Status |
|---|---|
| SER | ✅ share of reports with any error |
| WER P50 / P90 / P95 | ✅ |
| % perfect | ✅ also the "accepted unchanged" rate in a post-edit workflow |
| % catastrophic | ✅ threshold in `config.CATASTROPHIC_WER` |
| Empty-output rate | ✅ |

**Clinical rates, recomputed from summed counts**

| Metric | Status |
|---|---|
| Negation / laterality error rate | ✅ |
| Number / unit error rate | ✅ |
| Critical omission rate | ✅ |
| Unsupported addition rate | ✅ |
| Medical term precision / recall / F1 | ✅ |
| Review rate | ✅ share of reports needing a human look |

**Text metrics**

| Metric | Status |
|---|---|
| CER, chrF, punctuation F1 | ✅ mean |
| Script contamination (hypothesis and reference) | ✅ mean |
| Repetition score, hallucination ratio | ✅ mean **and** P95 / max |

The last row is the reason the split exists. chrF describes typical quality, so
its mean is the number to read. Repetition and length ratio are failure
detectors — their mean stays near zero even when a model loops on one report in
twenty, so the tail is where the failure is visible.

Rates are always **summed errors over summed opportunities**, never the mean of
per-report rates. That is why `clinical_counts` carries a denominator beside
every error count (`negation_scored`, `critical_omission_scored`, …): without
them a batch rate cannot be rebuilt from stored results.

### Not implemented

| Metric | Why |
|---|---|
| Correction return rate | Needs stored history — how many transcripts ever come back corrected. This service keeps no state, so it cannot know. |

## Run

```bash
pip install -r requirements.txt
python main.py          # or: run.bat (Windows) / ./run.sh (Linux)
```

Interactive docs at `/docs`. No model is loaded and no GPU is required —
scoring is pure text processing unless the semantic extras are requested.

```bash
curl -X POST http://localhost:8002/evaluate -H "Content-Type: application/json" -d '{
  "asset_id": "DPM89130",
  "model": "whisper-large-v3",
  "pipeline": "multimodal",
  "hypothesis": "There is a 6 mm stone in the distal right ureter.",
  "reference": "There is a 7 mm calculus in the distal right ureter."
}'
```

> In deployment this service is not called directly: the controller is the only
> entry point, and `POST /evaluate` on port `9002` passes the same body through
> to here. The direct call above is for development and testing.

## Batch scoring

```bash
python evaluate_results.py manifest.json --out results/
```

`manifest.json` is an array of pairs; `hypothesis` / `reference` may be inline
text or the path to a `.txt` file. One result file is written per report per
model, plus a `summary.json` holding the per-model totals and the distribution
metrics above.

Aggregate rates are computed from **summed counts**, never by averaging
per-report rates — in a short report with a single negation, one error becomes a
100% rate and drowns out everything else.

## Design decisions

**Align before counting.** `6 mm stone` against `7 mm calculus` is one number
error, not an addition plus an omission. Concepts are aligned through
`clinical_terms.json` (so `stone`, `calculus` and `سنگ` are one concept), and
measurements are paired by equal physical quantity first.

**Physical quantities, not unit strings.** `10 mm` and `1 cm` are equal. When
they genuinely differ: the same number with a different unit is a unit error,
the same unit with a different number is a number error.

**A bare number is not a measurement.** In Persian the word for *one* is also
the indefinite article, so counting every digit would invent measurements. A
number qualifies when it carries a unit or belongs to a dimension group
(`107 در 44`).

**Negation applies to findings only.** A report says "no stone", never "no
kidney". Scoring negation on anatomy would just measure how far the cue window
happened to reach.

**"Critical" is a structural rule, not a hand-maintained flag.** An omission is
critical when what went missing is a measurement, or a concept the reference
stated with a negation or a side — no per-term clinical judgement needed.

## Known limitations

**`unsupported_additions` only sees the vocabulary it knows.** It counts
concepts present in `clinical_terms.json`, so a model that invented a
*"68-year-old male with COPD"* history scored zero there — COPD is chest
vocabulary and this is an abdomen/pelvis list. `hallucination_ratio` catches
that case by length instead, which is why both are reported; log unrecognised
terms during evaluation so the vocabulary grows from real traffic.

**`clinical_terms.json` is currently a seed.** Around 40 concepts, built from
observed dictations and scoped to abdominal / pelvic ultrasound. For real
coverage: RadLex as the English backbone (freely licensed, exactly this domain),
plus frequency mining over your own corpus, plus a radiologist pass for
synonyms.

**Script contamination only means something as a pair.** It counts letters
that are not Arabic-script, which on most Persian corpora is leakage. Not on
this one: these radiologists dictate English terms on purpose, so a *perfectly
correct* transcript scores as heavily contaminated. That is why the reference's
own figure is reported next to it — a model drifting into Latin shows up as a
**gap between the two**, not as a high number. Read alone, it will mislead.

**Take the versions seriously.** Every result carries `metrics_version` and
`terms_sha`. Adding a term or fixing a comparator moves the numbers on its own,
so results are only comparable between reports scored with the same versions.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Negation, laterality, number and unit each have their own test file, as the task
document asks. Nothing loads a model or opens a socket; the semantic tests skip
themselves when the optional extras are absent.
