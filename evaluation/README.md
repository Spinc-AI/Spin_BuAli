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
| `medical_metrics.py` | Align the entities, then compute every metric (`evaluate()`) |
| `clinical_terms.json` | The clinical concept vocabulary |
| `evaluation_schema.json` | JSON Schema for the output |
| `main.py` | HTTP service (default port `8002`) |
| `evaluate_results.py` | Batch scoring from the command line, no server |

## Run

```bash
pip install -r requirements.txt
python main.py          # or: run.bat (Windows) / ./run.sh (Linux)
```

Interactive docs at `/docs`. No model is loaded and no GPU is required —
scoring is pure text processing.

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
model, plus a `summary.json`.

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

**Fabrication outside the vocabulary is invisible.** `unsupported_additions`
only counts concepts that exist in `clinical_terms.json`. In a real test, a
model that invented a *"68-year-old male with COPD"* history scored
`unsupported_additions = 0`, because COPD is chest vocabulary and this is an
abdomen/pelvis list. The only signal was `wer = 0.97`.

Practical consequence: log unrecognised terms during evaluation so the
vocabulary grows from real traffic, and read `wer` / `insertions` as the
complementary fabrication signal.

**`clinical_terms.json` is currently a seed.** Around 40 concepts, built from
observed dictations and scoped to abdominal / pelvic ultrasound. For real
coverage: RadLex as the English backbone (freely licensed, exactly this domain),
plus frequency mining over your own corpus, plus a radiologist pass for
synonyms.

**Take the versions seriously.** Every result carries `metrics_version` and
`terms_sha`. Adding a term or fixing a comparator moves the numbers on its own,
so results are only comparable between reports scored with the same versions.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

Negation, laterality, number and unit each have their own test file, as the task
document asks. No model is loaded and no network request is made.
