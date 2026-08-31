<style>
body { font-family: "Segoe UI", Tahoma, sans-serif; font-size: 11pt; }
h1 { font-size: 17pt; } h2 { font-size: 13pt; margin-top: 1em; }
table { border-collapse: collapse; width: 100%; margin-bottom: 0.6em; }
th, td { border: 1px solid #bbb; padding: 4px 8px; vertical-align: top; }
th { background: #f0f0f0; }
.added, .added * { color: #0b5394; }
.legend { font-size: 9.5pt; margin: 0.2em 0 1em 0; }
</style>

# Speech Recognition — Metrics Summary

Benchmark of local Persian STT models (FLEURS fa_ir)<span class="added">, plus the clinical
metrics used to evaluate dictated radiology reports</span>. Why each metric is tracked.

<p class="legend"><span class="added">■ Blue = added for clinical (radiology) evaluation.</span>
Black = the original STT benchmark set.</p>

## Accuracy

| Metric | Why we use it |
| ------ | ------------- |
| **WER** | % of words wrong — the standard ASR headline metric. |
| **CER** | % of characters wrong — fairer for Persian's attached suffixes/clitics. |
| **chrF** | Character n-gram overlap — gives partial credit for morphological variants WER counts as fully wrong. |
| **BERTScore (ParsBERT)** | Token-level *meaning* match — credits valid synonyms that WER penalizes. |
| **Semantic similarity** | Whole-sentence meaning match via embeddings — did it preserve the message. |

## Reliability (critical for medical use)

| Metric | Why we use it |
| ------ | ------------- |
| **SER** | % of clips with *any* error — how often a transcript is perfect. |
| **WER P50 / P90 / P95** | Typical vs worst-case clips; averages hide bad outliers. |
| **% perfect / % catastrophic** | Share of flawless vs badly-wrong clips. |
| **Sub / Ins / Del rates** | Error *type*; insertions = hallucinated words (highest medical risk). |
| **Empty-output rate** | How often the model returns nothing. |

## Persian-specific failure modes

| Metric | Why we use it |
| ------ | ------------- |
| **Hallucination ratio** | Output vs reference length; >1 means inventing words. |
| **Script contamination** | % non-Persian characters — flags Latin/English leakage. |
| **Repetition score** | Repeated n-grams — catches Whisper's loop-hallucination on silence. |
| **Punctuation F1** | Fairly compares punctuating models (Whisper) vs raw-text ones (Wav2Vec2/MMS). |

<div class="added" markdown="1">

> **Does not transfer to dictated radiology.** Script contamination assumes Latin
> characters are leakage. In radiology dictation the speaker mixes Persian with
> English terms on purpose (*hydronephrosis*, *UVJ*, *mm*), so a correct transcript
> scores as heavily contaminated. Treat it as a FLEURS-only metric, or replace it
> with a code-switching check that scores whether English terms were rendered
> correctly rather than whether they appear at all.

## Clinical accuracy (dictated radiology reports)

Scored on the report text, against a radiologist-verified reference. These are the
metrics that carry clinical risk — a report can have a good WER and still be unsafe.

| Metric | Why we use it |
| ------ | ------------- |
| **Negation error rate** | Presence/absence flips — "no stone" transcribed as "stone". Changes the diagnosis outright. |
| **Laterality error rate** | Right/left flips (راست/چپ, Right/Left) — sends a finding to the wrong side of the body. |
| **Number error rate** | Wrong measurement values — the single highest-risk error class; a 6 mm stone read as 1 cm changes management. |
| **Unit error rate** | mm vs cm vs میلی‌متر vs سانتی‌متر. Compare physical quantities, not unit strings, so 10 mm and 1 cm are equal. |
| **Medical term precision** | Of the clinical terms produced, how many are actually supported by the reference — catches invented terminology. |
| **Medical term recall** | Of the clinical terms that should be present, how many survived — catches dropped findings. |
| **Medical term F1** | Single combined figure for terminology handling. |
| **Critical omission rate** | Important reference content missing from the output. "Critical" must be defined explicitly (measurement, negated finding, laterality marker, or flagged term) or it collapses into "anything missing". |
| **Unsupported addition rate** | Output content with no support in the reference — the fabrication case (e.g. a model inventing a patient history that was never dictated). |
| **`requires_medical_review`** | Per-report flag raised by any critical error. In a post-edit workflow this fires *after* sign-off, so it is an audit trigger for a second look, not a safety gate. |

## Post-edit loop (live monitoring)

When the reference is the radiologist's correction of our own output, the numbers
mean something different: they measure editing effort, not correctness.

| Metric | Why we use it |
| ------ | ------------- |
| **Edit burden (post-edit WER)** | How much the radiologist had to change. A production workload signal — do not read it as accuracy, since the reference is derived from the model's own text. |
| **% accepted unchanged** | Share of reports signed with no edit. Track alongside engagement signals: a rubber-stamped report is indistinguishable from a correct one. |
| **Correction return rate** | Share of outputs that ever come back corrected. Low or skewed return means the evaluated sample is biased toward whichever reports are easy to edit. |

</div>
