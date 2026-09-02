"""Batch scoring from the command line -- no server involved.

Reads a manifest of report pairs, scores each one, and writes a result file per
report plus a summary across the batch.

    python evaluate_results.py manifest.json --out results/

The manifest is a JSON array; `hypothesis`/`reference` may be inline text or a
path to a .txt file:

    [
      {"asset_id": "DPM89130", "model": "whisper-large-v3",
       "hypothesis": "out/DPM89130.txt", "reference": "truth/DPM89130.txt"}
    ]

Aggregate rates are computed from summed counts, never by averaging per-report
rates: a short report with one negation would otherwise turn a single error
into a 100% rate and drown out everything else.
"""
import argparse
import json
import pathlib
import sys

import config
from extractors import ClinicalTerms
from medical_metrics import METRICS_VERSION, evaluate

COUNT_FIELDS = [
    "reference_entities", "matched_entities", "true_positive_terms",
    "false_positive_terms", "false_negative_terms",
    "reference_measurements", "hypothesis_measurements",
    "negation_errors", "laterality_errors", "number_errors", "unit_errors",
    "critical_omissions", "unsupported_additions",
    "negation_scored", "laterality_scored",
    "critical_omission_scored", "unsupported_addition_scored",
]

# Per-report text metrics that are scores rather than counts, so they aggregate
# as a distribution rather than a sum. Which end of the distribution matters is
# not the same for all of them: chrF and punctuation F1 describe typical
# quality, while repetition and length ratio are failure detectors -- their mean
# is near zero even when a model loops on one report in twenty, so the tail is
# the number worth reading.
TEXT_FIELDS = {
    "cer": "mean",
    "chrf": "mean",
    "punctuation_f1": "mean",
    "script_contamination": "mean",
    "reference_script_contamination": "mean",
    "repetition_score": "tail",
    "hallucination_ratio": "tail",
}

# Edit counts summed across the batch, which is what a corpus WER is made of.
EDIT_FIELDS = [
    "substitutions", "insertions", "deletions",
    "reference_words", "hypothesis_words",
    "character_errors", "reference_chars",
]


def _corpus_rates(results):
    """Corpus WER and CER: total edits over total reference length.

    This is the headline number, and it is not the mean of the per-report WERs.
    A mean weights a one-line report exactly like a full page, so a model that
    fails on the short ones looks worse than it is -- and one that fails on the
    long ones looks better. `wer_p50` and friends describe the spread around
    this; they do not replace it.
    """
    totals = {field: 0 for field in EDIT_FIELDS}
    for result in results:
        for field in EDIT_FIELDS:
            totals[field] += result["general"].get(field, 0)

    word_errors = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    reference_words = totals["reference_words"]
    return {
        **totals,
        "corpus_wer": round(_ratio(word_errors, reference_words), 4),
        "corpus_cer": round(_ratio(totals["character_errors"], totals["reference_chars"]), 4),
        # Which of the three the errors actually are. Insertions matter most in
        # a medical transcript: they are the fabrication signal, not just noise.
        "substitution_rate": round(_ratio(totals["substitutions"], reference_words), 4),
        "insertion_rate": round(_ratio(totals["insertions"], reference_words), 4),
        "deletion_rate": round(_ratio(totals["deletions"], reference_words), 4),
    }


def _read(value, base):
    """Inline text, or the contents of a file path relative to the manifest."""
    candidate = base / value
    if len(value) < 260 and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


def score_manifest(manifest_path, terms):
    manifest_path = pathlib.Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent

    results = []
    for entry in entries:
        result = evaluate(_read(entry["hypothesis"], base),
                          _read(entry["reference"], base), terms)
        results.append({
            "asset_id": entry["asset_id"],
            "model": entry["model"],
            "pipeline": entry.get("pipeline"),
            "model_version": entry.get("model_version"),
            **result,
        })
    return results


def _reliability(results):
    """Distribution metrics -- the ones that only exist across many reports.

    A mean WER hides the shape of the failures: a model can look acceptable on
    average while collapsing completely on one report in twenty. These say how
    often, and how badly.
    """
    wers = sorted(result["general"]["wer"] for result in results)
    total = len(wers)
    if not total:
        return {}

    def percentile(fraction):
        return round(wers[min(int(fraction * total), total - 1)], 4)

    perfect = sum(1 for wer in wers if wer == 0)
    return {
        # Share of reports with any error at all.
        "ser": round(_ratio(total - perfect, total), 4),
        "wer_p50": percentile(0.50),
        "wer_p90": percentile(0.90),
        "wer_p95": percentile(0.95),
        # In a post-edit workflow this is also the "accepted unchanged" rate:
        # a zero WER means the radiologist altered nothing.
        "pct_perfect": round(_ratio(perfect, total), 4),
        "pct_catastrophic": round(
            _ratio(sum(1 for wer in wers if wer >= config.CATASTROPHIC_WER), total), 4),
        "empty_output_rate": round(_ratio(
            sum(1 for r in results if r["general"]["hypothesis_words"] == 0), total), 4),
    }


def _text_distribution(results):
    """Corpus-level view of the per-report text scores.

    WER gets percentiles because it is the headline; these get the same
    treatment for the same reason -- a model that is fine on average and
    collapses on one report in twenty is not fine, and only the tail says so.
    """
    summary = {}
    for field, emphasis in TEXT_FIELDS.items():
        values = sorted(result["general"][field] for result in results
                        if field in result["general"])
        if not values:
            continue
        summary[f"{field}_mean"] = round(sum(values) / len(values), 4)
        if emphasis == "tail":
            summary[f"{field}_p95"] = round(values[min(int(0.95 * len(values)), len(values) - 1)], 4)
            summary[f"{field}_max"] = round(values[-1], 4)
    return summary


def summarize(results, terms):
    """Per-model totals, with rates recomputed from the summed counts."""
    by_model = {}
    for result in results:
        bucket = by_model.setdefault(result["model"], {
            "model": result["model"], "reports": 0,
            "requires_medical_review": 0,
            **{field: 0 for field in COUNT_FIELDS},
        })
        bucket["reports"] += 1
        bucket["requires_medical_review"] += int(result["requires_medical_review"])
        for field in COUNT_FIELDS:
            bucket[field] += result["clinical_counts"][field]

    for model, bucket in by_model.items():
        model_results = [r for r in results if r["model"] == model]
        bucket["corpus"] = _corpus_rates(model_results)
        bucket["reliability"] = _reliability(model_results)
        bucket["text"] = _text_distribution(model_results)
        measurements = bucket["reference_measurements"]
        produced = bucket["true_positive_terms"] + bucket["false_positive_terms"]
        expected = bucket["true_positive_terms"] + bucket["false_negative_terms"]
        precision = _ratio(bucket["true_positive_terms"], produced)
        recall = _ratio(bucket["true_positive_terms"], expected)
        # Every rate is errors over what there was to get wrong, summed across
        # the batch -- never the mean of the per-report rates. One negation
        # error in a two-sentence report is a 100% rate, and averaging that in
        # would drown out a hundred correct ones.
        bucket["rates"] = {
            "medical_term_precision": round(precision, 4),
            "medical_term_recall": round(recall, 4),
            "medical_term_f1": round(_ratio(2 * precision * recall, precision + recall), 4),
            "negation_error_rate": round(
                _ratio(bucket["negation_errors"], bucket["negation_scored"]), 4),
            "laterality_error_rate": round(
                _ratio(bucket["laterality_errors"], bucket["laterality_scored"]), 4),
            "number_error_rate": round(_ratio(bucket["number_errors"], measurements), 4),
            "unit_error_rate": round(_ratio(bucket["unit_errors"], measurements), 4),
            "critical_omission_rate": round(
                _ratio(bucket["critical_omissions"], bucket["critical_omission_scored"]), 4),
            "unsupported_addition_rate": round(
                _ratio(bucket["unsupported_additions"], bucket["unsupported_addition_scored"]), 4),
            "review_rate": round(_ratio(bucket["requires_medical_review"], bucket["reports"]), 4),
        }
    return {
        "models": sorted(by_model.values(), key=lambda b: b["model"]),
        "reports": len(results),
        "evaluation": {
            "metrics_version": METRICS_VERSION,
            "terms_version": terms.version,
            "terms_sha": terms.sha,
        },
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", help="JSON array of report pairs")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--terms", default=None, help="path to clinical_terms.json")
    args = parser.parse_args(argv)

    terms = ClinicalTerms(args.terms)
    results = score_manifest(args.manifest, terms)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        # One file per report per model, so nothing overwrites anything else.
        name = f"{result['asset_id']}__{result['model'].replace('/', '-')}.json"
        (out_dir / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(results, terms)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scored {len(results)} report(s) -> {out_dir}")
    for bucket in summary["models"]:
        rates = bucket["rates"]
        reliability = bucket["reliability"]
        print(f"  {bucket['model']:28} reports={bucket['reports']:4} "
              f"WER={bucket['corpus']['corpus_wer']:.3f} "
              f"F1={rates['medical_term_f1']:.3f} "
              f"num_err={rates['number_error_rate']:.3f} "
              f"review={rates['review_rate']:.0%} "
              f"wer_p90={reliability['wer_p90']:.3f} "
              f"catastrophic={reliability['pct_catastrophic']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
