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
    "false_positive_terms", "false_negative_terms", "reference_measurements",
    "negation_errors", "laterality_errors", "number_errors", "unit_errors",
    "critical_omissions", "unsupported_additions",
]


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
        bucket["reliability"] = _reliability(
            [r for r in results if r["model"] == model])
        measurements = bucket["reference_measurements"]
        produced = bucket["true_positive_terms"] + bucket["false_positive_terms"]
        expected = bucket["true_positive_terms"] + bucket["false_negative_terms"]
        precision = _ratio(bucket["true_positive_terms"], produced)
        recall = _ratio(bucket["true_positive_terms"], expected)
        bucket["rates"] = {
            "medical_term_precision": round(precision, 4),
            "medical_term_recall": round(recall, 4),
            "medical_term_f1": round(_ratio(2 * precision * recall, precision + recall), 4),
            "number_error_rate": round(_ratio(bucket["number_errors"], measurements), 4),
            "unit_error_rate": round(_ratio(bucket["unit_errors"], measurements), 4),
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
              f"F1={rates['medical_term_f1']:.3f} "
              f"num_err={rates['number_error_rate']:.3f} "
              f"review={rates['review_rate']:.0%} "
              f"wer_p90={reliability['wer_p90']:.3f} "
              f"catastrophic={reliability['pct_catastrophic']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
