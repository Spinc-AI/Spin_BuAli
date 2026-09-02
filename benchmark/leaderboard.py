"""Turning a summary into something a person can read.

The leaderboard deliberately shows accuracy and cost side by side. A model that
wins on WER while running at four times real time on a T4 has not won anything
a clinic can deploy, and a table that hides the second number invites exactly
that mistake.
"""
import csv
import json
import pathlib

# Column key, heading, format. Order is the reading order.
COLUMNS = [
    ("model", "model", "{}"),
    ("reports", "n", "{}"),
    # The headline is the corpus WER -- total edits over total words -- not the
    # median of per-report WERs. The percentiles beside it describe the spread.
    ("corpus_wer", "WER", "{:.3f}"),
    ("corpus_cer", "CER", "{:.3f}"),
    ("wer_p50", "p50", "{:.3f}"),
    ("wer_p90", "p90", "{:.3f}"),
    ("chrf_mean", "chrF", "{:.3f}"),
    ("medical_term_f1", "term F1", "{:.3f}"),
    # Rates, not raw counts: two models that saw the same references have the
    # same denominators, but the rate is what survives a change of corpus.
    ("negation_error_rate", "neg err", "{:.1%}"),
    ("laterality_error_rate", "lat err", "{:.1%}"),
    ("number_error_rate", "num err", "{:.1%}"),
    ("unit_error_rate", "unit err", "{:.1%}"),
    ("critical_omission_rate", "crit om", "{:.1%}"),
    ("unsupported_addition_rate", "unsup add", "{:.1%}"),
    ("repetition_score_p95", "rep p95", "{:.2f}"),
    # Shown next to the reference's own figure, never alone: this corpus
    # code-switches English terms on purpose, so the gap is the signal.
    ("script_contamination_mean", "latin", "{:.1%}"),
    ("reference_script_contamination_mean", "latin(ref)", "{:.1%}"),
    ("insertion_rate", "ins", "{:.3f}"),
    ("review_rate", "review", "{:.0%}"),
    ("pct_catastrophic", "catastr", "{:.0%}"),
    ("real_time_factor", "RTF", "{:.2f}"),
    ("throughput", "xRT", "{:.1f}"),
    ("peak_vram_gb", "VRAM GB", "{:.1f}"),
]

# Default ranking. Corpus WER is the number the field reports, so it is the
# number the table sorts on.
SORT_BY = "corpus_wer"


def rows(summary, sort_by=None):
    """One flat row per model, best first.

    Flattening matters: `summarize()` nests counts, rates, reliability and
    speed in four places, and a comparison table is unreadable if the reader
    has to remember which number lives where.
    """
    sort_by = sort_by or SORT_BY
    flattened = []
    for bucket in summary.get("models", []):
        reliability = bucket.get("reliability", {})
        flattened.append({
            "model": bucket["model"],
            "reports": bucket["reports"],
            **{key: reliability.get(key, 0.0) for key in
               ("wer_p50", "wer_p90", "wer_p95", "pct_perfect", "pct_catastrophic",
                "ser", "empty_output_rate")},
            **bucket.get("corpus", {}),
            **bucket.get("text", {}),
            **bucket.get("semantic", {}),
            **bucket.get("rates", {}),
            **{key: bucket.get(key, 0) for key in
               ("negation_errors", "laterality_errors", "number_errors", "unit_errors",
                "critical_omissions", "unsupported_additions",
                "negation_scored", "laterality_scored", "reference_measurements",
                "critical_omission_scored", "unsupported_addition_scored")},
            "real_time_factor": bucket.get("speed", {}).get("real_time_factor", 0.0),
            "peak_vram_gb": bucket.get("speed", {}).get("peak_vram_gb", 0.0),
            "load_seconds": bucket.get("speed", {}).get("load_seconds", 0.0),
            "failed": bucket.get("speed", {}).get("failed", 0),
            # Hours of audio per hour of wall clock -- the deployment question
            # RTF answers backwards.
            "throughput": round(1.0 / rtf, 2) if (rtf := bucket.get("speed", {}).get(
                "real_time_factor", 0.0)) else 0.0,
            **{f"wer_{name}": figures["wer"]
               for name, figures in bucket.get("by_duration", {}).items()},
        })
    return sorted(flattened, key=lambda row: row.get(sort_by, 0))


def to_markdown(summary, sort_by=None):
    """A markdown table -- readable in a terminal, rendered in a notebook."""
    table = rows(summary, sort_by)
    if not table:
        return "_no scored models_"

    header = "| " + " | ".join(heading for _, heading, _ in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, rule]
    for row in table:
        lines.append("| " + " | ".join(
            _cell(row.get(key), template) for key, _, template in COLUMNS) + " |")
    return "\n".join(lines)


def _cell(value, template):
    if value is None:
        return "-"
    try:
        return template.format(value)
    except (TypeError, ValueError):
        return str(value)


def worst(results, count=10, by="wer"):
    """The recordings that went worst, so a run ends with something to look at.

    An aggregate says a model is 12% wrong; it never says which twelve percent.
    These are the rows to read before trusting any of the other numbers.
    """
    ranked = sorted(results, key=lambda result: -result["general"].get(by, 0))
    return [{
        "asset_id": result["asset_id"],
        "model": result["model"],
        by: result["general"].get(by),
        "repetition_score": result["general"].get("repetition_score"),
        "hallucination_ratio": result["general"].get("hallucination_ratio"),
        "review_reasons": result.get("review_reasons", []),
        "critical_errors": len(result.get("critical_errors", [])),
    } for result in ranked[:count]]


def to_dataframe(summary, sort_by=None):
    """The same rows as a pandas DataFrame, for sorting and plotting in a
    notebook. pandas is not a dependency of the benchmark itself, so this is
    only importable where it is already installed."""
    import pandas as pd

    return pd.DataFrame(rows(summary, sort_by))


def check_writable(out_dir):
    """Fail now rather than after the GPU time is spent.

    The results are written at the end of a run that can take hours. A path
    that cannot be created -- the usual case being a Kaggle notebook pointed at
    the read-only `/kaggle/input` mount -- should say so in the first second,
    not the last.
    """
    out_dir = pathlib.Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write-check"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise OSError(
            f"cannot write results to {out_dir}: {error}. "
            f"On Kaggle use a path under /kaggle/working.") from error
    return out_dir


def per_report_rows(results):
    """One flat row per recording per model, for a spreadsheet.

    The nested JSON is the record of what happened; this is the form a person
    actually reads a few hundred rows in. Reference and hypothesis travel with
    the numbers so a bad score can be looked at rather than just counted.
    """
    flattened = []
    for result in results:
        flattened.append({
            "asset_id": result["asset_id"],
            "model": result["model"],
            "audio_seconds": result.get("audio_seconds"),
            "real_time_factor": result.get("real_time_factor"),
            **result["general"],
            **result["clinical_metrics"],
            **{f"n_{key}": value for key, value in result["clinical_counts"].items()},
            **(result.get("semantic") or {}),
            "requires_medical_review": result["requires_medical_review"],
            "review_reasons": ";".join(result.get("review_reasons", [])),
            "critical_errors": len(result.get("critical_errors", [])),
            "transcription_error": result.get("transcription_error") or "",
            "reference": result.get("reference", ""),
            "hypothesis": result.get("hypothesis", ""),
        })
    return flattened


def _write_csv(path, table):
    if not table:
        return
    # utf-8-sig: Excel reads Persian text as mojibake without the BOM.
    columns = list(dict.fromkeys(key for row in table for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


def write(out_dir, results, summary, runs=None):
    """Persist a run: per-report scores, the summary, and the raw transcripts.

    Written as both JSON and CSV -- JSON keeps the nesting and is what gets
    re-read, CSV is what gets opened and sorted by a person.

    Transcripts are written separately and always, including for unlabelled
    recordings -- they are the input to the next labelling round, not a
    by-product of this one.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _dump(out_dir / "summary.json", summary)
    _dump(out_dir / "results.json", results)
    _dump(out_dir / "leaderboard.md", to_markdown(summary), raw=True)
    _write_csv(out_dir / "leaderboard.csv", rows(summary))
    _write_csv(out_dir / "per_report.csv", per_report_rows(results))

    if runs:
        transcripts = {}
        for run in runs:
            transcripts[run.model] = {
                "summary": run.summary(),
                "transcripts": [t.as_dict() for t in run.transcripts],
            }
        _dump(out_dir / "transcripts.json", transcripts)
    return out_dir


def _dump(path, payload, raw=False):
    text = payload if raw else json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
