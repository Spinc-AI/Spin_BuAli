'''Stage 6 — the per-run CSV summary.

One row per input file, whether it succeeded or failed. The column set is fixed
so a report stays readable when stages are skipped: columns belonging to a stage
that did not run are simply blank, and ``stages`` records which ones did.
'''

import csv
import os
from datetime import datetime, timezone

NAME = "report"
REQUIRES = ()

REPORT_COLUMNS = [
    "file_id", "source_path", "status", "error", "stages",
    "source_format", "source_subtype", "source_sample_rate", "source_channels",
    "source_duration_sec", "peak", "rms",
    "standardized_duration_sec", "sample_rate", "channels", "subtype",
    "resampler", "clipped_samples",
    "vad_backend", "vad_segments", "speech_sec", "speech_ratio",
    "chunk_strategy", "chunk_count", "chunk_min_sec", "chunk_max_sec",
    "overlap_sec", "boundary_quietness",
    "checks_passed", "processed_at",
]


def row(result, status="ok", error=""):
    '''Flatten one processed file into a report row.'''
    entry = {column: "" for column in REPORT_COLUMNS}
    entry.update(
        {
            "file_id": result.get("file_id", ""),
            "source_path": result["source"]["path"],
            "status": status,
            "error": error,
            "stages": " ".join(result.get("stages_run", [])),
            "checks_passed": result.get("checks", {}).get("passed", ""),
            "processed_at": result.get("created_at", _now()),
        }
    )

    source = result["source"]
    for column, key in (
        ("source_format", "format"),
        ("source_subtype", "subtype"),
        ("source_sample_rate", "sample_rate"),
        ("source_channels", "channels"),
        ("source_duration_sec", "duration_sec"),
        ("peak", "peak"),
        ("rms", "rms"),
    ):
        entry[column] = source.get(key, "")

    if "standardized" in result:
        standard = result["standardized"]
        entry.update(
            {
                "standardized_duration_sec": standard["duration_sec"],
                "sample_rate": standard["sample_rate"],
                "channels": standard["channels"],
                "subtype": standard["subtype"],
                "resampler": standard["resampler"],
                "clipped_samples": standard["clipped_samples"],
            }
        )

    if "vad" in result:
        vad = result["vad"]
        entry.update(
            {
                "vad_backend": vad["backend"] or "",
                "vad_segments": len(vad["segments"]),
                "speech_sec": vad["speech_sec"],
                "speech_ratio": vad["speech_ratio"],
            }
        )

    if "chunks" in result:
        durations = [chunk["duration_sec"] for chunk in result["chunks"]]
        decision = result.get("chunking", {}).get("decision", {})
        regions = decision.get("regions", [])
        entry.update(
            {
                "chunk_strategy": decision.get("strategy", ""),
                "chunk_count": len(durations),
                "chunk_min_sec": round(min(durations), 3) if durations else "",
                "chunk_max_sec": round(max(durations), 3) if durations else "",
                "overlap_sec": result["chunking"].get("overlap_sec", ""),
                "boundary_quietness": regions[0].get("boundary_quietness", "") if regions else "",
            }
        )
    return entry


def failure_row(path, error, stages=()):
    '''Report row for a file that never made it through the pipeline.'''
    entry = {column: "" for column in REPORT_COLUMNS}
    entry.update(
        {
            "source_path": os.path.abspath(path),
            "status": "failed",
            "error": str(error),
            "stages": " ".join(stages),
            "checks_passed": False,
            "processed_at": _now(),
        }
    )
    return entry


def write_report(path, rows, append=True):
    '''Write (or append) the run's rows to preprocessing_report.csv.'''
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = append and os.path.isfile(path)
    with open(path, "a" if exists else "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    return path


def _now():
    '''Timestamp used when a row has no processed record to take one from.'''
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
