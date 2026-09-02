#!/usr/bin/env python3
"""
audio_metadata_extractor.py
----------------------------------------------------------------------------
Extracts technical, quality, and integrity metadata from a folder of audio
files and writes:
  * one JSON file per audio file  (metadata/<audio_id>.json)
  * one summary CSV for the whole batch (audio_inventory.csv)

Design goals (see task brief):
  1. NEVER modify the original audio files (read-only access only).
  2. Work on the REAL decoded content, not just the file extension
     (format/codec/duration/sample_rate/... come from ffprobe, which
     actually inspects the file's bitstream).
  3. Keep working even if one file is corrupted -> that file gets
     decode_status="error" but the batch still finishes.
  4. Detect duplicate files by content (SHA-256), not by filename.
  5. Runnable with a single command:
         python3 audio_metadata_extractor.py --input INPUT_DIR --output OUTPUT_DIR

External dependency: the `ffmpeg` / `ffprobe` command-line tools must be
installed and available on PATH. Everything else uses only the Python
standard library plus NumPy (for the numeric quality calculations).
----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------

# Any file smaller than this (or with fewer decoded samples) is treated
# with extra suspicion, but is still processed normally.
CHUNK_SIZE = 1024 * 1024  # 1 MB, used while streaming the file for hashing

# A sample is considered "clipped" if its absolute amplitude (normalized
# to the 0.0-1.0 range) is at or above this threshold.
CLIPPING_THRESHOLD = 0.999

# A sample is considered "silent" if its absolute amplitude (normalized)
# is at or below this threshold. This is an *approximation* of silence,
# as requested in the task brief ("silence_ratio تقریبی").
SILENCE_AMPLITUDE_THRESHOLD = 0.01  # ~ -40 dBFS

SCHEMA_VERSION = "1.0"

logger = logging.getLogger("audio_metadata_extractor")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class AudioRecord:
    """One row of metadata for a single audio file."""
    audio_id: str
    original_filename: str
    sha256: str
    file_size_bytes: int
    format: Optional[str] = None
    codec: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    peak_level: Optional[float] = None
    rms_level: Optional[float] = None
    clipping_ratio: Optional[float] = None
    silence_ratio: Optional[float] = None
    decode_status: str = "unknown"          # "ok" | "error"
    error_message: Optional[str] = None
    duplicate_of: Optional[str] = None       # original_filename of the first identical file
    schema_version: str = SCHEMA_VERSION
    processed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Step 1: identity & security -> SHA-256 based audio_id
# --------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    """
    Streams the file in fixed-size chunks so we never load a huge file
    fully into memory just to hash it. Opens the file strictly in
    read-binary mode ("rb") -> the original file is never written to.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def derive_audio_id(full_sha256: str) -> str:
    """audio_id = first 16 hex characters of the full SHA-256 digest."""
    return full_sha256[:16]


# --------------------------------------------------------------------------
# Step 2: technical metadata via ffprobe (reads the REAL bitstream,
# not just the file extension)
# --------------------------------------------------------------------------

def run_ffprobe(path: Path) -> dict:
    """
    Calls `ffprobe` and returns the parsed JSON description of the file's
    container (`format`) and streams. Raises RuntimeError on failure so
    the caller can mark the file as decode_status="error".
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:300]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}")


def extract_technical_fields(probe: dict) -> dict:
    """
    Pulls the fields we care about out of the raw ffprobe JSON.
    Picks the first audio stream found (files with a single audio
    stream, as expected here, but this is defensive against files
    that also carry e.g. embedded cover art as a video stream).
    """
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not audio_streams:
        raise RuntimeError("no audio stream found in file")

    stream = audio_streams[0]

    # duration: prefer the stream's own duration, fall back to container.
    duration = stream.get("duration") or fmt.get("duration")
    duration = float(duration) if duration is not None else None

    # bit depth is not always present (e.g. lossy codecs like MP3 don't
    # really have one) -> stays None in that case, which is correct.
    bit_depth = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
    bit_depth = int(bit_depth) if bit_depth not in (None, "0", 0) else None

    bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
    bitrate = int(bitrate) if bitrate is not None else None

    # format_name can be a comma-separated list (e.g. "mov,mp4,m4a,...");
    # the first entry is the canonical short name.
    format_name = fmt.get("format_name", "").split(",")[0] or None

    return {
        "format": format_name,
        "codec": stream.get("codec_name"),
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": int(stream["channels"]) if stream.get("channels") else None,
        "bit_depth": bit_depth,
        "bitrate": bitrate,
    }


# --------------------------------------------------------------------------
# Step 3: quality metrics -> decode to raw PCM samples with ffmpeg,
# then compute Peak / RMS / Clipping / Silence with NumPy
# --------------------------------------------------------------------------

def decode_to_samples(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    """
    Uses ffmpeg to decode ANY input format to raw 16-bit PCM on stdout
    (a temp file is never created, and the original file is only opened
    for reading by ffmpeg). Returns a 1-D float32 numpy array normalized
    to the [-1.0, 1.0] range (channels are interleaved/flattened together,
    which is sufficient for the approximate metrics requested here).
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:300]}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no audio samples")

    raw = np.frombuffer(proc.stdout, dtype="<i2")  # little-endian int16
    if raw.size == 0:
        raise RuntimeError("decoded sample buffer is empty")

    return raw.astype(np.float32) / 32768.0


def compute_quality_metrics(samples: np.ndarray) -> dict:
    """Peak, RMS, clipping ratio and an approximate silence ratio."""
    abs_samples = np.abs(samples)

    peak = float(abs_samples.max())
    rms = float(np.sqrt(np.mean(np.square(samples))))
    clipping_ratio = float(np.mean(abs_samples >= CLIPPING_THRESHOLD))
    silence_ratio = float(np.mean(abs_samples <= SILENCE_AMPLITUDE_THRESHOLD))

    return {
        "peak_level": round(peak, 6),
        "rms_level": round(rms, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
    }


# --------------------------------------------------------------------------
# Step 4: putting one file's record together
# --------------------------------------------------------------------------

def build_record(path: Path, seen_hashes: dict[str, str]) -> AudioRecord:
    """
    seen_hashes maps full_sha256 -> original_filename of the FIRST file that
    had that hash. Note: audio_id is derived directly from the hash, so two
    identical files always share the exact same audio_id -- that value can
    never distinguish "this file" from "the original it duplicates". The
    *filename* of the first occurrence is what's actually informative, so
    that's what duplicate_of stores.
    """
    file_size = path.stat().st_size
    full_hash = compute_sha256(path)
    audio_id = derive_audio_id(full_hash)

    record = AudioRecord(
        audio_id=audio_id,
        original_filename=path.name,
        sha256=full_hash,
        file_size_bytes=file_size,
    )

    if full_hash in seen_hashes:
        record.duplicate_of = seen_hashes[full_hash]
    else:
        seen_hashes[full_hash] = path.name

    try:
        probe = run_ffprobe(path)
        tech = extract_technical_fields(probe)
        for key, value in tech.items():
            setattr(record, key, value)

        if not tech["sample_rate"] or not tech["channels"]:
            raise RuntimeError("missing sample_rate/channels; cannot decode samples")

        samples = decode_to_samples(path, tech["sample_rate"], tech["channels"])
        quality = compute_quality_metrics(samples)
        for key, value in quality.items():
            setattr(record, key, value)

        record.decode_status = "ok"

    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch everything
        # A broken/corrupted file must NOT crash the whole batch.
        record.decode_status = "error"
        record.error_message = str(exc)
        logger.warning("Failed to fully process %s: %s", path.name, exc)

    return record


# --------------------------------------------------------------------------
# Step 5: output -> per-file JSON + one combined CSV
# --------------------------------------------------------------------------

CSV_COLUMNS = [
    "audio_id", "original_filename", "sha256", "format", "codec",
    "duration_seconds", "sample_rate", "channels", "bit_depth", "bitrate",
    "file_size_bytes", "peak_level", "rms_level", "clipping_ratio",
    "silence_ratio", "decode_status", "error_message", "duplicate_of",
    "processed_at",
]


def write_json_record(record: AudioRecord, metadata_dir: Path) -> Path:
    out_path = metadata_dir / f"{record.audio_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record.to_json_dict(), f, ensure_ascii=False, indent=2)
    return out_path


def write_csv_inventory(records: list[AudioRecord], csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            row = record.to_json_dict()
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def discover_audio_files(input_dir: Path) -> list[Path]:
    """
    Intentionally does NOT filter by file extension. The whole point of
    this tool is to determine the REAL format from file content (via
    ffprobe), not to trust the filename. So every regular, non-hidden
    file in the input folder is picked up and handed to ffprobe/ffmpeg;
    if a given file truly isn't audio, it will simply come back with
    decode_status="error" instead of being silently skipped.
    """
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    return files


def run(input_dir: Path, output_dir: Path) -> list[AudioRecord]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(input_dir)
    if not files:
        logger.warning("No audio files found in %s", input_dir)

    seen_hashes: dict[str, str] = {}
    records: list[AudioRecord] = []

    for path in files:
        logger.info("Processing %s ...", path.name)
        record = build_record(path, seen_hashes)
        write_json_record(record, metadata_dir)
        records.append(record)

    csv_path = output_dir / "audio_inventory.csv"
    write_csv_inventory(records, csv_path)

    ok_count = sum(1 for r in records if r.decode_status == "ok")
    err_count = len(records) - ok_count
    dup_count = sum(1 for r in records if r.duplicate_of)
    logger.info(
        "Done: %d files processed | %d ok | %d errors | %d duplicates",
        len(records), ok_count, err_count, dup_count,
    )
    logger.info("Per-file JSON written to: %s", metadata_dir)
    logger.info("Summary CSV written to:   %s", csv_path)

    return records


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract technical, quality and integrity metadata from audio files."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=Path("input"),
        help="Folder containing the original audio files (read-only). Default: ./input",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output"),
        help="Folder where metadata/ and audio_inventory.csv will be written. Default: ./output",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
