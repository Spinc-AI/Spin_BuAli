"""What to benchmark: the recordings, and the ground truth to score them against.

A benchmark item is one recording plus, optionally, the radiologist-verified
text for it. The reference is optional on purpose -- ground truth arrives later
and more slowly than audio does, and a run over unlabelled recordings is still
useful (it produces the drafts that become labels). Unlabelled items are
transcribed and timed like any other; they are simply left out of the scores
rather than counted as failures.
"""
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import soundfile as sf

import settings

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"}


@dataclass(frozen=True)
class Item:
    """One recording, and the text it should have produced."""
    asset_id: str
    audio: pathlib.Path
    reference: str | None = None

    @property
    def labelled(self):
        return bool(self.reference and self.reference.strip())


# --- Building a dataset ----------------------------------------------------
def from_json(manifest_path):
    """Read a manifest file.

    A JSON array of objects; `reference` may be inline text or the path to a
    `.txt` file, and both `audio` and `reference` paths are resolved relative to
    the manifest so a dataset directory can be moved as a unit.

        [{"asset_id": "DPM89130",
          "audio": "audio/DPM89130.mp3",
          "reference": "truth/DPM89130.txt"}]
    """
    manifest_path = pathlib.Path(manifest_path)
    base = manifest_path.parent
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    items = []
    for entry in entries:
        audio = (base / entry["audio"]).resolve()
        items.append(Item(
            asset_id=entry.get("asset_id") or audio.stem,
            audio=audio,
            reference=_read_reference(entry.get("reference"), base),
        ))
    return items


def from_directory(audio_dir, reference_dir=None):
    """Pair `<audio_dir>/X.mp3` with `<reference_dir>/X.txt` by stem.

    The convention exists so a directory of recordings can be benchmarked with
    no manifest at all, and so labels can be dropped in one at a time as they
    are transcribed -- an audio file with no matching `.txt` is unlabelled, not
    an error.
    """
    audio_dir = pathlib.Path(audio_dir)
    reference_dir = pathlib.Path(reference_dir) if reference_dir else None

    items = []
    for path in sorted(audio_dir.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        reference = None
        if reference_dir:
            truth = reference_dir / f"{path.stem}.txt"
            if truth.is_file():
                reference = truth.read_text(encoding="utf-8")
        items.append(Item(asset_id=path.stem, audio=path, reference=reference))
    return items


def _read_reference(value, base):
    """Inline text, or the contents of a `.txt` path relative to the manifest.

    Mirrors `evaluation/evaluate_results.py`, which accepts the same two forms,
    so a manifest written for one can be read by the other.
    """
    if not value:
        return None
    candidate = base / value
    if len(value) < 260 and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


def describe(items):
    """A one-line census, printed before a run commits to hours of GPU time."""
    labelled = sum(1 for item in items if item.labelled)
    missing = [item.asset_id for item in items if not item.audio.is_file()]
    return {
        "items": len(items),
        "labelled": labelled,
        "unlabelled": len(items) - labelled,
        "missing_audio": missing,
    }


# --- Audio -----------------------------------------------------------------
def load_audio(path, target_sr=settings.TARGET_SAMPLE_RATE):
    """Decode to mono float32 at `target_sr`.

    Returned as one array rather than a path because every model in the
    registry takes `(audio, sr)`, and because decoding once per recording
    instead of once per model keeps the comparison about inference speed.
    """
    audio, sr = _decode(pathlib.Path(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        audio, sr = _resample(audio, sr, target_sr), target_sr
    return audio, sr


def _decode(path):
    """soundfile first, ffmpeg second.

    libsndfile handles wav/flac/ogg and, since 1.1, mp3 -- but not m4a, and not
    every mp3 a phone produces. Rather than refuse those files, fall back to
    ffmpeg, which Kaggle and every Linux image already have.
    """
    try:
        return sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as error:  # noqa: BLE001 - any decode failure is worth retrying
        decoded = _decode_via_ffmpeg(path)
        if decoded is None:
            raise RuntimeError(f"could not decode {path.name}: {error}") from error
        return decoded


def _decode_via_ffmpeg(path):
    import io
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "wav", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return None
    return sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)


def _resample(audio, sr, target_sr):
    """torchaudio's resampler when it is installed, linear interpolation when
    it is not -- a benchmark should not fail to start over a missing optional
    dependency, and every model receives the identical array either way."""
    try:
        import torch
        import torchaudio

        tensor = torch.from_numpy(audio).unsqueeze(0)
        return torchaudio.functional.resample(tensor, sr, target_sr).squeeze(0).numpy()
    except ImportError:
        count = int(round(len(audio) * target_sr / sr))
        source = np.linspace(0, len(audio) - 1, num=count, dtype=np.float64)
        return np.interp(source, np.arange(len(audio)), audio).astype(np.float32)
