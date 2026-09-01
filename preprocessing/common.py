'''Shared plumbing for the preprocessing stages.

Not a stage itself — just the things more than one stage needs: the config, the
error type, the policy this pipeline refuses to break, and the audio I/O helpers.
Every stage module imports from here and from no other module in the package, so
a stage can be imported, tested and run on its own.
'''

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

log = logging.getLogger("audio_preprocessing")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "preprocessing_config.json")

# Extensions picked up when an input path is a directory.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm"}


class PreprocessingError(Exception):
    '''An input that cannot be turned into valid derived artefacts.'''


# What this pipeline is deliberately not allowed to do. The values are mirrored
# into every segments.json, and a config that flips one is rejected outright
# rather than accepted and quietly ignored.
POLICY = {
    "preserve_original": True,
    "denoise": False,
    "gain_normalization": False,
    "remove_silence": False,
    "vad_replaces_original": False,
}


# ============================================================
# Configuration
# ============================================================

def load_config(path=None):
    '''Read the JSON config, falling back to the one shipped next to this file.'''
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    check_config(config)
    return config


def check_config(config):
    '''Reject configurations that cannot produce chunks satisfying their own limits.'''
    chunking = config["chunking"]
    target, maximum = chunking["target_duration_sec"], chunking["max_duration_sec"]
    overlap = chunking["overlap_sec"]
    if target > maximum:
        raise ValueError("chunking.target_duration_sec exceeds chunking.max_duration_sec")
    if not 0 <= overlap < target:
        raise ValueError("chunking.overlap_sec must be >= 0 and < chunking.target_duration_sec")
    if chunking["min_duration_sec"] > target:
        raise ValueError("chunking.min_duration_sec exceeds chunking.target_duration_sec")
    if chunking["source"] not in ("standardized", "vad_segments"):
        raise ValueError("chunking.source must be 'standardized' or 'vad_segments'")
    if chunking["strategy"] not in ("adaptive", "uniform", "fixed"):
        raise ValueError("chunking.strategy must be 'adaptive', 'uniform' or 'fixed'")

    for name, required in POLICY.items():
        if config["policy"][name] != required:
            raise ValueError(
                f"policy.{name} must stay {str(required).lower()} — this pipeline only ever "
                "produces derived copies and never alters the recording it was given"
            )


# ============================================================
# Identity — the original file is only ever hashed, never written
# ============================================================

def file_sha256(path):
    '''Stream the file through SHA-256 and return the hex digest.'''
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_id(sha256, strategy="sha256_16"):
    '''Derive the per-file output directory name from its digest.'''
    if strategy == "sha256_16":
        return sha256[:16]
    if strategy == "sha256":
        return sha256
    raise ValueError(f"unknown output.id_strategy '{strategy}'")


# ============================================================
# Audio I/O
# ============================================================

def probe(path):
    '''Return the container facts soundfile can read without decoding samples.'''
    with sf.SoundFile(path) as handle:
        return {
            "format": handle.format,
            "subtype": handle.subtype,
            "sample_rate": int(handle.samplerate),
            "channels": int(handle.channels),
            "frames": int(handle.frames),
            "duration_sec": round(handle.frames / handle.samplerate, 6) if handle.samplerate else 0.0,
        }


def decode(path, max_duration_sec=None):
    '''Decode an audio file into a mono float32 array plus its sample rate.

    Mirrors how the STT service itself reads uploads (``stt/app/main.py::_read_audio``):
    soundfile, then average the channels down to mono. Formats libsndfile cannot
    open (m4a, wma, some mp3 builds) are handed to ffmpeg first, if it is installed.

    ``max_duration_sec`` is checked against the container header before any samples
    are pulled in, so an absurdly long file is refused rather than decoded into
    memory first and rejected afterwards.
    '''
    try:
        info = probe(path)
        _refuse_oversized(path, info, max_duration_sec)
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        decoder = "soundfile"
    except PreprocessingError:
        raise
    except Exception as exc:
        audio, sr, info = _decode_via_ffmpeg(path, exc, max_duration_sec)
        decoder = "ffmpeg"

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, np.float32), int(sr), info, decoder


def _refuse_oversized(path, info, max_duration_sec):
    '''Reject a file on its header alone, before it is read into memory.'''
    if max_duration_sec and info["duration_sec"] > max_duration_sec:
        raise PreprocessingError(
            f"duration {info['duration_sec']:.1f}s exceeds the {max_duration_sec}s maximum"
        )


def _decode_via_ffmpeg(path, original_error, max_duration_sec=None):
    '''Transcode to a temporary WAV with ffmpeg, then read that.

    The temporary file is a scratch copy — the input path is never touched.
    '''
    if shutil.which("ffmpeg") is None:
        raise PreprocessingError(
            f"could not decode {os.path.basename(path)} ({original_error}); "
            "install ffmpeg to handle this container"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, tmp.name],
            check=True,
        )
        info = probe(tmp.name)
        # The probe describes ffmpeg's temporary WAV; rate, channels and length
        # carry over from the source, but its container and sample format do not.
        info["format"] = os.path.splitext(path)[1].lstrip(".").upper() or "UNKNOWN"
        info["subtype"] = "unknown"
        _refuse_oversized(path, info, max_duration_sec)
        audio, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
    except subprocess.CalledProcessError as exc:
        raise PreprocessingError(f"ffmpeg could not decode {os.path.basename(path)}: {exc}")
    finally:
        os.unlink(tmp.name)
    return audio, sr, info


def write_wav(path, audio, config):
    '''Write a mono float32 array as WAV/PCM 16-bit at the configured rate.'''
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(
        path,
        audio,
        config["audio"]["target_sample_rate"],
        subtype=config["audio"]["target_subtype"],
        format=config["audio"]["target_format"],
    )


def frame_rms(audio, sr, frame_ms):
    '''Per-frame RMS of the signal, plus how many seconds each frame covers.

    The one shared measurement the VAD fallback and adaptive chunking both rely
    on, so the two cannot disagree about where a recording goes quiet.
    '''
    frame = max(1, int(sr * frame_ms / 1000))
    padded = np.pad(audio, (0, (-len(audio)) % frame))
    frames = padded.reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)), frame / sr
