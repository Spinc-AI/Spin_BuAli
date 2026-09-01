'''Stage 3 — voice activity detection. Experimental, and never destructive.

Silero locates speech; ``speech_pad_ms`` widens every span so the first and last
syllable of a phrase cannot be shaved off. The result is a report written into
``segments.json`` — nothing here is cut from the standardized WAV or the chunks,
so a wrong VAD decision costs nothing but a note in the metadata.

Skip the stage entirely and no ``vad`` key appears in the output at all:

    python audio_preprocessing.py recording.wav -o out --skip vad
'''

import numpy as np

from common import frame_rms, log

NAME = "vad"
REQUIRES = ("standardization",)


def run(context):
    '''Report where speech is, without altering a sample of it.'''
    return run_vad(
        context.standardized, context.config["audio"]["target_sample_rate"], context.config
    )


def run_vad(audio, sr, config):
    '''Locate speech regions and pad them, without altering the audio itself.'''
    settings = config["vad"]
    report = {
        "enabled": bool(settings["enabled"]),
        "requested_backend": settings["backend"],
        "backend": None,
        "available": False,
        "speech_pad_ms": settings["speech_pad_ms"],
        "threshold": settings["threshold"],
        "error": None,
        "segments": [],
        "speech_sec": 0.0,
        "speech_ratio": 0.0,
        "experimental": True,
        "applied_to_audio": False,
    }
    if not settings["enabled"]:
        return report

    duration = len(audio) / sr
    try:
        spans = _silero_speech_spans(audio, sr, settings)
        report["backend"] = "silero"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if settings["fallback"] != "energy":
            return report
        log.warning("Silero VAD unavailable (%s) — falling back to energy VAD", exc)
        spans = _energy_speech_spans(audio, sr, settings)
        report["backend"] = "energy"

    report["available"] = True
    report["segments"] = [
        {
            "index": index,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
        }
        for index, (start, end) in enumerate(spans)
    ]
    speech = sum(end - start for start, end in spans)
    report["speech_sec"] = round(speech, 3)
    report["speech_ratio"] = round(speech / duration, 4) if duration else 0.0
    return report


def _silero_speech_spans(audio, sr, settings):
    '''Speech spans in seconds from Silero VAD, padded by ``speech_pad_ms``.'''
    import torch

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
    except ImportError:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        get_speech_timestamps = utils[0]

    stamps = get_speech_timestamps(
        torch.from_numpy(np.asarray(audio, np.float32)),
        model,
        sampling_rate=sr,
        threshold=settings["threshold"],
        min_speech_duration_ms=settings["min_speech_duration_ms"],
        min_silence_duration_ms=settings["min_silence_duration_ms"],
        speech_pad_ms=settings["speech_pad_ms"],
    )
    return [(stamp["start"] / sr, stamp["end"] / sr) for stamp in stamps]


def _energy_speech_spans(audio, sr, settings):
    '''Short-term-energy speech spans, used only when Silero cannot be loaded.

    Deliberately crude: it exists so the pipeline still emits inspectable segments
    on a machine with no model weights, not as a replacement for a real VAD.
    '''
    energy, step = frame_rms(audio, sr, settings["energy"]["frame_ms"])
    if energy.max() <= 0:
        return []

    speaking = energy >= energy.max() * settings["energy"]["threshold_ratio"]
    spans = _true_runs(speaking, step)
    spans = _merge_close(spans, settings["min_silence_duration_ms"] / 1000)
    spans = [s for s in spans if s[1] - s[0] >= settings["min_speech_duration_ms"] / 1000]
    return _pad_spans(spans, settings["speech_pad_ms"] / 1000, len(audio) / sr)


def _true_runs(flags, step):
    '''Convert a boolean per-frame mask into (start, end) second pairs.'''
    spans, start = [], None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start * step, index * step))
            start = None
    if start is not None:
        spans.append((start * step, len(flags) * step))
    return spans


def _merge_close(spans, min_gap):
    '''Join spans separated by less than ``min_gap`` seconds of silence.'''
    merged = []
    for start, end in spans:
        if merged and start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _pad_spans(spans, pad, duration):
    '''Widen every span by ``pad`` seconds on both sides, clamped to the file.

    This is the guard against the first and last syllable of a phrase being cut
    off; overlapping results are merged so padding never produces double regions.
    '''
    padded = [(max(0.0, start - pad), min(duration, end + pad)) for start, end in spans]
    return _merge_close(padded, 0.0)


if __name__ == "__main__":
    import json
    import sys

    from common import decode, load_config

    settings = load_config()
    samples, rate, _, _ = decode(sys.argv[1])
    print(json.dumps(run_vad(samples, rate, settings), indent=2))
