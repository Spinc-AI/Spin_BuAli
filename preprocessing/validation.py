'''Stage 1 — validation and quality measurement.

Two jobs, and the difference between them matters.

**Validation** decides whether a file is worth processing at all: can it be
decoded, does it contain samples, are they finite, is its length plausible.
Failing any of these raises, and everything downstream may assume a file that
got past here is real audio.

**Measurement** describes what the audio is like — peak, RMS, how much of it is
clipped, how much is silence. None of these ever reject a file. They are
reported so a corpus can be characterised before anyone wonders why a model
does badly on it: clipping in particular is unrecoverable, and knowing 6% of a
recording is flat-topped explains far more than any amount of retuning.

Standalone:

    python validation.py recording.wav
'''

import numpy as np

from common import PreprocessingError, decode

NAME = "validation"
REQUIRES = ()


def run(context):
    '''Decode the source and check it, putting the audio into the context.'''
    config = context.config
    audio, sr, info, decoder = decode(context.path, config["validation"]["max_duration_sec"])
    measurements = validate(audio, sr, config)

    context.audio = audio
    context.sample_rate = sr
    return {"decoder": decoder, **info, **measurements}


def validate(audio, sr, config):
    '''Raise unless the decoded audio is non-empty, finite and of a sane length.

    Returns the measurements the checks were made on so they can be reported.
    '''
    rules = config["validation"]
    duration = len(audio) / sr if sr else 0.0

    if len(audio) == 0:
        raise PreprocessingError("file decodes to zero samples")
    if rules["reject_non_finite"] and not np.all(np.isfinite(audio)):
        raise PreprocessingError("decoded samples contain NaN or Inf")

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    if peak < rules["min_peak_amplitude"]:
        raise PreprocessingError(
            f"file is silent (peak {peak:.2e} < {rules['min_peak_amplitude']:.2e})"
        )
    if duration < rules["min_duration_sec"]:
        raise PreprocessingError(
            f"duration {duration:.3f}s is below the {rules['min_duration_sec']}s minimum"
        )
    if duration > rules["max_duration_sec"]:
        raise PreprocessingError(
            f"duration {duration:.1f}s exceeds the {rules['max_duration_sec']}s maximum"
        )
    return {
        "duration_sec": round(duration, 6),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        **quality(audio, config),
    }


def quality(audio, config):
    '''How much of the recording is clipped, and how much is silence.

    Descriptive, never a verdict — nothing here rejects a file.

    Clipping is the one worth reading first. A sample at full scale means the
    waveform was cut flat there, and no later stage recovers it: resampling,
    denoising and gain all work on what survived. A corpus with several percent
    clipped is a recording problem, and it is cheaper to fix the microphone
    than to keep tuning models against the damage.

    Silence is approximate on purpose — an amplitude threshold, not speech
    detection. It answers "how much of this is pauses"; ``vad.py`` answers
    where the speech actually is.
    '''
    rules = config["validation"]
    magnitude = np.abs(audio)
    if magnitude.size == 0:
        return {"clipping_ratio": 0.0, "silence_ratio": 0.0}
    return {
        "clipping_ratio": round(float(np.mean(magnitude >= rules["clipping_threshold"])), 6),
        "silence_ratio": round(float(np.mean(magnitude <= rules["silence_threshold"])), 6),
    }


if __name__ == "__main__":
    import json
    import sys

    from common import load_config

    settings = load_config()
    for argument in sys.argv[1:]:
        samples, rate, container, how = decode(argument, settings["validation"]["max_duration_sec"])
        print(argument, json.dumps(validate(samples, rate, settings)))
