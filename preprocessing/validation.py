'''Stage 1 — validation.

Decides whether a file is worth processing at all: can it be decoded, does it
contain samples, are they finite, and is its length plausible. Everything
downstream may assume a file that got past here is real audio.

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
    return {"duration_sec": round(duration, 6), "peak": round(peak, 6), "rms": round(rms, 6)}


if __name__ == "__main__":
    import json
    import sys

    from common import load_config

    settings = load_config()
    for argument in sys.argv[1:]:
        samples, rate, container, how = decode(argument, settings["validation"]["max_duration_sec"])
        print(argument, json.dumps(validate(samples, rate, settings)))
