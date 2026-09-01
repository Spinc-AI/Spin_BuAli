'''Stage 2 — standardization to WAV / PCM 16-bit / mono / 16 kHz.

The conversion the task specifies, and the only stage that writes a full-length
derived copy. It changes format, never content: no denoising, no gain, no silence
removal. The one thing it can alter — samples clipped because resampling nudged a
peak past full scale — is counted and reported rather than applied silently.

Standalone:

    python standardization.py recording.mp3 out/standardized.wav
'''

import math
import os

import numpy as np

from common import write_wav

NAME = "standardization"
REQUIRES = ("validation",)


def run(context):
    '''Standardize the decoded audio and write it next to the other artefacts.'''
    config = context.config
    standardized, record = standardize(context.audio, context.sample_rate, config)

    context.standardized = standardized
    if context.output_dir:
        name = config["output"]["standardized_name"]
        write_wav(os.path.join(context.output_dir, name), standardized, config)
        record["path"] = name
    return record


def resample(audio, sr, target_sr):
    '''Resample a mono float32 array, returning it with the method that was used.

    torchaudio is the preferred path because it is exactly what the STT service
    runs on the same audio (``stt/app/model.py::resample``), so the derived WAV
    matches what the model would have produced from the original. scipy and a
    plain-numpy Fourier resampler stand in when torchaudio is not installed; the
    method is written into ``segments.json`` either way.
    '''
    if sr == target_sr:
        return audio, "none"

    try:
        import torch
        import torchaudio

        tensor = torch.from_numpy(np.asarray(audio, np.float32)).unsqueeze(0)
        out = torchaudio.functional.resample(tensor, sr, target_sr)
        return out.squeeze(0).numpy(), "torchaudio"
    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(sr, target_sr)
        out = resample_poly(audio, target_sr // divisor, sr // divisor)
        return np.asarray(out, np.float32), "scipy.resample_poly"
    except ImportError:
        pass

    return _resample_fft(audio, sr, target_sr), "numpy.fft"


def _resample_fft(audio, sr, target_sr):
    '''Band-limited resampling in the frequency domain — the dependency-free path.'''
    n_in = len(audio)
    n_out = int(round(n_in * target_sr / sr))
    if n_out <= 0:
        return np.zeros(0, np.float32)
    spectrum = np.fft.rfft(audio)
    resized = np.zeros(n_out // 2 + 1, complex)
    keep = min(len(spectrum), len(resized))
    resized[:keep] = spectrum[:keep]
    out = np.fft.irfft(resized, n_out) * (n_out / n_in)
    return np.asarray(out, np.float32)


def standardize(audio, sr, config):
    '''Convert decoded audio to the target rate/channel count without touching gain.

    Returns the standardized array plus a record of what the conversion did.
    Resampling can nudge peaks a hair past full scale; those samples are clipped
    (PCM 16-bit has nowhere else to put them) and counted, because the pipeline is
    not allowed to change level silently.
    '''
    target_sr = config["audio"]["target_sample_rate"]
    audio, method = resample(audio, sr, target_sr)

    clipped = int(np.count_nonzero(np.abs(audio) > 1.0))
    if clipped:
        audio = np.clip(audio, -1.0, 1.0)

    record = {
        "sample_rate": target_sr,
        "channels": config["audio"]["target_channels"],
        "subtype": config["audio"]["target_subtype"],
        "format": config["audio"]["target_format"],
        "duration_sec": round(len(audio) / target_sr, 6),
        "resampler": method,
        "resampled_from_sample_rate": sr,
        "clipped_samples": clipped,
        "gain_applied_db": 0.0,
        "denoise_applied": False,
        "silence_removed": False,
    }
    return np.asarray(audio, np.float32), record


if __name__ == "__main__":
    import json
    import sys

    from common import decode, load_config

    settings = load_config()
    samples, rate, _, _ = decode(sys.argv[1])
    converted, summary = standardize(samples, rate, settings)
    write_wav(sys.argv[2], converted, settings)
    print(json.dumps(summary, indent=2))
