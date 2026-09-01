'''Fixtures shared by the per-stage test modules.

Synthetic audio only — no model weights, no network, no sample files in the repo.
'''

import numpy as np
import pytest
import soundfile as sf

from common import load_config

# Timestamps are written in milliseconds, so comparisons against real audio
# allow that rounding plus the sample a boundary lands on.
MILLISECOND = 2e-3


@pytest.fixture
def config():
    '''The shipped configuration, so the tests check what production runs.'''
    return load_config()


def speech_like(duration, sr, silence_head=0.0, silence_tail=0.0):
    '''A tone burst padded with digital silence — stands in for a spoken phrase.'''
    times = np.arange(int(round(duration * sr))) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * times) * (1 + 0.5 * np.sin(2 * np.pi * 3 * times))
    return np.concatenate(
        [
            np.zeros(int(round(silence_head * sr)), np.float32),
            tone.astype(np.float32),
            np.zeros(int(round(silence_tail * sr)), np.float32),
        ]
    )


def phrases(pattern, sr):
    '''Audio built from alternating (speech, pause) durations in seconds.

    Lets a test place pauses where it wants them and then check that chunking
    actually cut there.
    '''
    parts = []
    for kind, seconds in pattern:
        if kind == "speech":
            parts.append(speech_like(seconds, sr))
        else:
            parts.append(np.zeros(int(round(seconds * sr)), np.float32))
    return np.concatenate(parts)


def write_source(path, audio, sr, channels=1, subtype="PCM_24"):
    '''Write a deliberately non-standard source file (wrong rate/width/channels).'''
    data = np.stack([audio, audio], axis=1) if channels == 2 else audio
    sf.write(str(path), data, sr, subtype=subtype)
    return str(path)


def no_silero(*args, **kwargs):
    '''Stand in for a machine with no Silero weights, so the fallback is exercised.'''
    raise RuntimeError("silero unavailable in this test")


@pytest.fixture
def source(tmp_path):
    '''A 70s stereo 44.1 kHz 24-bit file — nothing about it already matches the target.'''
    sr = 44100
    audio = speech_like(68.0, sr, silence_head=1.0, silence_tail=1.0)
    # In its own directory, so a test can assert the pipeline wrote nothing beside it.
    directory = tmp_path / "src"
    directory.mkdir(exist_ok=True)
    return write_source(directory / "recording.wav", audio, sr, channels=2, subtype="PCM_24")
