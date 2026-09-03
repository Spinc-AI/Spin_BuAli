"""Shared fixtures. Nothing here loads weights, touches a GPU, or opens a socket."""
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import dataset  # noqa: E402


class FakeModel:
    """Stands in for an stt model class.

    Returns one word per window so the windowing and stitching can be checked
    exactly, and counts its calls so a test can prove a model was loaded once
    per device rather than once per recording.
    """

    def __init__(self, model_id="fake/model", device="cpu", texts=None, fail_on=None):
        self.model_id = model_id
        self.device = device
        self.loaded = False
        self.load_count = 0
        self.calls = []
        self._texts = list(texts or [])
        self._fail_on = fail_on

    def load(self):
        self.loaded = True
        self.load_count += 1

    def unload(self):
        self.loaded = False

    def transcribe(self, audio, sr, language=None):
        if not self.loaded:
            raise RuntimeError("transcribe before load")
        self.calls.append((len(audio), sr, language))
        if self._fail_on is not None and len(self.calls) == self._fail_on:
            raise RuntimeError("boom")
        if self._texts:
            return self._texts[(len(self.calls) - 1) % len(self._texts)]
        return f"window{len(self.calls)}"


@pytest.fixture
def factory():
    """A `model_factory` that hands every device the same recording stub."""
    made = []

    def build(key, device, **kwargs):
        model = FakeModel(model_id=f"fake/{key}", device=device, **kwargs)
        made.append(model)
        return model

    build.made = made
    return build


@pytest.fixture
def tone(tmp_path):
    """Write a WAV of a given length and return its path."""
    def make(name, seconds=1.0, sr=16000):
        path = tmp_path / name
        samples = np.zeros(int(seconds * sr), dtype=np.float32)
        sf.write(str(path), samples, sr)
        return path
    return make


@pytest.fixture
def items(tone):
    return [
        dataset.Item("A1", tone("A1.wav", 1.0), "There is a 6 mm stone in the right kidney"),
        dataset.Item("A2", tone("A2.wav", 1.0), "The urinary bladder is normal"),
    ]
