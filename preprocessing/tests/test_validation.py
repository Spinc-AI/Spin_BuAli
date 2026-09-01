'''Stage 1 — validation.'''

import numpy as np
import pytest

import audio_preprocessing as ap
from common import PreprocessingError
from conftest import speech_like, write_source


def test_measurements_describe_the_decoded_audio(tmp_path, config):
    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(5.0, sr), sr, subtype="PCM_16")
    audio, rate, _, _ = ap.validation.decode(source)

    measured = ap.validation.validate(audio, rate, config)

    assert measured["duration_sec"] == pytest.approx(5.0, abs=1e-3)
    assert 0.4 < measured["peak"] <= 1.0
    assert 0 < measured["rms"] < measured["peak"]


def test_empty_file_is_rejected(tmp_path, config):
    source = write_source(tmp_path / "empty.wav", np.zeros(0, np.float32), 16000, subtype="PCM_16")
    with pytest.raises(PreprocessingError):
        ap.process_file(source, str(tmp_path / "out"), config)


def test_silent_file_is_rejected(tmp_path, config):
    source = write_source(
        tmp_path / "silent.wav", np.zeros(16000 * 5, np.float32), 16000, subtype="PCM_16"
    )
    with pytest.raises(PreprocessingError, match="silent"):
        ap.process_file(source, str(tmp_path / "out"), config)


def test_too_short_file_is_rejected(tmp_path, config):
    source = write_source(tmp_path / "blip.wav", speech_like(0.05, 16000), 16000, subtype="PCM_16")
    with pytest.raises(PreprocessingError, match="minimum"):
        ap.process_file(source, str(tmp_path / "out"), config)


def test_too_long_file_is_refused_from_its_header(tmp_path, config):
    '''The length check must not require decoding the whole file into memory first.'''
    sr = 16000
    source = write_source(tmp_path / "long.wav", speech_like(30.0, sr), sr, subtype="PCM_16")
    config["validation"]["max_duration_sec"] = 10.0

    with pytest.raises(PreprocessingError, match="maximum"):
        ap.process_file(source, str(tmp_path / "out"), config)


def test_non_finite_samples_are_rejected(config):
    audio = np.array([0.5, np.nan, 0.5] * 16000, np.float32)
    with pytest.raises(PreprocessingError, match="NaN"):
        ap.validation.validate(audio, 16000, config)


def test_undecodable_file_is_rejected(tmp_path, config):
    source = tmp_path / "broken.wav"
    source.write_bytes(b"this is not audio")
    with pytest.raises(PreprocessingError, match="could not decode"):
        ap.process_file(str(source), str(tmp_path / "out"), config)
