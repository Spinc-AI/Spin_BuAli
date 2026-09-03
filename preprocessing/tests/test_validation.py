'''Stage 1 — validation.'''

import numpy as np
import pytest

import audio_preprocessing as ap
import validation
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


class TestQualityMeasurement:
    """Descriptive, never a verdict — none of these reject a file.

    They exist so a corpus can be characterised before anyone wonders why a
    model does badly on it.
    """

    @staticmethod
    def _config():
        from common import load_config

        return load_config()

    def test_clipping_is_counted_not_corrected(self):
        """A sample at full scale means the waveform was cut flat there. No
        later stage recovers it, so the only useful response is to report it."""
        audio = np.zeros(1000, dtype=np.float32)
        audio[:60] = 1.0                       # 6% pinned at full scale
        measured = validation.quality(audio, self._config())
        assert measured["clipping_ratio"] == pytest.approx(0.06, abs=0.001)

    def test_clean_audio_reports_no_clipping(self):
        audio = (0.5 * np.sin(np.linspace(0, 100, 1000))).astype(np.float32)
        assert validation.quality(audio, self._config())["clipping_ratio"] == 0.0

    def test_silence_is_the_share_below_the_threshold(self):
        audio = np.full(1000, 0.5, dtype=np.float32)
        audio[:250] = 0.0
        measured = validation.quality(audio, self._config())
        assert measured["silence_ratio"] == pytest.approx(0.25, abs=0.001)

    def test_silence_is_an_amplitude_test_not_speech_detection(self):
        """It answers "how much of this is pauses". vad.py answers where the
        speech is, and the two are not interchangeable."""
        quiet_but_present = np.full(1000, 0.005, dtype=np.float32)
        assert validation.quality(quiet_but_present, self._config())["silence_ratio"] == 1.0

    def test_measurement_never_rejects(self, tmp_path):
        """A heavily clipped file still validates — it is bad audio, not
        invalid audio, and the pipeline's job is to say so, not to refuse."""
        config = self._config()
        clipped = np.ones(int(1.0 * 16000), dtype=np.float32)
        measured = validation.validate(clipped, 16000, config)
        assert measured["clipping_ratio"] == 1.0

    def test_empty_audio_does_not_divide_by_zero(self):
        assert validation.quality(np.array([], dtype=np.float32),
                                  self._config()) == {"clipping_ratio": 0.0,
                                                      "silence_ratio": 0.0}

    def test_the_measurements_ride_along_with_validation(self):
        audio = (0.3 * np.sin(np.linspace(0, 50, 16000))).astype(np.float32)
        measured = validation.validate(audio, 16000, self._config())
        assert set(measured) >= {"duration_sec", "peak", "rms",
                                 "clipping_ratio", "silence_ratio"}
