'''Stage 2 — standardization, and the guarantee that the original is untouched.'''

import os
import wave

import numpy as np
import pytest
import soundfile as sf

import audio_preprocessing as ap
from common import file_sha256, probe
from conftest import speech_like, write_source


# ============================================================
# Acceptance: every output is WAV / PCM 16-bit / mono / 16 kHz
# ============================================================

def test_standardized_and_chunks_are_pcm16_mono_16k(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)

    paths = [os.path.join(result["output_dir"], "standardized.wav")]
    paths += [os.path.join(result["output_dir"], c["file"]) for c in result["chunks"]]
    assert len(paths) > 1

    for path in paths:
        info = probe(path)
        assert info["format"] == "WAV"
        assert info["subtype"] == "PCM_16"
        assert info["channels"] == 1
        assert info["sample_rate"] == 16000

    with wave.open(paths[0]) as handle:
        assert handle.getsampwidth() == 2


def test_duration_survives_standardization(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    assert result["standardized"]["duration_sec"] == pytest.approx(70.0, abs=0.01)
    assert result["source"]["sample_rate"] == 44100
    assert result["source"]["channels"] == 2


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_any_input_rate_lands_on_16k(tmp_path, config, rate):
    source = write_source(tmp_path / f"{rate}.wav", speech_like(30.0, rate), rate, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert result["standardized"]["sample_rate"] == 16000
    assert result["standardized"]["duration_sec"] == pytest.approx(30.0, abs=0.01)
    assert probe(os.path.join(result["output_dir"], "standardized.wav"))["sample_rate"] == 16000


def test_stereo_is_averaged_to_mono(tmp_path, config):
    '''Downmixing must average the channels, matching what the STT service does.'''
    sr = 16000
    left = speech_like(5.0, sr)
    right = -left
    path = str(tmp_path / "stereo.wav")
    sf.write(path, np.stack([left, right], axis=1), sr, subtype="PCM_16")

    audio, rate, _, _ = ap.validation.decode(path)
    assert np.max(np.abs(audio)) < 1e-4  # opposite channels average to silence


# ============================================================
# Acceptance: the original file is never touched
# ============================================================

def test_original_file_is_left_byte_identical(tmp_path, source, config):
    before = file_sha256(source)
    before_mtime = os.path.getmtime(source)

    ap.process_file(source, str(tmp_path / "out"), config)

    assert file_sha256(source) == before
    assert os.path.getmtime(source) == before_mtime


def test_outputs_live_outside_the_source_directory(tmp_path, source, config):
    listing_before = sorted(os.listdir(os.path.dirname(source)))
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert sorted(os.listdir(os.path.dirname(source))) == listing_before
    assert not result["output_dir"].startswith(os.path.dirname(source) + os.sep)


def test_silence_is_kept_and_gain_is_untouched(tmp_path, config):
    '''Silence must survive into the standardized copy — VAD only annotates it.'''
    sr = 16000
    audio = speech_like(4.0, sr, silence_head=3.0, silence_tail=3.0)
    source = write_source(tmp_path / "gap.wav", audio, sr, subtype="PCM_16")

    result = ap.process_file(source, str(tmp_path / "out"), config)
    standardized, _ = sf.read(
        os.path.join(result["output_dir"], "standardized.wav"), dtype="float32"
    )

    assert len(standardized) == pytest.approx(10.0 * sr, abs=2)
    assert np.max(np.abs(standardized[: int(2.5 * sr)])) < 1e-4
    assert result["standardized"]["gain_applied_db"] == 0.0
    assert result["standardized"]["denoise_applied"] is False
    assert result["standardized"]["silence_removed"] is False
    assert result["vad"]["applied_to_audio"] is False
    # Same rate in and out, so the samples should be the untouched originals.
    assert np.max(np.abs(standardized[: len(audio)] - audio)) < 1e-4


def test_policy_cannot_be_switched_off_in_the_config(config):
    '''The no-denoise / no-gain promises are not knobs a config may flip.'''
    from common import check_config

    config["policy"]["denoise"] = True
    with pytest.raises(ValueError, match="policy.denoise"):
        check_config(config)


# ============================================================
# Resampling
# ============================================================

def test_every_resampler_preserves_length_and_level(config):
    '''The fallbacks must agree with torchaudio on duration and rough loudness.'''
    sr, target = 44100, 16000
    audio = speech_like(3.0, sr)

    outputs = {}
    for name, function in (
        ("fft", ap.standardization._resample_fft),
    ):
        outputs[name] = function(audio, sr, target)

    resampled, method = ap.standardization.resample(audio, sr, target)
    outputs[method] = resampled

    for name, out in outputs.items():
        assert len(out) == pytest.approx(3.0 * target, abs=2), name
        assert np.sqrt(np.mean(out**2)) == pytest.approx(
            np.sqrt(np.mean(audio**2)), rel=0.1
        ), name


def test_resampling_is_a_no_op_at_the_target_rate(config):
    audio = speech_like(2.0, 16000)
    out, method = ap.standardization.resample(audio, 16000, 16000)
    assert method == "none"
    assert out is audio
