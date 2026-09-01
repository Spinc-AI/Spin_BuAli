'''Stage 3 — VAD: padding, honest backend reporting, and never touching the audio.

Silero is a real speech model and will not fire on a synthetic tone, so these
tests do not assert what it detects. They pin what the pipeline is responsible
for: that the padding is applied, that ``speech_pad_ms`` reaches the library, and
that the backend actually used is reported truthfully.
'''

import pytest

import audio_preprocessing as ap
import vad as vad_stage
from conftest import no_silero, speech_like, write_source


# ============================================================
# Acceptance: VAD pads the start and end of speech
# ============================================================

def test_padding_widens_spans_and_merges_what_it_joins():
    '''The padding step itself: wider on both sides, clamped, no double regions.'''
    assert vad_stage._pad_spans([(3.0, 7.0)], 0.3, 10.0) == [(2.7, 7.3)]

    # Clamped at the edges rather than running off the file.
    assert vad_stage._pad_spans([(0.1, 9.9)], 0.5, 10.0) == [(0.0, 10.0)]

    # Two spans whose padding overlaps become one, not two overlapping regions.
    assert vad_stage._pad_spans([(1.0, 2.0), (2.4, 3.0)], 0.3, 10.0) == [(0.7, 3.3)]


def test_vad_pads_around_detected_speech(tmp_path, config, monkeypatch):
    '''End-to-end padding, on the backend whose detection we control.'''
    monkeypatch.setattr(vad_stage, "_silero_speech_spans", no_silero)

    sr = 16000
    speech_start, speech_end = 3.0, 7.0
    audio = speech_like(speech_end - speech_start, sr, silence_head=speech_start, silence_tail=3.0)
    source = write_source(tmp_path / "padded.wav", audio, sr, subtype="PCM_16")

    result = ap.process_file(source, str(tmp_path / "out"), config)
    vad = result["vad"]
    assert vad["backend"] == "energy"
    assert vad["segments"], "the energy backend found no speech in a clearly voiced clip"

    pad = config["vad"]["speech_pad_ms"] / 1000
    first, last = vad["segments"][0], vad["segments"][-1]
    assert first["start_sec"] < speech_start
    assert first["start_sec"] >= speech_start - pad - 0.15
    assert last["end_sec"] > speech_end
    assert last["end_sec"] <= speech_end + pad + 0.15


def test_silero_adapter_forwards_config_and_returns_seconds(config, monkeypatch):
    '''Whatever Silero decides, it must be asked with the configured settings.'''
    torch = pytest.importorskip("torch")
    captured = {}

    def fake_timestamps(tensor, model, **kwargs):
        captured.update(kwargs)
        return [{"start": 16000, "end": 32000}]

    module = pytest.importorskip("silero_vad")
    monkeypatch.setattr(module, "get_speech_timestamps", fake_timestamps)
    monkeypatch.setattr(module, "load_silero_vad", lambda *a, **k: object())

    spans = vad_stage._silero_speech_spans(speech_like(3.0, 16000), 16000, config["vad"])

    assert spans == [(1.0, 2.0)]
    assert captured["speech_pad_ms"] == config["vad"]["speech_pad_ms"]
    assert captured["threshold"] == config["vad"]["threshold"]
    assert captured["min_speech_duration_ms"] == config["vad"]["min_speech_duration_ms"]
    assert captured["min_silence_duration_ms"] == config["vad"]["min_silence_duration_ms"]
    assert captured["sampling_rate"] == 16000


# ============================================================
# Backend reporting
# ============================================================

def test_real_silero_runs_when_it_is_installed(tmp_path, config):
    '''With weights present the pipeline must reach Silero, not the fallback.'''
    pytest.importorskip("silero_vad")

    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(40.0, sr), sr, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert result["vad"]["backend"] == "silero"
    assert result["vad"]["error"] is None
    assert result["checks"]["passed"]


def test_missing_silero_falls_back_and_says_so(tmp_path, config, monkeypatch):
    monkeypatch.setattr(vad_stage, "_silero_speech_spans", no_silero)

    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(30.0, sr), sr, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert result["vad"]["backend"] == "energy"
    assert "silero unavailable" in result["vad"]["error"]
    assert result["vad"]["available"] is True


def test_fallback_can_be_turned_off(tmp_path, config, monkeypatch):
    monkeypatch.setattr(vad_stage, "_silero_speech_spans", no_silero)
    config["vad"]["fallback"] = "none"

    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(30.0, sr), sr, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert result["vad"]["available"] is False
    assert result["vad"]["backend"] is None
    assert result["vad"]["segments"] == []
    assert result["checks"]["passed"], "a VAD that found nothing must not fail the run"
