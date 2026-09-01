'''Stage 4 — chunking: the length window, the overlap, and adaptive layout.'''

import os

import numpy as np
import pytest

import audio_preprocessing as ap
import chunking
from common import probe
from conftest import MILLISECOND, phrases, speech_like, write_source


def plan(duration, config, audio=None, sr=16000):
    '''Plan windows, dropping the decision record the tests do not need.'''
    windows, _ = chunking.plan_chunks(duration, config, None, audio, sr)
    return windows


# ============================================================
# Acceptance: no chunk exceeds the configured maximum
# ============================================================

def test_no_chunk_exceeds_the_maximum(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    limit = config["chunking"]["max_duration_sec"]
    for chunk in result["chunks"]:
        assert chunk["duration_sec"] <= limit
        assert probe(os.path.join(result["output_dir"], chunk["file"]))["duration_sec"] <= limit


@pytest.mark.parametrize("strategy", ["adaptive", "uniform", "fixed"])
@pytest.mark.parametrize("duration", [5.0, 25.0, 31.0, 70.0, 200.0, 601.3])
def test_every_strategy_respects_the_window_at_any_length(duration, strategy, config):
    config["chunking"]["strategy"] = strategy
    sr = 16000
    audio = speech_like(duration, sr) if strategy == "adaptive" else None

    windows = plan(duration, config, audio, sr)

    assert windows
    assert all(end - start <= config["chunking"]["max_duration_sec"] + 1e-9 for start, end in windows)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(duration)


# ============================================================
# Acceptance: chunks overlap so boundary words are not lost
# ============================================================

def test_consecutive_chunks_overlap(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    chunks = result["chunks"]
    assert len(chunks) >= 2

    wanted = config["chunking"]["overlap_sec"]
    for earlier, later in zip(chunks, chunks[1:]):
        assert later["start_sec"] < earlier["end_sec"]
        assert earlier["end_sec"] - later["start_sec"] == pytest.approx(wanted, abs=0.01)


def test_chunks_cover_the_whole_file_without_gaps(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    chunks = result["chunks"]
    assert chunks[0]["start_sec"] == 0.0
    assert chunks[-1]["end_sec"] == pytest.approx(result["standardized"]["duration_sec"], abs=0.01)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later["start_sec"] <= earlier["end_sec"]


# ============================================================
# Adaptive: the layout is chosen per recording
# ============================================================

def pauses_at(centres, duration, sr, pause=2.0):
    '''Audio of ``duration`` seconds that is silent only around ``centres``.'''
    pattern, cursor = [], 0.0
    for centre in centres:
        pattern.append(("speech", centre - pause / 2 - cursor))
        pattern.append(("pause", pause))
        cursor = centre + pause / 2
    pattern.append(("speech", duration - cursor))
    return phrases(pattern, sr)


def test_adaptive_cuts_at_the_pause_not_at_the_arithmetic_midpoint(config):
    '''With the count fixed, every boundary should land in a pause near it.'''
    sr = 16000
    # 100s at five chunks puts the even boundaries at 20/40/60/80. The pauses sit
    # a second off each of those — reachable, but only by actually listening.
    audio = pauses_at([21.0, 39.0, 61.0, 79.0], 100.0, sr)
    config["chunking"]["adaptive"]["max_extra_chunks"] = 0

    windows, decision = chunking.plan_chunks(len(audio) / sr, config, None, audio, sr)
    half = config["chunking"]["overlap_sec"] / 2
    boundaries = [end - half for _, end in windows[:-1]]

    assert decision["strategy"] == "adaptive"
    assert decision["listened_to_audio"] is True
    assert len(boundaries) == 4
    for boundary, pause in zip(boundaries, [21.0, 39.0, 61.0, 79.0]):
        assert abs(boundary - pause) <= 1.0, f"cut at {boundary:.2f}s, pause was at {pause}s"
    assert decision["regions"][0]["boundary_quietness"] > 0.95


def test_snapping_can_be_turned_off(config):
    '''A zero snap window must reproduce the even split exactly.'''
    sr = 16000
    audio = pauses_at([21.0, 39.0, 61.0, 79.0], 100.0, sr)
    config["chunking"]["adaptive"]["max_extra_chunks"] = 0
    config["chunking"]["adaptive"]["snap_window_sec"] = 0.0

    windows, _ = chunking.plan_chunks(len(audio) / sr, config, None, audio, sr)
    half = config["chunking"]["overlap_sec"] / 2
    boundaries = [end - half for _, end in windows[:-1]]

    for boundary, even in zip(boundaries, [20.0, 40.0, 60.0, 80.0]):
        assert boundary == pytest.approx(even, abs=0.05)


def test_adaptive_and_uniform_agree_when_there_is_nothing_to_hear(config):
    '''Uniform noise gives the snapper no reason to move a boundary anywhere.'''
    sr = 16000
    rng = np.random.default_rng(0)
    audio = (0.2 * rng.standard_normal(int(70 * sr))).astype(np.float32)

    config["chunking"]["strategy"] = "uniform"
    uniform = plan(70.0, config, None, sr)
    config["chunking"]["strategy"] = "adaptive"
    config["chunking"]["adaptive"]["snap_window_sec"] = 0.0
    adaptive = plan(70.0, config, audio, sr)

    assert len(adaptive) == len(uniform)
    for (a_start, a_end), (u_start, u_end) in zip(adaptive, uniform):
        assert a_start == pytest.approx(u_start, abs=0.05)
        assert a_end == pytest.approx(u_end, abs=0.05)


def test_adaptive_picks_the_chunk_count_per_file(config):
    '''Same length, same candidate counts — the pauses decide which one is used.'''
    sr = 16000
    # 100s admits either four chunks (boundaries at 25/50/75) or five (20/40/60/80).
    suits_four = pauses_at([25.0, 50.0, 75.0], 100.0, sr)
    suits_five = pauses_at([20.0, 40.0, 60.0, 80.0], 100.0, sr)

    _, four = chunking.plan_chunks(100.0, config, None, suits_four, sr)
    _, five = chunking.plan_chunks(100.0, config, None, suits_five, sr)

    assert four["regions"][0]["counts_considered"] == [4, 5]
    assert five["regions"][0]["counts_considered"] == [4, 5]
    assert four["regions"][0]["chunk_count"] == 4
    assert five["regions"][0]["chunk_count"] == 5


def test_adaptive_records_why_it_chose_that_layout(tmp_path, config):
    sr = 16000
    audio = phrases([("speech", 23.0), ("pause", 2.0)] * 3 + [("speech", 22.0)], sr)
    source = write_source(tmp_path / "paced.wav", audio, sr, subtype="PCM_16")

    result = ap.process_file(source, str(tmp_path / "out"), config)
    decision = result["chunking"]["decision"]["regions"][0]

    assert decision["chunk_count"] == len(result["chunks"])
    assert decision["counts_considered"]
    assert 0.0 <= decision["boundary_quietness"] <= 1.0
    assert decision["reason"]
    for rejected in decision["rejected"]:
        assert rejected["chunk_count"] != decision["chunk_count"]


def test_adaptive_falls_back_to_uniform_without_audio(config):
    '''Planning from a duration alone must still work, and say that it did.'''
    windows, decision = chunking.plan_chunks(97.0, config)

    assert decision["strategy"] == "uniform"
    assert decision["requested_strategy"] == "adaptive"
    assert decision["listened_to_audio"] is False
    assert all(end - start <= config["chunking"]["max_duration_sec"] for start, end in windows)


def test_snapping_never_produces_an_illegal_chunk(config):
    '''Even with an absurd snap window, the length limits must hold.'''
    sr = 16000
    audio = phrases([("speech", 4.0), ("pause", 1.0)] * 24, sr)
    config["chunking"]["adaptive"]["snap_window_sec"] = 60.0

    windows = plan(len(audio) / sr, config, audio, sr)

    for start, end in windows:
        assert end - start <= config["chunking"]["max_duration_sec"] + 1e-9
    for earlier, later in zip(windows, windows[1:]):
        assert later[0] < earlier[1]


# ============================================================
# Strategies
# ============================================================

def test_fixed_strategy_keeps_exact_target_lengths(config):
    config["chunking"]["strategy"] = "fixed"
    windows = plan(97.0, config)

    target = config["chunking"]["target_duration_sec"]
    assert all(end - start == pytest.approx(target) for start, end in windows[:-1])
    assert windows[-1][1] == pytest.approx(97.0)


def test_uniform_strategy_keeps_chunks_even(config):
    config["chunking"]["strategy"] = "uniform"
    windows = plan(97.0, config)

    lengths = [end - start for start, end in windows]
    assert max(lengths) - min(lengths) <= config["chunking"]["overlap_sec"] + 1e-9


def test_short_region_becomes_a_single_chunk(config):
    windows = plan(12.0, config, speech_like(12.0, 16000), 16000)
    assert windows == [(0.0, 12.0)]


# ============================================================
# Chunking over VAD regions
# ============================================================

def test_vad_chunking_uses_speech_regions_when_there_are_any(tmp_path, config, monkeypatch):
    import vad as vad_stage

    monkeypatch.setattr(vad_stage, "_silero_speech_spans", lambda *a, **k: [(5.0, 15.0), (30.0, 45.0)])
    config["chunking"]["source"] = "vad_segments"

    sr = 16000
    source = write_source(tmp_path / "speech.wav", speech_like(50.0, sr), sr, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    spans = [(c["start_sec"], c["end_sec"]) for c in result["chunks"]]
    assert spans == [(5.0, 15.0), (30.0, 45.0)]
    assert result["checks"]["passed"]


def test_vad_chunking_falls_back_when_no_speech_is_found(tmp_path, config, monkeypatch):
    '''Chunking over VAD regions must never produce an output with zero chunks.'''
    import vad as vad_stage

    monkeypatch.setattr(vad_stage, "_silero_speech_spans", lambda *a, **k: [])
    config["vad"]["fallback"] = "none"
    config["chunking"]["source"] = "vad_segments"

    sr = 16000
    source = write_source(tmp_path / "quiet.wav", speech_like(50.0, sr), sr, subtype="PCM_16")
    result = ap.process_file(source, str(tmp_path / "out"), config)

    assert result["vad"]["segments"] == []
    assert result["vad"]["chunking_fell_back_to_whole_file"] is True
    assert result["chunks"]
    assert result["chunks"][-1]["end_sec"] == pytest.approx(50.0, abs=MILLISECOND)
