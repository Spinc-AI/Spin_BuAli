'''Stage 5 — segments.json and the acceptance checks it runs.'''

import json
import os

import pytest

import audio_preprocessing as ap
from common import probe
from conftest import MILLISECOND, speech_like, write_source


# ============================================================
# Acceptance: timestamps agree with the audio they describe
# ============================================================

def test_timestamps_match_the_written_chunks(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    duration = result["standardized"]["duration_sec"]

    for chunk in result["chunks"]:
        on_disk = probe(os.path.join(result["output_dir"], chunk["file"]))
        assert on_disk["duration_sec"] == pytest.approx(chunk["duration_sec"], abs=MILLISECOND)
        assert on_disk["duration_sec"] == pytest.approx(
            chunk["end_sec"] - chunk["start_sec"], abs=MILLISECOND
        )
        assert 0 <= chunk["start_sec"] < chunk["end_sec"] <= duration + MILLISECOND


def test_sample_indices_agree_with_the_seconds(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    sr = result["standardized"]["sample_rate"]

    for chunk in result["chunks"]:
        assert chunk["start_sample"] == pytest.approx(chunk["start_sec"] * sr, abs=1)
        assert chunk["end_sample"] == pytest.approx(chunk["end_sec"] * sr, abs=1)


def test_vad_segments_stay_inside_the_file(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    duration = result["standardized"]["duration_sec"]
    for segment in result["vad"]["segments"]:
        assert 0 <= segment["start_sec"] < segment["end_sec"] <= duration + MILLISECOND


def test_segments_json_is_written_and_self_consistent(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    with open(os.path.join(result["output_dir"], "segments.json"), encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["file_id"] == result["file_id"]
    assert payload["checks"]["passed"] is True
    assert payload["source"]["sha256"]
    assert len(payload["chunks"]) == len(result["chunks"])
    for chunk in payload["chunks"]:
        assert os.path.isfile(os.path.join(result["output_dir"], chunk["file"]))


def test_metadata_is_carried_through(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config, {"patient": "x", "study": "ct"})
    with open(os.path.join(result["output_dir"], "segments.json"), encoding="utf-8") as handle:
        assert json.load(handle)["metadata"] == {"patient": "x", "study": "ct"}


def test_policy_is_recorded_in_every_output(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    assert result["policy"]["preserve_original"] is True
    assert result["policy"]["denoise"] is False
    assert result["policy"]["remove_silence"] is False


# ============================================================
# The checks themselves
# ============================================================

def test_verify_catches_a_chunk_that_is_too_long(tmp_path, source, config):
    '''The acceptance check must fail on a bad artefact, not just pass on good ones.'''
    result = ap.process_file(source, str(tmp_path / "out"), config)
    config["chunking"]["max_duration_sec"] = 1.0

    problems = ap.timestamps.verify(result, config, result["output_dir"])

    assert problems
    assert any("maximum" in problem for problem in problems)


def test_verify_catches_a_changed_original(tmp_path, source, config):
    result = ap.process_file(source, str(tmp_path / "out"), config)
    with open(source, "ab") as handle:
        handle.write(b"tampered")

    problems = ap.timestamps.verify(result, config, result["output_dir"])

    assert any("original file changed" in problem for problem in problems)


def test_a_rerun_does_not_leave_orphaned_chunks(tmp_path, config):
    '''A shorter second run must not leave the first run's extra chunk files behind.'''
    sr = 16000
    long_audio = speech_like(90.0, sr)
    source = write_source(tmp_path / "clip.wav", long_audio, sr, subtype="PCM_16")
    output = str(tmp_path / "out")

    first = ap.process_file(source, output, config)
    written_first = len(os.listdir(os.path.join(first["output_dir"], "chunks")))

    config["chunking"]["max_duration_sec"] = 95.0
    config["chunking"]["target_duration_sec"] = 95.0
    config["chunking"]["min_duration_sec"] = 10.0
    second = ap.process_file(source, output, config)

    assert len(second["chunks"]) < written_first
    assert len(os.listdir(os.path.join(second["output_dir"], "chunks"))) == len(second["chunks"])
