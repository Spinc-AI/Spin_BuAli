'''The runner — stage selection, batch runs, the report, and the CLI.

This is the file that checks you can run the six stages together or pick a
subset: skipping VAD must leave it out of the metadata entirely, not leave an
empty stub behind.
'''

import csv
import json
import os

import pytest

import audio_preprocessing as ap
import report as report_stage
from conftest import speech_like, write_source


# ============================================================
# Stage selection
# ============================================================

def test_default_selection_runs_all_six():
    assert [stage.NAME for stage in ap.select_stages()] == [
        "validation", "standardization", "vad", "chunking", "timestamps", "report",
    ]


def test_skip_removes_only_that_stage():
    names = [stage.NAME for stage in ap.select_stages(skip=["vad"])]
    assert "vad" not in names
    assert names == ["validation", "standardization", "chunking", "timestamps", "report"]


def test_only_keeps_the_named_stages_and_their_prerequisites():
    names = [stage.NAME for stage in ap.select_stages(only=["standardization", "timestamps"])]
    assert names == ["validation", "standardization", "timestamps"]


@pytest.mark.parametrize(
    "selection, message",
    [
        ({"only": ["chunking"]}, "needs standardization"),
        ({"skip": ["validation"]}, "cannot be skipped"),
        ({"skip": ["nonsense"]}, "unknown stage"),
        ({"only": ["nonsense"]}, "unknown stage"),
        ({"only": ["vad"], "skip": ["report"]}, "not both"),
    ],
)
def test_impossible_selections_are_refused(selection, message):
    with pytest.raises(ValueError, match=message):
        ap.select_stages(**selection)


# ============================================================
# Skipping a stage removes it from the metadata
# ============================================================

def test_skipping_vad_leaves_no_vad_key_in_the_metadata(tmp_path, source, config):
    stages = ap.select_stages(skip=["vad"])
    result = ap.process_file(source, str(tmp_path / "out"), config, stages=stages)

    with open(os.path.join(result["output_dir"], "segments.json"), encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "vad" not in payload
    assert "chunks" in payload
    assert "standardized" in payload
    assert payload["stages_run"] == ["validation", "standardization", "chunking", "timestamps", "report"]
    assert payload["checks"]["passed"] is True


def test_skipping_chunking_leaves_no_chunks(tmp_path, source, config):
    stages = ap.select_stages(skip=["chunking"])
    result = ap.process_file(source, str(tmp_path / "out"), config, stages=stages)

    with open(os.path.join(result["output_dir"], "segments.json"), encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "chunks" not in payload
    assert "chunking" not in payload
    assert "vad" in payload
    assert not os.path.isdir(os.path.join(result["output_dir"], "chunks"))


def test_validation_only_checks_without_writing_audio(tmp_path, source, config):
    stages = ap.select_stages(only=["validation"])
    result = ap.process_file(source, str(tmp_path / "out"), config, stages=stages)

    assert "standardized" not in result
    assert "chunks" not in result
    assert result["source"]["duration_sec"] == pytest.approx(70.0, abs=0.01)
    assert not os.path.isdir(result["output_dir"] or "")


def test_skipped_stage_still_leaves_the_others_intact(tmp_path, source, config):
    with_vad = ap.process_file(source, str(tmp_path / "on"), config)
    without = ap.process_file(
        source, str(tmp_path / "off"), config, stages=ap.select_stages(skip=["vad"])
    )

    assert [(c["start_sec"], c["end_sec"]) for c in without["chunks"]] == [
        (c["start_sec"], c["end_sec"]) for c in with_vad["chunks"]
    ]


# ============================================================
# Batch runs and the report
# ============================================================

def test_batch_run_writes_a_report_row_per_file(tmp_path, config):
    sr = 16000
    good = write_source(tmp_path / "good.wav", speech_like(40.0, sr), sr, subtype="PCM_16")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")

    output = str(tmp_path / "out")
    results, rows = ap.process_all([good, str(bad)], output, config)
    path = report_stage.write_report(os.path.join(output, "preprocessing_report.csv"), rows)

    assert len(results) == 1
    assert [row["status"] for row in rows] == ["ok", "failed"]
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["subtype"] == "PCM_16"
    assert rows[0]["chunk_max_sec"] <= config["chunking"]["max_duration_sec"]
    assert rows[0]["chunk_strategy"] == "adaptive"
    assert rows[1]["error"]

    with open(path, encoding="utf-8-sig", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert set(written[0]) == set(report_stage.REPORT_COLUMNS)


def test_report_columns_are_blank_for_stages_that_did_not_run(tmp_path, config):
    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(40.0, sr), sr, subtype="PCM_16")

    _, rows = ap.process_all(
        [source], str(tmp_path / "out"), config, stages=ap.select_stages(skip=["vad"])
    )

    assert rows[0]["vad_backend"] == ""
    assert rows[0]["vad_segments"] == ""
    assert rows[0]["chunk_count"] > 0
    assert "vad" not in rows[0]["stages"]


def test_report_appends_across_runs(tmp_path, config):
    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(30.0, sr), sr, subtype="PCM_16")
    path = str(tmp_path / "report.csv")

    for _ in range(2):
        _, rows = ap.process_all([source], str(tmp_path / "out"), config)
        report_stage.write_report(path, rows, append=True)

    with open(path, encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


# ============================================================
# CLI
# ============================================================

def test_cli_processes_a_directory(tmp_path):
    sr = 22050
    inputs = tmp_path / "in"
    inputs.mkdir()
    # Different lengths: identical content would hash to one id and one output.
    for name, seconds in (("a.wav", 30.0), ("b.wav", 45.0)):
        write_source(inputs / name, speech_like(seconds, sr), sr, subtype="PCM_16")

    output = tmp_path / "out"
    code = ap.main([str(inputs), "-o", str(output)])

    assert code == 0
    assert os.path.isfile(output / "preprocessing_report.csv")
    produced = [name for name in os.listdir(output) if os.path.isdir(output / name)]
    assert len(produced) == 2
    for directory in produced:
        assert os.path.isfile(output / directory / "standardized.wav")
        assert os.path.isfile(output / directory / "segments.json")
        assert os.listdir(output / directory / "chunks")


def test_cli_skip_reaches_the_output(tmp_path):
    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(40.0, sr), sr, subtype="PCM_16")
    output = tmp_path / "out"

    assert ap.main([source, "-o", str(output), "--skip", "vad"]) == 0

    directory = [name for name in os.listdir(output) if os.path.isdir(output / name)][0]
    with open(output / directory / "segments.json", encoding="utf-8") as handle:
        assert "vad" not in json.load(handle)


def test_cli_strategy_override(tmp_path):
    sr = 16000
    source = write_source(tmp_path / "clip.wav", speech_like(90.0, sr), sr, subtype="PCM_16")
    output = tmp_path / "out"

    assert ap.main([source, "-o", str(output), "--strategy", "fixed"]) == 0

    directory = [name for name in os.listdir(output) if os.path.isdir(output / name)][0]
    with open(output / directory / "segments.json", encoding="utf-8") as handle:
        assert json.load(handle)["chunking"]["decision"]["strategy"] == "fixed"


def test_cli_rejects_an_impossible_selection(tmp_path):
    source = write_source(tmp_path / "clip.wav", speech_like(30.0, 16000), 16000, subtype="PCM_16")
    assert ap.main([source, "-o", str(tmp_path / "out"), "--only", "chunking"]) == 2


def test_cli_lists_the_stages(capsys):
    assert ap.main(["--list-stages"]) == 0
    printed = capsys.readouterr().out
    for name in ap.STAGE_NAMES:
        assert name in printed


def test_cli_reports_failure_with_a_nonzero_exit(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"nope")
    assert ap.main([str(bad), "-o", str(tmp_path / "out")]) == 1


class TestDuplicateDetection:
    """Found by content, not by name — which is the case that matters, since
    renaming a file is exactly how a duplicate gets past a filename check."""

    def test_the_same_audio_under_two_names_is_linked(self, tmp_path, config):
        source = tmp_path / "in"
        source.mkdir()
        audio = speech_like(2.0, 16000)
        write_source(source / "original.wav", audio, 16000)
        write_source(source / "renamed_copy.wav", audio, 16000)

        results, rows = ap.process_all(sorted(source.iterdir()), tmp_path / "out", config)
        duplicates = [r.get("duplicate_of") for r in results]
        assert duplicates.count(None) == 1, "exactly one is the original"
        assert any(d for d in duplicates), "the other names it"

    def test_different_audio_is_not_flagged(self, tmp_path, config):
        source = tmp_path / "in"
        source.mkdir()
        write_source(source / "one.wav", speech_like(2.0, 16000), 16000)
        write_source(source / "two.wav", speech_like(3.0, 16000), 16000)

        results, _ = ap.process_all(sorted(source.iterdir()), tmp_path / "out", config)
        assert all(r.get("duplicate_of") is None for r in results)

    def test_the_report_carries_the_quality_columns(self, tmp_path, config):
        source = tmp_path / "in"
        source.mkdir()
        write_source(source / "one.wav", speech_like(2.0, 16000), 16000)

        _, rows = ap.process_all(sorted(source.iterdir()), tmp_path / "out", config)
        assert {"clipping_ratio", "silence_ratio", "duplicate_of"} <= set(rows[0])
