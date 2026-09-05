"""The dataset finder. Used to be a string embedded in the notebook, which
meant a bug in it could only be fixed by regenerating and re-uploading the
whole file. It is a normal module now, tested like one."""
import pytest

import kaggle_dataset as kd


class TestFindLabels:
    def test_finds_a_labels_csv_one_level_deep(self, tmp_path):
        (tmp_path / "dataset").mkdir()
        (tmp_path / "dataset" / "labels.csv").write_text("x", encoding="utf-8")
        found = kd.find_labels(roots=[tmp_path])
        assert found == [tmp_path / "dataset" / "labels.csv"]

    def test_finds_a_labels_csv_three_levels_deep(self, tmp_path):
        """The bug this guards against: an old two-level-only glob missed
        exactly this layout, which is what an extra folder inside a zip
        produces."""
        deep = tmp_path / "spin-buali-dataset" / "Spin_BuAli_DataSet" / "Small_Demo"
        deep.mkdir(parents=True)
        (deep / "labels.csv").write_text("x", encoding="utf-8")
        assert kd.find_labels(roots=[tmp_path]) == [deep / "labels.csv"]

    def test_matches_the_filename_case_insensitively(self, tmp_path):
        (tmp_path / "LABELS.CSV").write_text("x", encoding="utf-8")
        assert len(kd.find_labels(roots=[tmp_path])) == 1

    def test_shallowest_match_comes_first(self, tmp_path):
        (tmp_path / "labels.csv").write_text("shallow", encoding="utf-8")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "labels.csv").write_text("deep", encoding="utf-8")
        found = kd.find_labels(roots=[tmp_path])
        assert found[0] == tmp_path / "labels.csv"

    def test_a_missing_root_is_not_an_error(self, tmp_path):
        assert kd.find_labels(roots=[tmp_path / "does-not-exist"]) == []

    def test_no_duplicates_when_two_roots_overlap(self, tmp_path):
        (tmp_path / "labels.csv").write_text("x", encoding="utf-8")
        assert len(kd.find_labels(roots=[tmp_path, tmp_path])) == 1


class TestResolve:
    def test_an_override_wins_even_if_the_search_would_find_something_else(self, tmp_path):
        (tmp_path / "labels.csv").write_text("x", encoding="utf-8")
        override = tmp_path / "elsewhere.csv"
        override.write_text("y", encoding="utf-8")
        assert kd.resolve(roots=[tmp_path], override=str(override)) == override

    def test_a_bad_override_fails_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="override"):
            kd.resolve(override=str(tmp_path / "nope.csv"))

    def test_nothing_found_names_what_is_actually_there(self, tmp_path):
        (tmp_path / "some_other_file.txt").write_text("x", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="some_other_file.txt"):
            kd.resolve(roots=[tmp_path])


class TestCountAudio:
    def test_counts_without_doubling_on_a_case_insensitive_match(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"x")
        (tmp_path / "b.MP3").write_bytes(b"x")
        assert kd.count_audio(tmp_path) == 2

    def test_ignores_non_audio_files(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"x")
        (tmp_path / "labels.csv").write_text("x", encoding="utf-8")
        assert kd.count_audio(tmp_path) == 1
