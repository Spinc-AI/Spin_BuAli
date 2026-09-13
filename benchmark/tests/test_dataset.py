"""Finding the recordings and their labels, and decoding audio."""
import json

import numpy as np
import pytest
import soundfile as sf

import dataset


class TestFromDirectory:
    def test_audio_is_paired_with_matching_truth_by_stem(self, tmp_path):
        audio_dir, truth_dir = tmp_path / "audio", tmp_path / "truth"
        audio_dir.mkdir()
        truth_dir.mkdir()
        sf.write(str(audio_dir / "A1.wav"), np.zeros(16000, np.float32), 16000)
        (truth_dir / "A1.txt").write_text("the liver is normal", encoding="utf-8")

        items = dataset.from_directory(audio_dir, truth_dir)
        assert [item.asset_id for item in items] == ["A1"]
        assert items[0].reference == "the liver is normal"
        assert items[0].labelled

    def test_audio_without_a_label_is_unlabelled_not_an_error(self, tmp_path):
        """Ground truth arrives slower than audio does. A recording with no
        label yet is still worth transcribing -- that draft is what becomes
        the label."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        sf.write(str(audio_dir / "A2.wav"), np.zeros(16000, np.float32), 16000)

        items = dataset.from_directory(audio_dir, tmp_path / "truth")
        assert len(items) == 1 and not items[0].labelled

    def test_non_audio_files_are_ignored(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        sf.write(str(audio_dir / "A1.wav"), np.zeros(16000, np.float32), 16000)
        (audio_dir / "notes.md").write_text("x", encoding="utf-8")
        assert len(dataset.from_directory(audio_dir)) == 1


class TestFromJson:
    def test_inline_and_file_references_both_work(self, tmp_path):
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        sf.write(str(tmp_path / "A2.wav"), np.zeros(16000, np.float32), 16000)
        (tmp_path / "A2.txt").write_text("from a file", encoding="utf-8")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([
            {"asset_id": "A1", "audio": "A1.wav", "reference": "inline text"},
            {"asset_id": "A2", "audio": "A2.wav", "reference": "A2.txt"},
        ]), encoding="utf-8")

        items = dataset.from_json(manifest)
        assert items[0].reference == "inline text"
        assert items[1].reference == "from a file"

    def test_paths_resolve_relative_to_the_manifest(self, tmp_path):
        """So a dataset directory can be moved -- or mounted on Kaggle at a
        different path -- without editing every entry."""
        (tmp_path / "clips").mkdir()
        sf.write(str(tmp_path / "clips" / "A1.wav"), np.zeros(16000, np.float32), 16000)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{"audio": "clips/A1.wav"}]), encoding="utf-8")

        items = dataset.from_json(manifest)
        assert items[0].audio.is_file()
        assert items[0].asset_id == "A1", "asset_id defaults to the filename stem"

    def test_modality_and_regions_are_read_when_present(self, tmp_path):
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([
            {"asset_id": "A1", "audio": "A1.wav", "modality": "Ultrasound",
             "regions": ["Abdomen", "Pelvis"]},
        ]), encoding="utf-8")

        items = dataset.from_json(manifest)
        assert items[0].modality == "Ultrasound"
        assert items[0].regions == ("Abdomen", "Pelvis")

    def test_modality_and_regions_default_to_empty_when_absent(self, tmp_path):
        """Most manifests will not have these yet -- absence must not error."""
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{"asset_id": "A1", "audio": "A1.wav"}]), encoding="utf-8")

        items = dataset.from_json(manifest)
        assert items[0].modality is None
        assert items[0].regions == ()


class TestFromCsv:
    def _write_csv(self, tmp_path, header, *rows):
        import csv

        path = tmp_path / "labels.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_modality_and_region_are_read_when_present(self, tmp_path):
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        csv_path = self._write_csv(
            tmp_path, ["asset_id", "audio", "report", "modality", "region"],
            ["A1", "A1.wav", "the liver is normal", "Ultrasound", "Abdomen;Pelvis;Retroperitoneum"])

        items = dataset.from_csv(csv_path)
        assert items[0].modality == "Ultrasound"
        assert items[0].regions == ("Abdomen", "Pelvis", "Retroperitoneum")

    def test_a_csv_without_those_columns_still_loads(self, tmp_path):
        """The columns are new -- most labels.csv files, including this
        one until now, do not have them."""
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        csv_path = self._write_csv(
            tmp_path, ["asset_id", "audio", "report"], ["A1", "A1.wav", "the liver is normal"])

        items = dataset.from_csv(csv_path)
        assert items[0].modality is None
        assert items[0].regions == ()

    def test_a_blank_region_cell_is_no_regions_not_one_empty_region(self, tmp_path):
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        csv_path = self._write_csv(
            tmp_path, ["asset_id", "audio", "report", "modality", "region"],
            ["A1", "A1.wav", "text", "Ultrasound", ""])

        items = dataset.from_csv(csv_path)
        assert items[0].regions == ()


class TestDescribe:
    def test_counts_labelled_and_missing(self, tmp_path):
        sf.write(str(tmp_path / "A1.wav"), np.zeros(16000, np.float32), 16000)
        census = dataset.describe([
            dataset.Item("A1", tmp_path / "A1.wav", "text"),
            dataset.Item("A2", tmp_path / "A2.wav", None),
        ])
        assert census == {"items": 2, "labelled": 1, "unlabelled": 1, "missing_audio": ["A2"]}


class TestLoadAudio:
    def test_mono_float32_at_the_target_rate(self, tmp_path):
        path = tmp_path / "A1.wav"
        sf.write(str(path), np.zeros(8000, np.float32), 8000)
        audio, sr = dataset.load_audio(path)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) == pytest.approx(16000, rel=0.01)

    def test_stereo_is_mixed_down(self, tmp_path):
        path = tmp_path / "stereo.wav"
        sf.write(str(path), np.zeros((16000, 2), np.float32), 16000)
        audio, _ = dataset.load_audio(path)
        assert audio.ndim == 1

    def test_an_undecodable_file_says_which_one(self, tmp_path):
        path = tmp_path / "broken.wav"
        path.write_bytes(b"not audio")
        with pytest.raises(RuntimeError, match="broken.wav"):
            dataset.load_audio(path)
