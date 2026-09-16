"""`data.py`'s loading and splitting -- everything except `to_hf_dataset`,
which needs the `datasets` package and is exercised in
`test_to_hf_dataset.py` only when that import succeeds (see its own
docstring). Real audio files, real `labels.csv` parsing, via the
`tiny_dataset` fixture -- this reuses `benchmark/dataset.py` for real, not a
stub of it.
"""
import pytest

import data


class TestLoadItems:
    def test_loads_every_labelled_row(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        assert len(items) == 9
        assert all(item.labelled for item in items)

    def test_an_unlabelled_row_is_dropped_not_kept_as_an_empty_example(self, tiny_dataset):
        text = tiny_dataset.read_text(encoding="utf-8")
        text += "CLIP009,audio/CLIP000.wav,\n"  # reuses CLIP000's audio, blank report
        tiny_dataset.write_text(text, encoding="utf-8")

        items = data.load_items(tiny_dataset)
        assert len(items) == 9  # still 9, the unlabelled row never became a 10th
        assert "CLIP009" not in {item.asset_id for item in items}

    def test_a_labelled_row_with_missing_audio_raises(self, tiny_dataset):
        text = tiny_dataset.read_text(encoding="utf-8")
        text += "CLIPXXX,audio/does_not_exist.wav,some finding\n"
        tiny_dataset.write_text(text, encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="CLIPXXX"):
            data.load_items(tiny_dataset)


class TestTrainEvalSplit:
    def test_splits_nine_items_with_default_settings(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        train_items, eval_items = data.train_eval_split(items, eval_fraction=0.15, min_eval=1)
        assert len(train_items) + len(eval_items) == 9
        assert len(eval_items) >= 1
        assert len(train_items) >= 1

    def test_train_and_eval_never_share_an_item(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        train_items, eval_items = data.train_eval_split(items)
        train_ids = {item.asset_id for item in train_items}
        eval_ids = {item.asset_id for item in eval_items}
        assert not (train_ids & eval_ids)
        assert train_ids | eval_ids == {item.asset_id for item in items}

    def test_min_eval_is_a_floor_a_small_fraction_cannot_go_below(self, tiny_dataset):
        """0.15 of 9 rounds to 1 anyway -- the case that actually exercises
        the floor is a fraction that would otherwise round to 0."""
        items = data.load_items(tiny_dataset)
        _, eval_items = data.train_eval_split(items, eval_fraction=0.01, min_eval=3)
        assert len(eval_items) == 3

    def test_too_few_items_for_the_eval_floor_raises_rather_than_training_on_nothing(
            self, tiny_dataset):
        items = data.load_items(tiny_dataset)[:1]
        with pytest.raises(ValueError, match="not enough"):
            data.train_eval_split(items, min_eval=1)

    def test_the_same_seed_reproduces_the_same_split(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        first = data.train_eval_split(items, seed=7)
        second = data.train_eval_split(items, seed=7)
        assert [item.asset_id for item in first[0]] == [item.asset_id for item in second[0]]
        assert [item.asset_id for item in first[1]] == [item.asset_id for item in second[1]]

    def test_row_order_in_the_csv_does_not_change_the_split(self, tiny_dataset):
        """A labels.csv re-exported from a spreadsheet can reorder rows
        without changing their content -- the split must not care."""
        import random

        items = data.load_items(tiny_dataset)
        reordered = list(items)
        random.Random(1).shuffle(reordered)

        original_split = data.train_eval_split(items, seed=7)
        reordered_split = data.train_eval_split(reordered, seed=7)
        assert {i.asset_id for i in original_split[1]} == {i.asset_id for i in reordered_split[1]}

    def test_a_different_seed_can_produce_a_different_split(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        split_a = {item.asset_id for item in data.train_eval_split(items, seed=1)[1]}
        split_b = {item.asset_id for item in data.train_eval_split(items, seed=2)[1]}
        assert split_a != split_b
