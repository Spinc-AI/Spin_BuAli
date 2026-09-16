"""`data.to_hf_dataset` -- separated from `test_data.py` because it needs the
`datasets` package, which is not installed in every environment this repo's
test suite runs in (it is not installed in the one this pipeline was built
in, for instance). `pytest.importorskip` makes that an honest skip here and
a real, executed test wherever `datasets` is actually present -- Kaggle
included.
"""
import pytest

datasets = pytest.importorskip("datasets")

import data  # noqa: E402


class TestToHFDataset:
    def test_has_the_three_columns_the_gemma_collator_expects(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        hf_dataset = data.to_hf_dataset(items)
        assert set(hf_dataset.column_names) == {"asset_id", "audio_path", "text"}

    def test_row_count_matches_the_item_count(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        hf_dataset = data.to_hf_dataset(items)
        assert len(hf_dataset) == len(items)

    def test_audio_path_is_a_plain_string_not_a_decoded_column(self, tiny_dataset):
        """gemma/collator.py hands this straight to
        processor.apply_chat_template as {"type": "audio", "url": ...} --
        Gemma's own processor loads and resamples the file itself, the
        same way core_llm/model.py's GemmaAudioModel does at inference."""
        items = data.load_items(tiny_dataset)
        hf_dataset = data.to_hf_dataset(items)
        row = hf_dataset[0]
        assert isinstance(row["audio_path"], str)
        assert row["audio_path"].endswith(".wav")

    def test_text_matches_the_items_reference(self, tiny_dataset):
        items = data.load_items(tiny_dataset)
        hf_dataset = data.to_hf_dataset(items)
        by_asset_id = {row["asset_id"]: row["text"] for row in hf_dataset}
        for item in items:
            assert by_asset_id[item.asset_id] == item.reference
