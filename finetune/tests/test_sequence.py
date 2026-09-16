"""`sequence.build_batch`'s masking and padding arithmetic -- the one part
of a speech-LM collator that can be tested without a GPU, a real tokenizer,
or `torch`, and the part most likely to be wrong in a way that still trains
"successfully" (a loss that goes down) while teaching the model nothing
useful. See `finetune/sequence.py`'s module docstring for why this file
exists separately from the collators that call it.
"""
import pytest

import sequence


class TestBuildLabelledSequence:
    def test_input_ids_is_prompt_then_target_then_eos(self):
        input_ids, _ = sequence.build_labelled_sequence([1, 2, 3], [4, 5], eos_id=99)
        assert input_ids == [1, 2, 3, 4, 5, 99]

    def test_labels_masks_the_prompt_and_keeps_target_and_eos(self):
        _, labels = sequence.build_labelled_sequence([1, 2, 3], [4, 5], eos_id=99)
        assert labels == [-100, -100, -100, 4, 5, 99]

    def test_an_empty_prompt_masks_nothing(self):
        input_ids, labels = sequence.build_labelled_sequence([], [4, 5], eos_id=99)
        assert input_ids == [4, 5, 99]
        assert labels == [4, 5, 99]

    def test_an_empty_target_still_gets_its_eos_scored(self):
        """An empty reference transcript is a real, if unusual, input (a
        clip with no ground truth text at all) -- the model should still be
        taught to emit eos rather than the prompt continuing to mask
        everything and the sequence teaching nothing."""
        input_ids, labels = sequence.build_labelled_sequence([1, 2], [], eos_id=99)
        assert input_ids == [1, 2, 99]
        assert labels == [-100, -100, 99]


class TestPadRows:
    def test_pads_every_row_to_the_widest(self):
        assert sequence.pad_rows([[1, 2], [1, 2, 3, 4]], fill_value=0) == \
            [[1, 2, 0, 0], [1, 2, 3, 4]]

    def test_an_explicit_width_wider_than_any_row_is_honoured(self):
        assert sequence.pad_rows([[1]], fill_value=0, width=3) == [[1, 0, 0]]

    def test_a_row_already_at_width_is_untouched(self):
        assert sequence.pad_rows([[1, 2, 3]], fill_value=0, width=3) == [[1, 2, 3]]


class TestBuildBatch:
    def test_a_two_item_batch_pads_to_the_longer_sequence(self):
        batch = sequence.build_batch(
            prompt_ids_batch=[[1, 1], [1, 1, 1]],
            prompt_attention_batch=[[1, 1], [1, 1, 1]],
            target_ids_batch=[[9], [9, 9]],
            eos_id=0, pad_id=-1)
        # item 0: [1, 1, 9, 0]        (len 4)
        # item 1: [1, 1, 1, 9, 9, 0]  (len 6) -- the longer one, sets the width
        assert batch["input_ids"] == [[1, 1, 9, 0, -1, -1], [1, 1, 1, 9, 9, 0]]
        assert batch["labels"] == [[-100, -100, 9, 0, -100, -100], [-100, -100, -100, 9, 9, 0]]
        assert batch["attention_mask"] == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]

    def test_every_row_has_matching_lengths_across_the_three_keys(self):
        batch = sequence.build_batch(
            prompt_ids_batch=[[1], [1, 2, 3]],
            prompt_attention_batch=[[1], [1, 1, 1]],
            target_ids_batch=[[9, 9, 9], [9]],
            eos_id=0, pad_id=-1)
        for row_ids, row_labels, row_attn in zip(
                batch["input_ids"], batch["labels"], batch["attention_mask"]):
            assert len(row_ids) == len(row_labels) == len(row_attn)

    def test_mismatched_prompt_and_target_counts_raises(self):
        with pytest.raises(ValueError, match="prompt.*target"):
            sequence.build_batch(
                prompt_ids_batch=[[1], [1]], prompt_attention_batch=[[1], [1]],
                target_ids_batch=[[9]], eos_id=0, pad_id=-1)

    def test_max_length_truncates_the_target_not_the_prompt(self):
        batch = sequence.build_batch(
            prompt_ids_batch=[[1, 2, 3]], prompt_attention_batch=[[1, 1, 1]],
            target_ids_batch=[[9, 9, 9, 9, 9]], eos_id=0, pad_id=-1, max_length=6)
        # room for target: 6 - 3 (prompt) - 1 (eos) = 2 target tokens kept
        assert batch["input_ids"] == [[1, 2, 3, 9, 9, 0]]
        assert len(batch["input_ids"][0]) == 6

    def test_a_prompt_alone_at_or_past_max_length_raises_rather_than_dropping_audio(self):
        with pytest.raises(ValueError, match="prompt alone"):
            sequence.build_batch(
                prompt_ids_batch=[[1, 2, 3]], prompt_attention_batch=[[1, 1, 1]],
                target_ids_batch=[[9]], eos_id=0, pad_id=-1, max_length=3)

    def test_a_single_item_batch_needs_no_padding(self):
        batch = sequence.build_batch(
            prompt_ids_batch=[[1, 1]], prompt_attention_batch=[[1, 1]],
            target_ids_batch=[[9]], eos_id=0, pad_id=-1)
        assert batch["input_ids"] == [[1, 1, 9, 0]]
        assert -1 not in batch["input_ids"][0]
