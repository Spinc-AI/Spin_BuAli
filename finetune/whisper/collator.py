"""Batching audio+text pairs for a Whisper encoder-decoder.

`DataCollatorSpeechSeq2SeqWithPadding` below is a direct port of the PEFT
project's own official example
(`peft/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb`,
huggingface/peft) -- the single most-reproduced LoRA-fine-tune-Whisper recipe
found while researching this pipeline, maintained by PEFT's own authors as
their example rather than a third party's. Nothing about the padding or
label-masking logic is changed; only the docstrings and a `pad_to_multiple_of`
knob are added.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Encoder input (`input_features`, log-mel spectrograms -- already a
    fixed width, so a plain feature-extractor pad is enough) and decoder
    target (`labels`, variable-length token ids) are padded independently,
    because they come from two different padders with two different ideas
    of what "padding" means: the feature extractor pads spectrogram frames,
    the tokenizer pads token ids, and pretending one collator can do both
    with a single `pad()` call is the mistake this class exists to avoid.

    Padding positions in `labels` are replaced with -100 (not the
    tokenizer's own pad id) so the loss ignores them -- the same masking
    idea as `finetune/sequence.py`'s prompt masking, applied here to padding
    instead of to a prompt, because Whisper's decoder never sees a text
    prompt at all: the encoder consumes the audio, and every decoder token
    is real target text.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        import torch

        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # The decoder-start token is prepended by generate() at inference
        # time; if it also survived here as the first label, the model would
        # be asked to predict a token that is never actually missing from
        # its input, which teaches nothing. Strip it if present.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_example(example, processor, target_sample_rate=16000):
    """One dataset row -> `{"input_features": ..., "labels": ...}`.

    Meant for `datasets.Dataset.map`, mirroring the HF Whisper fine-tuning
    blog's own `prepare_dataset` function. Kept separate from the collator
    (which only ever sees already-prepared rows) so the expensive
    feature-extraction step runs once per example via `.map`'s own caching,
    not once per epoch inside the collator.
    """
    audio = example["audio"]
    assert audio["sampling_rate"] == target_sample_rate, (
        f"{example.get('asset_id', '?')}: audio arrived at {audio['sampling_rate']} Hz, "
        f"expected {target_sample_rate} -- data.to_hf_dataset should have resampled it")
    features = processor.feature_extractor(
        audio["array"], sampling_rate=target_sample_rate).input_features[0]
    labels = processor.tokenizer(example["text"]).input_ids
    return {"input_features": features, "labels": labels}
