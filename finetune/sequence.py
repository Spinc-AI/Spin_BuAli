"""Building one supervised training sequence out of a prompt and a target.

This is the part of a causal-LM speech collator that is actually worth a unit
test, and the only part of one that can be tested without a GPU, a real
tokenizer, or `torch` installed: everything here is opaque integer token IDs
in plain Python lists. `finetune/gemma/collator.py` is a thin wrapper that
calls the real processor for the prompt half and the real tokenizer for the
target half, then hands both to `build_batch` below -- so a mistake in the
masking or padding arithmetic (the actual failure mode that makes a model
train to reproduce its own prompt, or silently drop the wrong tokens from the
loss) shows up here, not three hours into a Kaggle session.

The shape is the LLaMA-style prompt masking every reference implementation
found while researching this pipeline uses -- both
`Deep-unlearning/Finetune-Voxtral-ASR`'s `VoxtralDataCollator` and the PEFT
project's own `DataCollatorSpeechSeq2SeqWithPadding` example mask the prompt
with -100 and train only on the target (see `finetune/README.md`'s Research
section for why those two, not Gemma-specific ones, were the templates this
module's shape was drawn from -- no Gemma-specific reference implementation
with this level of detail was found).
"""


def build_labelled_sequence(prompt_ids, target_ids, eos_id):
    """One (input_ids, labels) pair: prompt + target + eos.

    `labels` masks the prompt with -100 so the loss is computed only on the
    target and its trailing eos -- the model is being taught to *produce*
    the target given the prompt, not to reproduce the prompt back. Getting
    this backwards (or forgetting it) is a real, easy mistake: the model
    still trains, the loss still goes down, and what it learns is to recite
    prompts instead of transcribing audio.
    """
    input_ids = list(prompt_ids) + list(target_ids) + [eos_id]
    labels = [-100] * len(prompt_ids) + list(target_ids) + [eos_id]
    return input_ids, labels


def pad_rows(rows, fill_value, width=None):
    """Right-pad a batch of variable-length rows to one common width."""
    width = width if width is not None else max(len(row) for row in rows)
    return [list(row) + [fill_value] * (width - len(row)) for row in rows]


def build_batch(prompt_ids_batch, prompt_attention_batch, target_ids_batch,
                eos_id, pad_id, label_pad_id=-100, max_length=None):
    """A full, padded `{input_ids, attention_mask, labels}` batch.

    `max_length`, when given, truncates the TARGET half only. The prompt --
    the audio placeholder tokens, for the models this folder trains -- is
    never truncated: cutting it would silently drop audio the model never
    gets to hear, which is a much worse failure than a shortened reference
    transcript. A prompt that alone reaches `max_length` raises instead of
    truncating anything, since there would be no room left for even one
    target token plus eos.
    """
    if len(prompt_ids_batch) != len(target_ids_batch):
        raise ValueError(
            f"{len(prompt_ids_batch)} prompt(s) but {len(target_ids_batch)} "
            "target(s) -- every item needs exactly one of each")

    input_rows, attention_rows, label_rows = [], [], []
    for prompt_ids, prompt_attention, target_ids in zip(
            prompt_ids_batch, prompt_attention_batch, target_ids_batch):
        if max_length is not None:
            if len(prompt_ids) >= max_length:
                raise ValueError(
                    f"a prompt alone is {len(prompt_ids)} token(s), at or "
                    f"past max_length={max_length} -- truncating it would "
                    "silently drop audio, not text, so there is nothing "
                    "safe to cut")
            target_ids = target_ids[:max_length - len(prompt_ids) - 1]  # room for eos

        input_ids, labels = build_labelled_sequence(prompt_ids, target_ids, eos_id)
        attention_mask = list(prompt_attention) + [1] * (len(target_ids) + 1)

        input_rows.append(input_ids)
        attention_rows.append(attention_mask)
        label_rows.append(labels)

    width = max(len(row) for row in input_rows)
    return {
        "input_ids": pad_rows(input_rows, pad_id, width),
        "attention_mask": pad_rows(attention_rows, 0, width),
        "labels": pad_rows(label_rows, label_pad_id, width),
    }
