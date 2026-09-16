"""Batching audio+text pairs for Voxtral, in its own transcription-request
shape rather than the chat/JSON shape `core_llm/model.py`'s `VoxtralModel`
uses at inference time.

Why transcription mode, not chat mode: `processor.apply_transcription_request`
is the prompt Voxtral was itself trained for -- audio in, transcript out, no
JSON wrapper, no report structure. That is also the shape the one published,
working Voxtral ASR fine-tune found while researching this pipeline uses
(`Deep-unlearning/Finetune-Voxtral-ASR`, MIT-adjacent reference code, see
`finetune/README.md`'s Research section for the full citation trail). LoRA
weights trained this way still change the underlying attention/MLP layers
`core_llm/model.py`'s chat-mode `VoxtralModel` calls too -- a LoRA adapter is
not tied to one prompt template -- but the model was only ever *rewarded*
here for verbatim transcription, not for restructuring that transcription
into the benchmark's JSON report. Whether that reward transfers to chat-mode
report generation is not proven by this file; it is an empirical question for
`benchmark/runner.py` to answer once a checkpoint exists (see the README's
"After training" section for how to point the benchmark at one).
"""
from sequence import build_batch


class VoxtralTranscriptionCollator:
    """One batch: `processor.apply_transcription_request`'s prompt for each
    clip, this dataset's `text` column as the target to train on.

    `model_id` is passed to `apply_transcription_request` because Voxtral's
    prompt format is itself checkpoint-versioned (see the tokenizer-version
    validation `core_llm/model.py`'s `VoxtralModel` docstring hit and worked
    around for chat mode) -- the processor needs to know which checkpoint's
    convention to build the prompt in.
    """

    def __init__(self, processor, model_id, language="en", max_length=448,
                target_max_tokens=256):
        self.processor = processor
        self.model_id = model_id
        self.language = language
        self.max_length = max_length
        self.target_max_tokens = target_max_tokens
        self.tokenizer = processor.tokenizer

    def __call__(self, features):
        import torch

        audios = [feature["audio"]["array"] for feature in features]
        texts = [feature["text"] for feature in features]

        prompt = self.processor.apply_transcription_request(
            language=self.language, model_id=self.model_id, audio=audios,
            format=["WAV"] * len(audios), return_tensors="pt")
        # Everything the model needs besides input_ids/attention_mask -- the
        # audio features themselves -- passed through to the model call
        # untouched; build_batch below only ever reconstructs the token half.
        passthrough = {key: value for key, value in prompt.items()
                       if key not in ("input_ids", "attention_mask")}

        target_ids_batch = self.tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=True,
            max_length=self.target_max_tokens)["input_ids"]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch = build_batch(
            prompt["input_ids"].tolist(), prompt["attention_mask"].tolist(),
            target_ids_batch, eos_id=self.tokenizer.eos_token_id, pad_id=pad_id,
            max_length=self.max_length)

        collated = {key: torch.tensor(value, dtype=torch.long)
                   for key, value in batch.items()}
        collated.update(passthrough)
        return collated
