"""Batching audio+text pairs for Gemma 4's encoder-free multimodal chat
format.

This is the SAME message shape `core_llm/model.py`'s `GemmaAudioModel` uses
at inference -- `processor.apply_chat_template` over a system turn, a user
turn carrying the audio, and (here, only during training) an assistant turn
holding the target JSON -- not a separate ASR-only prompt format. That is the
whole point of choosing Gemma over Whisper/Voxtral for this: a LoRA adapter
trained against this exact prompt shape has no "does this transfer to
production?" gap, because production IS this prompt shape.

The system prompt itself is not rewritten here. `benchmark/pipeline.py`'s own
`audio_prompt()` builds it -- the same `controller/prompts.py` template the
real benchmark and the real controller both use -- imported rather than
copied, for the reason `benchmark/pipeline.py`'s own module docstring gives:
"A benchmark that phrased the instruction its own way would be ranking
models on a system nobody ships." Training against a copy would have exactly
that problem.
"""
import json
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK_DIR = FINETUNE_DIR.parent / "benchmark"
for _dir in (FINETUNE_DIR, BENCHMARK_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))

import pipeline as bm_pipeline  # noqa: E402
from sequence import build_batch  # noqa: E402


def build_target(reference_text):
    """The assistant turn's target JSON, from one dataset row's reference
    report.

    `raw_transcript` == `corrected_transcript` == `final_text` == the report
    text itself: this project's own benchmark already treats the
    radiologist's signed report as the ground-truth transcript for WER
    purposes (see `benchmark/scoring.py`) -- there is no separately-labelled
    verbatim dictation to train against, and inventing one (a fake
    "raw_transcript" with disfluencies the real dataset never recorded)
    would be a bigger, less honest assumption than reusing the one label
    that actually exists. `discrepancies_found` is empty and `notes` is
    null, matching what a clean multimodal reply looks like when nothing
    needed correcting -- see `controller/prompts.py`'s `TRANSCRIBE_FROM_AUDIO`
    for the contract this is filling in.
    """
    return {
        "raw_transcript": reference_text,
        "corrected_transcript": reference_text,
        "final_text": reference_text,
        "discrepancies_found": [],
        "notes": None,
    }


class GemmaChatCollator:
    """One batch: each example's own system+user prompt (audio, no
    cross-check transcript -- the multimodal shape `pipeline.audio_prompt`
    returns when `transcripts={}`) built via
    `processor.apply_chat_template`, and the target JSON (`build_target`)
    tokenized and appended as the assistant turn, masked out of the loss by
    `sequence.build_batch` the same way every other collator in this repo
    masks its prompt.

    Built one example at a time, not as a single batched
    `apply_chat_template` call, and **`--per-device-batch-size 1` is the
    only configuration this class has actually been reasoned through
    carefully**: `core_llm/model.py`'s `GemmaAudioModel` itself never
    batches more than one conversation into one `apply_chat_template` call
    either, at inference or anywhere else in this repo, so batch size 1
    here is provably the same shape as code already running successfully in
    production. A batch size greater than 1 concatenates each example's own
    audio-feature tensors along dim 0 below, which is a reasonable
    extrapolation, not something verified against a documented multi-clip
    Gemma 4 training example -- no such example was found while researching
    this pipeline (see `finetune/README.md`). Use a batch size above 1 with
    that in mind; `--gradient-accumulation-steps` is the documented,
    lower-risk way to reach a larger effective batch instead.
    """

    def __init__(self, processor, structure_guide=None, target_max_tokens=768,
                max_length=1536):
        self.processor = processor
        self.structure_guide = structure_guide
        self.target_max_tokens = target_max_tokens
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def prompt_inputs(self, audio_path):
        """One example's prompt, tokenized -- the exact call
        `GemmaAudioModel.chat` makes, minus generation."""
        system_prompt, _ = bm_pipeline.audio_prompt(
            transcripts={}, structure_guide=self.structure_guide)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": ""},
                # str(), not the Path itself -- see GemmaAudioModel's own
                # comment: the processor accepts a numpy array or a string
                # (URL, local path, base64), not a pathlib.Path.
                {"type": "audio", "url": str(audio_path)},
            ]},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")

    def __call__(self, features):
        import torch

        prompt_ids_batch, prompt_attn_batch, target_ids_batch = [], [], []
        audio_kwargs_per_example = []
        for feature in features:
            prompt_inputs = self.prompt_inputs(feature["audio_path"])
            prompt_ids_batch.append(prompt_inputs["input_ids"][0].tolist())
            prompt_attn_batch.append(prompt_inputs["attention_mask"][0].tolist())
            audio_kwargs_per_example.append({
                key: value for key, value in prompt_inputs.items()
                if key not in ("input_ids", "attention_mask")})

            target_json = json.dumps(build_target(feature["text"]), ensure_ascii=False)
            target_ids = self.tokenizer(
                target_json, add_special_tokens=False, truncation=True,
                max_length=self.target_max_tokens)["input_ids"]
            target_ids_batch.append(target_ids)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch = build_batch(
            prompt_ids_batch, prompt_attn_batch, target_ids_batch,
            eos_id=self.tokenizer.eos_token_id, pad_id=pad_id,
            max_length=self.max_length)

        collated = {key: torch.tensor(value, dtype=torch.long)
                   for key, value in batch.items()}
        # Audio feature tensors are per-example, not padded like token ids --
        # see this class's docstring for the batch-size-1 caveat this
        # concatenation carries above 1.
        for key in audio_kwargs_per_example[0]:
            collated[key] = torch.cat(
                [kwargs[key] for kwargs in audio_kwargs_per_example], dim=0)
        return collated
