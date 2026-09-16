#!/usr/bin/env python3
"""Baseline vs fine-tuned WER on the held-out eval split, generating exactly
the way `core_llm/model.py`'s `GemmaAudioModel` does at inference (chat
template, audio in the user turn, JSON reply parsed back out) -- not a
separate evaluation-only prompt.

    python -m gemma.evaluate --labels-csv /path/to/labels.csv \\
        --model-checkpoint google/gemma-4-E4B-it \\
        --adapter-dir ./gemma-buali-lora

Runs the exact split `gemma/train_lora.py` used (same `--eval-fraction`,
`--min-eval-items`, `--split-seed` defaults) against two checkpoints in turn
-- the base model, then the base model with the trained LoRA adapter applied
-- and prints a before/after WER on `final_text` against the reference
report. `--adapter-dir` omitted evaluates the base checkpoint alone, which
is also how to sanity-check the eval split before spending any GPU time on
training.

WER on `final_text` here, not the benchmark's full medical-term/negation/
laterality suite -- see `finetune/README.md`'s "After training" section for
pointing `benchmark/runner.py` at a trained checkpoint for that fuller
picture.
"""
import argparse
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK_DIR = FINETUNE_DIR.parent / "benchmark"
for _dir in (FINETUNE_DIR, BENCHMARK_DIR):
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))

import bridge  # noqa: E402
import data as ft_data  # noqa: E402
import pipeline_settings as settings  # noqa: E402
from gemma.collator import GemmaChatCollator  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=pathlib.Path, required=True)
    parser.add_argument("--model-checkpoint", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter-dir", type=pathlib.Path, default=None,
                        help="A directory saved by gemma/train_lora.py. Omit to "
                             "evaluate the base checkpoint alone.")
    parser.add_argument("--structure-guide", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=settings.EVAL_FRACTION)
    parser.add_argument("--min-eval-items", type=int, default=settings.MIN_EVAL_ITEMS)
    parser.add_argument("--split-seed", type=int, default=settings.SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    return parser.parse_args(argv)


def load_model(model_checkpoint, adapter_dir):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_checkpoint, padding_side="left")
    model = AutoModelForMultimodalLM.from_pretrained(
        model_checkpoint, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    model.eval()
    return processor, model


def generate_report(processor, model, collator, audio_path, max_new_tokens):
    """One reply, parsed the same tolerant way `benchmark/pipeline.py`'s
    `build_report` parses a live model's reply -- a malformed JSON reply is
    a data point about this checkpoint, not a crash."""
    import torch

    prompt_inputs = collator.prompt_inputs(audio_path)
    prompt_inputs = {key: value.to(model.device) for key, value in prompt_inputs.items()}
    input_len = prompt_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(**prompt_inputs, max_new_tokens=max_new_tokens)
    reply = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    try:
        return bridge.extract_json(reply).get("final_text") or ""
    except Exception as error:  # noqa: BLE001 - a bad reply is a result, not a crash
        return f"<unparseable reply: {type(error).__name__}: {reply[:200]!r}>"


def word_error_rate(hypotheses, references):
    import jiwer

    return jiwer.wer(references, hypotheses)


def main(argv=None):
    args = parse_args(argv)
    items = ft_data.load_items(args.labels_csv)
    _, eval_items = ft_data.train_eval_split(
        items, eval_fraction=args.eval_fraction, min_eval=args.min_eval_items,
        seed=args.split_seed)

    label = f"{args.model_checkpoint}" + (f" + {args.adapter_dir}" if args.adapter_dir else "")
    print(f"evaluating: {label}")
    print(f"{len(eval_items)} eval item(s)")

    structure_guide = None
    if args.structure_guide:
        import report_structure

        structure_guide = report_structure.GUIDE

    processor, model = load_model(args.model_checkpoint, args.adapter_dir)
    collator = GemmaChatCollator(processor, structure_guide=structure_guide)

    hypotheses, references = [], []
    for item in eval_items:
        hypothesis = generate_report(processor, model, collator, item.audio,
                                     args.max_new_tokens)
        hypotheses.append(hypothesis)
        references.append(item.reference)
        print(f"  {item.asset_id:14} ref : {item.reference[:80]!r}")
        print(f"  {'':14} hyp : {hypothesis[:80]!r}")

    wer = word_error_rate(hypotheses, references)
    print(f"\nWER (final_text vs reference report): {wer:.4f}")
    return wer


if __name__ == "__main__":
    main()
