#!/usr/bin/env python3
"""Baseline vs fine-tuned WER on the held-out eval split.

    python -m voxtral.evaluate --labels-csv /path/to/labels.csv \\
        --model-checkpoint mistralai/Voxtral-Mini-3B-2507 \\
        --adapter-dir ./voxtral-buali-lora

Runs the exact split `voxtral/train_lora.py` used (same `--eval-fraction`,
`--min-eval-items`, `--split-seed` defaults) against two checkpoints in turn
-- the base model, then the base model with the trained LoRA adapter applied
-- and prints a before/after WER, mirroring the "baseline evaluation before
fine-tuning ... final evaluation" workflow Trelis Research's Voxtral recipe
documents. `--adapter-dir` omitted evaluates the base checkpoint alone,
which is also how to sanity-check the eval split itself before spending any
GPU time on training.

WER here, not the benchmark's full medical-term/negation/laterality suite --
see `finetune/README.md`'s "After training" section for pointing
`benchmark/runner.py` at a trained checkpoint for that fuller picture. This
script exists to answer one narrower question fast: did fine-tuning move
transcription accuracy at all.
"""
import argparse
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

import data as ft_data  # noqa: E402
import pipeline_settings as settings  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=pathlib.Path, required=True)
    parser.add_argument("--model-checkpoint", default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--adapter-dir", type=pathlib.Path, default=None,
                        help="A directory saved by voxtral/train_lora.py. Omit to "
                             "evaluate the base checkpoint alone.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--eval-fraction", type=float, default=settings.EVAL_FRACTION)
    parser.add_argument("--min-eval-items", type=int, default=settings.MIN_EVAL_ITEMS)
    parser.add_argument("--split-seed", type=int, default=settings.SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def load_model(model_checkpoint, adapter_dir):
    import torch
    from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

    processor = VoxtralProcessor.from_pretrained(model_checkpoint)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint, dtype=torch.bfloat16, device_map="auto")
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    model.eval()
    return processor, model


def transcribe(processor, model, item, language, max_new_tokens):
    import torch

    audio, _ = ft_data.bm_dataset.load_audio(item.audio)
    inputs = processor.apply_transcription_request(
        language=language, audio=[audio], format=["WAV"], return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


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

    processor, model = load_model(args.model_checkpoint, args.adapter_dir)
    hypotheses, references = [], []
    for item in eval_items:
        hypothesis = transcribe(processor, model, item, args.language, args.max_new_tokens)
        hypotheses.append(hypothesis)
        references.append(item.reference)
        print(f"  {item.asset_id:14} ref : {item.reference[:80]!r}")
        print(f"  {'':14} hyp : {hypothesis[:80]!r}")

    wer = word_error_rate(hypotheses, references)
    print(f"\nWER: {wer:.4f}")
    return wer


if __name__ == "__main__":
    main()
