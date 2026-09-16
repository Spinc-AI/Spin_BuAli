#!/usr/bin/env python3
"""LoRA fine-tune of Whisper on this project's own dictations.

    python -m whisper.train_lora --labels-csv /path/to/labels.csv --dry-run
    python -m whisper.train_lora --labels-csv /path/to/labels.csv

The most-reproduced fine-tuning recipe in this whole pipeline -- see
`finetune/README.md`'s Research section. Two independent, published
precedents back it:

  * PEFT's own official example
    (`examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb`,
    huggingface/peft) is where `--lora-r 32 --lora-alpha 64
    --target-modules q_proj v_proj` below comes from -- maintained by the
    PEFT authors as their reference LoRA-on-Whisper recipe, int8-loaded, run
    on a single T4.
  * MediBeng-Whisper-Tiny (`pr0mila-gh0sh/MediBeng-Whisper-Tiny`, medRxiv
    preprint) is the closest published precedent to this project's own
    domain: a Whisper checkpoint fine-tuned specifically on code-switched
    (Bengali-English) clinical dictation, WER falling from 107.7 to ~29.5
    within under one epoch. There is no public Persian-English clinical
    equivalent (checked; see the README) -- this is evidence the *technique*
    (domain-adapt Whisper on code-switched medical speech) works, not that
    this project's own numbers will match it.

Default checkpoint is `nezamisafa/whisper-persian-v4` -- the checkpoint
`stt/app/config.py`'s MODEL_REGISTRY already uses in production under the key
"whisper" (see `benchmark/runner.py`'s `TOP3_STT`). This script domain-adapts
that same checkpoint further, rather than starting from a generic multilingual
Whisper that has never seen Persian at all.
"""
import argparse
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

import data as ft_data  # noqa: E402
import pipeline_settings as settings  # noqa: E402
from whisper.collator import DataCollatorSpeechSeq2SeqWithPadding, prepare_example  # noqa: E402

DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=pathlib.Path, required=True)
    parser.add_argument("--model-checkpoint", default="nezamisafa/whisper-persian-v4")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("./whisper-buali-lora"))
    parser.add_argument("--language", default="english",
                        help="Whisper's own language name, used to set the forced "
                             "decoder prompt -- \"english\" since this dataset is "
                             "English-dominant with Persian words mixed in.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    split = parser.add_argument_group("train/eval split")
    split.add_argument("--eval-fraction", type=float, default=settings.EVAL_FRACTION)
    split.add_argument("--min-eval-items", type=int, default=settings.MIN_EVAL_ITEMS)
    split.add_argument("--split-seed", type=int, default=settings.SPLIT_SEED)

    lora = parser.add_argument_group("LoRA (PEFT's own Whisper example recipe by default)")
    lora.add_argument("--lora-r", type=int, default=32)
    lora.add_argument("--lora-alpha", type=int, default=64)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)

    train = parser.add_argument_group("training")
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--per-device-batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--warmup-steps", type=int, default=50)
    train.add_argument("--generation-max-length", type=int, default=225)
    train.add_argument("--logging-steps", type=int, default=5)
    train.add_argument("--eval-steps", type=int, default=20)
    train.add_argument("--save-steps", type=int, default=20)

    parser.add_argument("--dry-run", action="store_true",
                        help="Build the model, the dataset, one batch, and run a single "
                             "forward+backward step, then exit.")
    return parser.parse_args(argv)


def build_model(model_checkpoint, target_modules, r, alpha, dropout):
    """int8-loaded base checkpoint wrapped in a LoRA adapter -- exactly
    PEFT's own example: `load_in_8bit=True`, then
    `prepare_model_for_kbit_training` before `get_peft_model`, so training
    only ever touches the LoRA weights, everything else stays 8-bit and
    frozen."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        model_checkpoint, load_in_8bit=True, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules,
                             lora_dropout=dropout, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_split(args, processor):
    items = ft_data.load_items(args.labels_csv)
    train_items, eval_items = ft_data.train_eval_split(
        items, eval_fraction=args.eval_fraction, min_eval=args.min_eval_items,
        seed=args.split_seed)
    print(f"{len(items)} labelled item(s): {len(train_items)} train, "
         f"{len(eval_items)} eval")

    def prepare(dataset):
        prepared = dataset.map(
            lambda example: prepare_example(example, processor),
            remove_columns=dataset.column_names)
        return prepared

    train_dataset = prepare(ft_data.to_hf_dataset(train_items))
    eval_dataset = prepare(ft_data.to_hf_dataset(eval_items)) if eval_items else None
    return train_dataset, eval_dataset


def make_compute_metrics(processor):
    """WER on decoded predictions vs. decoded labels -- PEFT's own example's
    `compute_metrics`, unchanged: swap -100 back to the pad id before
    decoding (it is not a real token, `batch_decode` would choke on it),
    then let `jiwer` do the comparison jiwer does everywhere else in this
    repo's own scoring code.
    """
    import jiwer
    import numpy as np

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids != -100, pred.label_ids,
                             processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * jiwer.wer(label_str, pred_str)}

    return compute_metrics


def main(argv=None):
    args = parse_args(argv)

    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(
        args.model_checkpoint, language=args.language, task=args.task)
    train_dataset, eval_dataset = load_split(args, processor)

    model = build_model(args.model_checkpoint, args.target_modules, args.lora_r,
                        args.lora_alpha, args.lora_dropout)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    if args.dry_run:
        import torch

        sample_count = min(2, len(train_dataset))
        batch = collator([train_dataset[i] for i in range(sample_count)])
        batch = {key: value.to(model.device) for key, value in batch.items()}
        model.train()
        outputs = model(**batch)
        outputs.loss.backward()
        print(f"dry run OK -- loss={outputs.loss.item():.4f}, "
             f"input_features shape={tuple(batch['input_features'].shape)}")
        return

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        eval_strategy="epoch" if eval_dataset else "no",
        predict_with_generate=bool(eval_dataset),
        fp16=True,
        generation_max_length=args.generation_max_length,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=collator,
        compute_metrics=make_compute_metrics(processor) if eval_dataset else None,
        tokenizer=processor.feature_extractor)
    trainer.train()

    print(f"saving to {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(str(args.output_dir))

    if eval_dataset:
        print(trainer.evaluate())


if __name__ == "__main__":
    main()
