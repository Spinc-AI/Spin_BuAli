#!/usr/bin/env python3
"""LoRA fine-tune of Voxtral-Mini-3B on this project's own dictations.

    python -m voxtral.train_lora --labels-csv /path/to/labels.csv --dry-run
    python -m voxtral.train_lora --labels-csv /path/to/labels.csv

Recipe sourced from two independently published, working implementations --
see `finetune/README.md`'s Research section for the full citation trail and
why each one is trusted:

  * `Deep-unlearning/Finetune-Voxtral-ASR` -- the collator shape this file's
    `voxtral/collator.py` and `sequence.py` are built from: transcription-
    request prompting, prompt tokens masked out of the loss.
  * Trelis Research's "Train Voxtral Transcription (ASR) Models" -- the LoRA
    hyperparameters below (r=32, alpha=32, rslora, attention + MLP +
    projector targets, frozen audio tower, lr=5e-5), reported as the recipe
    that worked for Voxtral Mini specifically, not a generic LoRA default.

Neither is reproduced blind: `sequence.py`'s masking/padding arithmetic is
pinned by `finetune/tests/test_sequence.py`, and `--dry-run` below runs the
whole pipeline -- load data, build one batch, one forward+backward step --
end to end before any real GPU-hours are spent, so a wiring mistake shows up
in seconds, not partway through epoch one.
"""
import argparse
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

import data as ft_data  # noqa: E402
import pipeline_settings as settings  # noqa: E402
from voxtral.collator import VoxtralTranscriptionCollator  # noqa: E402

# Trelis' reported recipe for Voxtral Mini: attention (QKVO), the MLP's three
# linear layers, and the multi-modal projector that bridges the frozen audio
# tower to the language model -- the projector is where "what the audio
# encoder heard" turns into token embeddings, so it is the one audio-adjacent
# piece worth adapting even with the tower itself frozen.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "multi_modal_projector.linear_1", "multi_modal_projector.linear_2",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=pathlib.Path, required=True,
                        help="labels.csv or a JSON manifest -- see benchmark/dataset.py")
    parser.add_argument("--model-checkpoint", default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("./voxtral-buali-lora"))
    parser.add_argument("--language", default="en",
                        help="Passed to apply_transcription_request; the audio here is "
                             "English-dominant with Persian words mixed in, not "
                             "Persian-dominant -- see the project's own notes on that.")

    split = parser.add_argument_group("train/eval split")
    split.add_argument("--eval-fraction", type=float, default=settings.EVAL_FRACTION)
    split.add_argument("--min-eval-items", type=int, default=settings.MIN_EVAL_ITEMS)
    split.add_argument("--split-seed", type=int, default=settings.SPLIT_SEED)

    lora = parser.add_argument_group("LoRA (Trelis' Voxtral Mini recipe by default)")
    lora.add_argument("--lora-r", type=int, default=32)
    lora.add_argument("--lora-alpha", type=int, default=32)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    lora.add_argument("--no-rslora", dest="rslora", action="store_false",
                      help="Disable rank-stabilized LoRA scaling (on by default, "
                           "per Trelis' recipe).")

    train = parser.add_argument_group("training")
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--per-device-batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--max-length", type=int, default=448,
                       help="Total sequence length (prompt + target + eos) a batch is "
                            "padded to; the target is truncated to fit, never the "
                            "prompt -- see sequence.build_batch.")
    train.add_argument("--target-max-tokens", type=int, default=256)
    train.add_argument("--logging-steps", type=int, default=5)
    train.add_argument("--eval-steps", type=int, default=20)
    train.add_argument("--save-steps", type=int, default=20)

    parser.add_argument("--dry-run", action="store_true",
                        help="Build the model, the dataset, one batch, and run a single "
                             "forward+backward step, then exit -- proves the pipeline is "
                             "wired correctly before spending real GPU-hours on it.")
    return parser.parse_args(argv)


def build_model(model_checkpoint, target_modules, r, alpha, dropout, use_rslora):
    """The base checkpoint, audio tower frozen, wrapped in a LoRA adapter.

    Freezing `audio_tower` (Trelis' recipe, matched by
    `Deep-unlearning/Finetune-Voxtral-ASR`'s script) keeps what the encoder
    already hears fixed and only teaches the language model side how to use
    it differently -- the audio encoder is a general-purpose Whisper
    encoder, not something nine (or nine thousand) domain clips should be
    allowed to specialize away from everything else it already does well.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import VoxtralForConditionalGeneration

    model = VoxtralForConditionalGeneration.from_pretrained(
        model_checkpoint, dtype=torch.bfloat16, device_map="auto")
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=target_modules, use_rslora=use_rslora,
        task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_split(args):
    items = ft_data.load_items(args.labels_csv)
    train_items, eval_items = ft_data.train_eval_split(
        items, eval_fraction=args.eval_fraction, min_eval=args.min_eval_items,
        seed=args.split_seed)
    print(f"{len(items)} labelled item(s): {len(train_items)} train, "
         f"{len(eval_items)} eval")
    train_dataset = ft_data.to_hf_dataset(train_items)
    eval_dataset = ft_data.to_hf_dataset(eval_items) if eval_items else None
    return train_dataset, eval_dataset


def main(argv=None):
    args = parse_args(argv)
    train_dataset, eval_dataset = load_split(args)

    from transformers import Trainer, TrainingArguments, VoxtralProcessor

    processor = VoxtralProcessor.from_pretrained(args.model_checkpoint)
    model = build_model(args.model_checkpoint, args.target_modules, args.lora_r,
                        args.lora_alpha, args.lora_dropout, args.rslora)
    collator = VoxtralTranscriptionCollator(
        processor, args.model_checkpoint, language=args.language,
        max_length=args.max_length, target_max_tokens=args.target_max_tokens)

    if args.dry_run:
        sample_count = min(2, len(train_dataset))
        batch = collator([train_dataset[i] for i in range(sample_count)])
        model.train()
        outputs = model(**batch)
        outputs.loss.backward()
        print(f"dry run OK -- loss={outputs.loss.item():.4f}, "
             f"input_ids shape={tuple(batch['input_ids'].shape)}")
        return

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(args.per_device_batch_size, 2),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=1,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      eval_dataset=eval_dataset, data_collator=collator)
    trainer.train()

    print(f"saving to {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(str(args.output_dir))

    if eval_dataset:
        print(trainer.evaluate())


if __name__ == "__main__":
    main()
