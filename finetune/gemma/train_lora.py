#!/usr/bin/env python3
"""LoRA fine-tune of Gemma 4 (audio-capable, encoder-free) on this project's
own dictations, trained against the exact prompt shape production uses.

    python -m gemma.train_lora --labels-csv /path/to/labels.csv --dry-run
    python -m gemma.train_lora --labels-csv /path/to/labels.csv

Recipe sourced from a published, working Gemma 4 audio fine-tune -- see
`finetune/README.md`'s Research section for the full citation and its
honest gaps:

  * "Fine-Tuning Gemma 4 for Transcription" (DebuggerCafe) -- the one
    precedent found that fine-tunes a Gemma 4 checkpoint on audio through
    the *chat template*, system/user/assistant turns, the same shape
    `core_llm/model.py`'s `GemmaAudioModel` and `gemma/collator.py` use --
    not a separate transcription-only API. Reports the model going from
    failing the task entirely to producing usable output after LoRA.

Two real differences from that source, both deliberate:

  * `--target-modules` defaults to PEFT's own `"all-linear"` wildcard, not
    the article's specific audio-layer names (`post`, `linear_start`,
    `linear_end`, `embedding_projection`). Those names come from a blog
    post, not independently verified against this transformers version's
    own `AutoModelForMultimodalLM` source -- guessing wrong would silently
    train zero of the intended layers rather than raising. `"all-linear"`
    is documented, verified PEFT behaviour (targets every linear/Conv1D
    layer in the model) and needs no such guess. Pass the article's names
    explicitly with `--target-modules` if you have verified them against
    the installed transformers version yourself.
  * The training target is this project's own `report_template.json`
    contract (`raw_transcript`/`corrected_transcript`/`final_text`/
    `discrepancies_found`/`notes`), not the article's transcription+
    translation text -- see `gemma/collator.py`'s `build_target`.

`--dry-run` runs the whole pipeline -- load data, build one batch, one
forward+backward step -- end to end before any real GPU-hours are spent.
"""
import argparse
import pathlib
import sys

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

import data as ft_data  # noqa: E402
import pipeline_settings as settings  # noqa: E402
from gemma.collator import GemmaChatCollator  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-csv", type=pathlib.Path, required=True,
                        help="labels.csv or a JSON manifest -- see benchmark/dataset.py")
    parser.add_argument("--model-checkpoint", default="google/gemma-4-E4B-it",
                        help="Matches core_llm/config.py's GEMMA_E4B_MODEL_ID default -- "
                             "use google/gemma-4-12B-it for the larger, int8-tier model "
                             "instead (pass --load-in-8bit too; see below).")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("./gemma-buali-lora"))
    parser.add_argument("--structure-guide", action="store_true",
                        help="Include report_structure.GUIDE in the system prompt, "
                             "matching the benchmark's own default (structure_guided=True) "
                             "-- see benchmark/notebooks build_notebook.py cell 6's note on "
                             "why that default exists.")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Quantize the base model (bitsandbytes int8) before adding "
                             "the LoRA adapter -- needed for gemma-4-12B-it on a 16 GB "
                             "card, matching the tier benchmark/tiers.py already places "
                             "it at.")

    split = parser.add_argument_group("train/eval split")
    split.add_argument("--eval-fraction", type=float, default=settings.EVAL_FRACTION)
    split.add_argument("--min-eval-items", type=int, default=settings.MIN_EVAL_ITEMS)
    split.add_argument("--split-seed", type=int, default=settings.SPLIT_SEED)

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--lora-r", type=int, default=8)
    lora.add_argument("--lora-alpha", type=int, default=16)
    lora.add_argument("--lora-dropout", type=float, default=0.0)
    lora.add_argument("--target-modules", nargs="+", default=["all-linear"],
                      help="PEFT's own wildcard by default -- see this module's "
                           "docstring for why the article's specific audio-layer names "
                           "are not the default.")

    train = parser.add_argument_group("training")
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--epochs", type=float, default=3.0)
    # 1, deliberately: see GemmaChatCollator's own docstring -- batch size 1
    # is the only shape this pipeline's audio-feature concatenation has been
    # reasoned through against a real, running precedent (GemmaAudioModel
    # itself never batches audio either). Raise gradient-accumulation-steps
    # for a larger effective batch instead of raising this.
    train.add_argument("--per-device-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--max-length", type=int, default=1536,
                       help="Total sequence length (prompt + target + eos) a batch is "
                            "padded to; the target is truncated to fit, never the "
                            "prompt -- see sequence.build_batch.")
    train.add_argument("--target-max-tokens", type=int, default=768)
    train.add_argument("--logging-steps", type=int, default=5)
    train.add_argument("--eval-steps", type=int, default=20)
    train.add_argument("--save-steps", type=int, default=20)

    parser.add_argument("--dry-run", action="store_true",
                        help="Build the model, the dataset, one batch, and run a single "
                             "forward+backward step, then exit -- proves the pipeline is "
                             "wired correctly before spending real GPU-hours on it.")
    return parser.parse_args(argv)


def build_model(model_checkpoint, target_modules, r, alpha, dropout, load_in_8bit):
    """The base checkpoint wrapped in a LoRA adapter.

    No component is frozen the way Whisper's/Voxtral's audio encoder would
    be -- Gemma 4's "Unified" architecture has no separate audio tower to
    freeze at all (raw audio projects straight into the shared decoder;
    see `core_llm/model.py`'s `GemmaAudioModel` docstring), which is also
    why `"all-linear"` covers the audio-adjacent projection layers along
    with everything else: there is nothing audio-specific living outside
    the linear layers this wildcard already reaches.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM

    load_kwargs = {"device_map": "auto"}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForMultimodalLM.from_pretrained(
        model_checkpoint, attn_implementation="sdpa", **load_kwargs)

    lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                             target_modules=target_modules, task_type="CAUSAL_LM")
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

    from transformers import AutoProcessor, Trainer, TrainingArguments

    structure_guide = None
    if args.structure_guide:
        sys.path.insert(0, str(FINETUNE_DIR.parent / "benchmark"))
        import report_structure

        structure_guide = report_structure.GUIDE

    processor = AutoProcessor.from_pretrained(args.model_checkpoint, padding_side="left")
    model = build_model(args.model_checkpoint, args.target_modules, args.lora_r,
                        args.lora_alpha, args.lora_dropout, args.load_in_8bit)
    collator = GemmaChatCollator(processor, structure_guide=structure_guide,
                                 target_max_tokens=args.target_max_tokens,
                                 max_length=args.max_length)

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
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=not args.load_in_8bit,
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
