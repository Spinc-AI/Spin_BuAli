# Domain fine-tuning: Gemma 4

LoRA fine-tunes Gemma 4 — the encoder-free, audio-capable "Unified" variant
already in the benchmark's own `multimodal` roster (`gemma-4-e4b`,
`gemma-4-12b`) — on this project's own dictations: English-dominant, Persian
words mixed in, radiology vocabulary. Trained against the **exact prompt
shape production uses**, not a separate transcription-only format, so there
is no "does this transfer to the real pipeline?" gap to wonder about
afterward.

## Why Gemma, and why chat-mode training specifically

`core_llm/model.py`'s `GemmaAudioModel` prompts Gemma at inference with a
chat template: a system turn (the JSON-report instructions from
`controller/prompts.py`), a user turn carrying the audio, and — once
generated — an assistant turn holding the model's JSON reply. This pipeline
trains against exactly that shape (`gemma/collator.py` builds the same
system+user prompt via `benchmark/pipeline.py`'s own `audio_prompt()`, and
appends an assistant turn as the training target). A fine-tune done any
other way — a transcription-only prompt, say — would change the same
weights the real pipeline reads, but would only ever have been *rewarded*
for a task the benchmark never actually asks the model to do. This one
closes that gap by construction.

## Research

1. **["Fine-Tuning Gemma 4 for Transcription"](https://debuggercafe.com/fine-tuning-gemma-4-for-transcription/)**
   (DebuggerCafe) — the one precedent found that fine-tunes a Gemma 4
   checkpoint on audio through the chat template (system/user/assistant
   turns) rather than a separate transcription-only API. Reports the model
   going from failing the task entirely to producing usable
   transcription+translation output after LoRA. `gemma/train_lora.py`'s
   `--lora-r 8 --lora-alpha 16` defaults are this article's reported recipe.

2. **[PEFT's `target_modules="all-linear"`](https://huggingface.co/docs/peft/en/developer_guides/lora)**
   — documented, verified PEFT behaviour (every linear/Conv1D layer, output
   layer excluded) used here instead of the article's own reported
   audio-specific layer names (`post`, `linear_start`, `linear_end`,
   `embedding_projection`). Those names come from a blog post, not
   independently verified against this project's installed `transformers`
   version's own `AutoModelForMultimodalLM` source — guessing wrong would
   silently train zero of the intended layers rather than raising.
   `"all-linear"` needs no such guess, and Gemma 4's architecture gives it
   nothing to miss: the "Unified" design has no separate audio encoder the
   way Whisper or Voxtral do (raw audio projects straight into the shared
   decoder — see `core_llm/model.py`'s `GemmaAudioModel` docstring, and
   [this architecture summary](https://dev.to/creeta/gemma-4-12b-skips-the-audio-encoder-is-16-gb-enough-3mmp)),
   so there is no audio-specific component living *outside* the linear
   layers the wildcard already reaches.

3. **[`benchmark/pipeline.py`](../benchmark/pipeline.py) and
   [`controller/prompts.py`](../controller/prompts.py)** — not third-party
   research, but the actual source of the system prompt and JSON contract
   this pipeline trains against. Imported, not copied, for the reason
   `pipeline.py`'s own module docstring gives: "A benchmark that phrased the
   instruction its own way would be ranking models on a system nobody
   ships." Training against a copy would have exactly that problem.

**Also researched and set aside:** Whisper (PEFT's own official LoRA
example, plus a published precedent — MediBeng-Whisper-Tiny — fine-tuning
Whisper on code-switched Bengali-English *clinical* speech, WER 107.7→~29.5)
and Voxtral (`Deep-unlearning/Finetune-Voxtral-ASR` + Trelis Research's
published Voxtral Mini hyperparameters) both had real, citable recipes and
were built and tested at one point in this branch's history. Removed by
request to keep this folder to the one model actually wanted — their
recipes are recorded here in case a Whisper- or Voxtral-shaped need comes up
again; both are real, working precedents, just not what this folder builds
today.

**Deliberately not used:** `trl`'s `SFTTrainer` gained a
[work-in-progress PR for audio support](https://github.com/huggingface/trl/pull/5830),
but it is on hold, not merged — the PR itself says the maintainers "were not
getting strong signal" for it. `gemma/train_lora.py` uses plain
`transformers.Trainer` instead.

## What "proven to work" does and does not mean here

**Proven:** the recipe is real, published, and reproduced elsewhere (see
above) — not invented for this pipeline. The prompt/target masking-and-
padding arithmetic every collator in this repo depends on
(`sequence.py`) has its own unit tests (`finetune/tests/test_sequence.py`)
that need no GPU or even `torch` installed, and pass. `gemma/collator.py`'s
`build_target` — the one piece of this recipe with no external precedent to
cite, because it fills in this project's own JSON contract from a plain
report label — is unit-tested directly
(`finetune/tests/test_gemma_collator.py`). Data loading and the train/eval
split (`data.py`) are tested against real audio files and a real
`labels.csv`. `gemma/train_lora.py` and `gemma/evaluate.py` were both run
for real against a fixture dataset and reached exactly the point where
`torch`/`transformers`/`datasets` (not installed in the environment this
pipeline was written in) are needed, before failing — everything before
that boundary is exercised, not just imported.

**Not proven:** no training loop has actually executed. No WER number from
this project's own data exists yet. `GemmaChatCollator`'s handling of a
per-device batch size greater than 1 (concatenating each example's own
audio-feature tensors) is a reasonable extrapolation from a real, running
precedent (`GemmaAudioModel` itself never batches more than one
conversation per call, anywhere in this repo), not something verified
against a documented multi-clip Gemma 4 training example — none was found.
That is exactly why `--per-device-batch-size` defaults to **1**, with
`--gradient-accumulation-steps` as the documented way to reach a larger
effective batch instead of raising it.

**What closes the remaining gap:** `finetune/notebooks/kaggle_finetune.ipynb`
runs `--dry-run` — one batch, one forward+backward step — *before* any real
training cell, on the exact hardware (Kaggle T4×2) this whole project
already runs on. Do that first, on a public dataset if the real one feels
too precious to risk a wasted GPU-hour on a typo. Only nine labelled clips
exist in `Spin_BuAli_DataSet/Small_Demo` today — nowhere near enough to
expect a real fine-tune to generalize from; this pipeline is built to be
correct and ready to scale to the 3-4M-record dataset mentioned as this
project's longer-term goal, not to claim nine examples are enough on their
own.

## Layout

```
finetune/
  pipeline_settings.py   knobs (EVAL_FRACTION, MIN_EVAL_ITEMS, SPLIT_SEED, ...)
  data.py                 Item -> HF datasets.Dataset (asset_id, audio_path, text), train/eval split
  sequence.py              prompt+target token-sequence masking/padding (unit-tested, torch-free)
  gemma/
    collator.py             chat-template prompt (via benchmark/pipeline.py's audio_prompt),
                              JSON target built from the reference report (build_target)
    train_lora.py            LoRA fine-tune CLI
    evaluate.py               baseline vs fine-tuned WER on final_text
  notebooks/
    build_notebook.py         generates kaggle_finetune.ipynb
  tests/
    test_sequence.py, test_data.py, test_to_hf_dataset.py,
    test_gemma_collator.py, test_cli.py, test_notebook.py
```

`pipeline_settings.py` is not called `settings.py` on purpose — see its own
module docstring: that name collided with `benchmark/settings.py` the
moment `data.py` reached across to `benchmark/dataset.py`, silently
resolving every later `import settings` to the wrong module. Caught by
`test_cli.py` while building this pipeline, which is exactly the kind of
mistake a repo full of sibling `config.py`/`settings.py` files (see
`bridge.py`'s own module docstring) is one `import` away from making again.

## Running it

```bash
pip install -r finetune/requirements.txt

# Prove the wiring first -- seconds, not epochs:
python -m gemma.train_lora --labels-csv path/to/labels.csv --dry-run

# Then the real thing:
python -m gemma.train_lora --labels-csv path/to/labels.csv

# Baseline vs fine-tuned:
python -m gemma.evaluate --labels-csv path/to/labels.csv
python -m gemma.evaluate --labels-csv path/to/labels.csv --adapter-dir ./gemma-buali-lora
```

Or, on Kaggle: import `finetune/notebooks/kaggle_finetune.ipynb`, same GPU
T4×2 / Internet On settings as the benchmark notebook.

`labels.csv` is `benchmark/dataset.py`'s own format —
`asset_id,audio,report` columns, audio paths relative to the CSV — reused
rather than reinvented, so the real `Spin_BuAli_DataSet/Small_Demo/labels.csv`
works here unchanged.

`--model-checkpoint` defaults to `google/gemma-4-E4B-it` (matching
`core_llm/config.py`'s `GEMMA_E4B_MODEL_ID`). Pass `--model-checkpoint
google/gemma-4-12B-it --load-in-8bit` for the larger model — `benchmark/tiers.py`
already places it at int8 on this project's own T4×2 hardware, and this
script's own `--load-in-8bit` flag follows the same choice.

## After training

Merge the LoRA adapter into the base weights (`PeftModel.merge_and_unload()`,
as `gemma/evaluate.py` already does) and save that; then set
`GEMMA_E4B_MODEL_ID=<merged checkpoint path>` before
`benchmark/notebooks/kaggle_dual_t4.ipynb`'s cell 3 — `core_llm/config.py`
reads that environment variable as `GemmaAudioModel`'s checkpoint for the
`gemma-4-e4b` key. Run that notebook unchanged after that: it will score the
fine-tuned checkpoint through the full metric suite (medical term F1,
negation/laterality/number error rate), not WER alone — a checkpoint that
lowers WER but not those is a different, and arguably more important,
finding than one that lowers both.
