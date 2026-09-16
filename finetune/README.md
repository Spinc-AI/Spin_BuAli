# Domain fine-tuning

Adapts an audio model to this project's own dictations — English-dominant,
Persian words mixed in, radiology vocabulary — instead of only measuring how
a stock checkpoint does, which is `benchmark/`'s job. "Recognize this kind of
audio better" is a fine-tuning question, not a benchmarking one, and this
folder is the answer to it: two independently researched, cited recipes, each
with a pipeline that is tested as far as it can be without a GPU and designed
to prove itself the rest of the way in minutes, not epochs, once it has one.

## The two recipes, and why both

| | `whisper/` | `voxtral/` |
|---|---|---|
| Base checkpoint | `nezamisafa/whisper-persian-v4` (already this project's own STT registry key `"whisper"`) | `mistralai/Voxtral-Mini-3B-2507` (already the lightest model in the benchmark's `multimodal` roster) |
| What it learns | Better transcription (encoder-decoder ASR) | Better transcription (`apply_transcription_request` mode) |
| Proof it's a real recipe | PEFT's own official example notebook + a published clinical code-switching precedent | Two independent working GitHub implementations |
| Where the result plugs back in | `stt/app/config.py`'s `MODEL_REGISTRY` | `core_llm/config.py`'s `VOXTRAL_MINI_MODEL_ID` |

Whisper is the more conservative, more proven choice — it is the single most
reproduced LoRA fine-tune in the ecosystem, and this dataset's whole shape
(one speaker, mixed-language dictation, no need for the model to also
understand or restructure what it hears) is exactly the ASR task Whisper was
built for. Voxtral is included because it is literally *an LLM* with audio
input, already the model this project's own multimodal benchmark runs, and a
LoRA adapter trained on it changes the same attention/MLP weights the
benchmark's chat-mode inference uses — so a Voxtral fine-tune is the one with
a real chance of moving the benchmark's actual production pipeline, not just
a WER number in a side script. Neither is asserted to be "the" answer; run
`--dry-run` for both, then decide which is worth real GPU-hours.

## Research — what is cited, and why each source is trusted

Every hyperparameter default in this folder traces to one of these. None is
a from-memory guess dressed up as a citation.

1. **[PEFT's official Whisper LoRA example](https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb)**
   (`huggingface/peft`) — `whisper/train_lora.py`'s LoRA config
   (`r=32, lora_alpha=64, target_modules=["q_proj","v_proj"]`), training
   arguments (`lr=1e-3, warmup_steps=50, per_device_train_batch_size=8`), and
   `whisper/collator.py`'s `DataCollatorSpeechSeq2SeqWithPadding` are a direct
   port. Trusted because it is maintained *by the PEFT project itself* as
   their own reference recipe, not a third party's interpretation of one.

2. **[Sanchit Gandhi's "Fine-Tune Whisper for Multilingual ASR"](https://huggingface.co/blog/fine-tune-whisper)**
   (Hugging Face blog) — the historical origin of the `prepare_dataset` /
   feature-extraction-then-collate shape `whisper/collator.py`'s
   `prepare_example` follows. The most-reproduced Whisper fine-tuning
   walkthrough in the ecosystem (community-events repo, multiple ports).

3. **[MediBeng-Whisper-Tiny](https://huggingface.co/pr0mila-gh0sh/MediBeng-Whisper-Tiny)**
   (HF model + [medRxiv preprint](https://www.medrxiv.org/content/10.1101/2025.04.25.25326406v1))
   — the closest published precedent to this project's own domain found:
   Whisper Tiny fine-tuned specifically on **code-switched Bengali-English
   clinical dictation**. Reported WER fell from 107.7 to ~29.5 within under
   one epoch, full fine-tune, on CPU. Cited as evidence the *technique*
   (domain-adapt Whisper on code-switched medical speech) is real and
   published, not as a claim this project's own numbers will match it — a
   different language pair, a different-sized model, and a full fine-tune
   rather than LoRA are all real differences. No public Persian-English
   *clinical* ASR dataset was found (searched specifically; see below) to
   validate against more directly.

4. **[`Deep-unlearning/Finetune-Voxtral-ASR`](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)**
   — `voxtral/collator.py`'s prompt shape
   (`processor.apply_transcription_request` for the prompt half, prompt
   tokens masked with `-100`, target tokenized and appended) is a direct
   port of this repo's `train_lora.py`. The only published, working Voxtral
   ASR fine-tuning implementation found; Apache-2.0 licensed.

5. **[Trelis Research, "Train Voxtral Transcription (ASR) Models"](https://trelis.substack.com/p/train-voxtral-transcription-asr-models)**
   — `voxtral/train_lora.py`'s LoRA hyperparameters
   (`r=32, alpha=32, rslora=True`, targeting attention + MLP + the
   multi-modal projector, frozen audio tower, `lr=5e-5`) are this post's
   reported recipe *for Voxtral Mini specifically* (not a generic LoRA
   default guessed at), with a documented baseline-then-fine-tuned
   evaluation workflow `voxtral/evaluate.py` mirrors. The underlying
   training code and notebook are paywalled (Trelis.com); the
   hyperparameters and workflow description are public and are all that is
   reused here.

6. **A community confirmation from a `transformers` maintainer**
   ([HF discussion](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/discussions/1))
   that Voxtral fine-tunes through plain `transformers`, plus a pointer to
   [`Deep-unlearning/Finetune-Voxtral-ASR`](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)
   (source 4) as a working example — corroborates the approach rather than
   introducing a new one.

**One deliberate omission:** `trl`'s `SFTTrainer` gained a
[work-in-progress PR for audio support](https://github.com/huggingface/trl/pull/5830)
(Qwen2-Audio and Voxtral named explicitly), but it is **on hold, not
merged** — the PR itself says the maintainers "were not getting strong
signal" for it. Building on an unmerged, paused feature would be exactly
the kind of unverified guess this pipeline's own culture (see this repo's
`CLAUDE.md`/commit history: "don't make assumptions, search the internet")
exists to avoid. Both scripts here use the plain `transformers.Trainer` /
`Seq2SeqTrainer` instead, which is also what every cited reference
implementation actually uses.

**Also researched and set aside:** fine-tuning Qwen2-Audio via `ms-swift` or
LLaMA-Factory — both are real, documented options (`ms-swift` explicitly
supports Qwen2-Audio LoRA/QLoRA), but neither produced a citable recipe with
concrete, checkpoint-specific hyperparameters the way the two recipes above
did, and Qwen2-Audio is not yet a strong performer in this project's own
multimodal benchmark (see `benchmark/`'s recent run notes) — not a good
candidate to spend a from-scratch fine-tuning effort on before its
prompt-following issues are better understood.

## What "proven to work" does and does not mean here

**Proven:** every recipe above is a real, published, independently
reproduced technique — not invented for this pipeline. The masking/padding
arithmetic every collator depends on (`sequence.py`) has its own unit tests
(`finetune/tests/test_sequence.py`) that do not need a GPU or even `torch`
installed, and pass. The data-loading and train/eval-split logic
(`data.py`) is tested against real audio files and a real `labels.csv`
(`finetune/tests/test_data.py`, `test_to_hf_dataset.py`). Both CLIs parse
their documented-recipe defaults correctly (`test_cli.py`) and were run for
real against a fixture dataset, reaching exactly the point where they need
`torch`/`transformers`/`datasets` (not installed in the environment this
pipeline was written in) before failing — i.e. everything up to that
boundary is exercised, not just imported.

**Not proven, and said so rather than implied otherwise:** this pipeline has
never executed a real training loop, because the environment it was written
in has no GPU and does not have `torch` installed at all. Neither `--dry-run`
mode has been run for real. No WER number from this project's own data
exists yet. Whether a Voxtral LoRA adapter trained in transcription mode
transfers usefully to the benchmark's chat-mode JSON-report generation is an
open, stated question (see `voxtral/collator.py` and the notebook's cell 9),
not a claim.

**What closes that gap:** `finetune/notebooks/kaggle_finetune.ipynb` (built
by `notebooks/build_notebook.py`, same convention as
`benchmark/notebooks/build_notebook.py`) runs `--dry-run` for both scripts
*before* any real training cell — a five-minute, one-batch, one-step check
that the whole wiring (data → collate → forward → backward) actually works,
on the exact hardware (Kaggle T4×2) this whole project already runs on. Do
that first, on a public dataset if the real one feels too precious to risk a
wasted GPU-hour on a typo. Only nine labelled clips exist in
`Spin_BuAli_DataSet/Small_Demo` today — nowhere near enough to expect a real
fine-tune to generalize from; this pipeline is built to be correct and
ready to scale to the 3-4M-record dataset mentioned as this project's
longer-term goal, not to claim nine examples are enough on their own.

## Layout

```
finetune/
  pipeline_settings.py   knobs (EVAL_FRACTION, MIN_EVAL_ITEMS, SPLIT_SEED, ...)
  data.py                 Item -> HF datasets.Dataset, train/eval split
  sequence.py              prompt+target token-sequence masking/padding (unit-tested, torch-free)
  voxtral/
    collator.py            transcription-request prompt shape, -100 masking
    train_lora.py           LoRA fine-tune CLI
    evaluate.py              baseline vs fine-tuned WER
  whisper/
    collator.py             DataCollatorSpeechSeq2SeqWithPadding (ported from PEFT's example)
    train_lora.py            LoRA fine-tune CLI (includes its own WER eval via compute_metrics)
  notebooks/
    build_notebook.py        generates kaggle_finetune.ipynb
  tests/
    test_sequence.py, test_data.py, test_to_hf_dataset.py, test_cli.py, test_notebook.py
```

`pipeline_settings.py` is not called `settings.py` on purpose — see its own
module docstring: that name collides with `benchmark/settings.py` the moment
`data.py` reaches across to `benchmark/dataset.py`, silently resolving every
later `import settings` to the wrong module. Caught by `test_cli.py` while
building this pipeline, which is exactly the kind of mistake a repo full of
sibling `config.py`/`settings.py` files (see `bridge.py`'s own module
docstring) is one `import` away from making again.

## Running it

```bash
pip install -r finetune/requirements.txt

# Prove the wiring first -- seconds, not epochs:
python -m voxtral.train_lora --labels-csv path/to/labels.csv --dry-run
python -m whisper.train_lora --labels-csv path/to/labels.csv --dry-run

# Then the real thing:
python -m voxtral.train_lora --labels-csv path/to/labels.csv
python -m whisper.train_lora --labels-csv path/to/labels.csv

# Baseline vs fine-tuned, Voxtral:
python -m voxtral.evaluate --labels-csv path/to/labels.csv
python -m voxtral.evaluate --labels-csv path/to/labels.csv --adapter-dir ./voxtral-buali-lora
```

Or, on Kaggle: import `finetune/notebooks/kaggle_finetune.ipynb`, same GPU
T4×2 / Internet On settings as the benchmark notebook.

`labels.csv` is `benchmark/dataset.py`'s own format —
`asset_id,audio,report` columns, audio paths relative to the CSV — reused
rather than reinvented, so the real `Spin_BuAli_DataSet/Small_Demo/labels.csv`
works here unchanged.

## After training

See `finetune/notebooks/kaggle_finetune.ipynb`'s final cell for how to point
`benchmark/runner.py` at a trained checkpoint and get the full metric suite
(medical term F1, negation/laterality/number error rate) instead of WER
alone — a checkpoint that lowers WER but not those is a different, and
arguably more important, finding than one that lowers both.
