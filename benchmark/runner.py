"""One function per notebook cell: `run_one()`.

This is what makes the notebook match the shape you actually want -- one run,
one cell, one CSV, printed the way the old FLEURS notebook printed it. It does
not loop over a plan and it does not know about tiers or sessions; a human
decides what to run next by writing the next cell, and can stop between any
two without losing anything, because each call finishes by writing its own
file before returning.

Everything it does is a call into an already-tested module:

    transcribe.transcribe_batch   -- STT, windowing, stitching
    llm.build / LocalLLM / CloudLLM -- loading a language model at its tier
    pipeline.build_reports        -- the controller's prompts, one call per clip
    scoring.score_reports         -- the real evaluation metrics
    leaderboard.per_report_rows / rows -- flattening a score into a CSV row

`run_one` only wires them together and writes the file.

Speech recognition is cached across calls, keyed on `(preprocessing, stt_key)`
alone -- not the LLM, not the pipeline. Trying five language models against
one STT engine costs one transcription and five LLM passes, because the
transcript is identical in all five; only the STT stage produces it. The cache
lives on disk as one JSON file per pair, so it survives across cells in the
same session and, if `results_dir` is a Kaggle Dataset re-attached next
session, across sessions too.
"""
import json
import time
from pathlib import Path

import bridge
import leaderboard
import llm as llm_module
import pipeline
import scoring
import settings
import transcribe

# The three lowest-WER models named in docs/STT_Models.pdf, in that
# document's own rank order, on the FLEURS fa_ir benchmark. Keys are
# stt/app/config.py's MODEL_REGISTRY keys, not the checkpoint names -- the
# registry has ten entries; the other seven (mms-fl102, whisper-vhdm,
# whisper-halakoo, mms-all, wav2vec2-xlsr53, whisper-large-v3,
# whisper-large-v3-turbo, and stock Whisper below large-v3) are not run here
# because that PDF already showed they lose.
TOP3_STT = [
    "seamless",         # facebook/seamless-m4t-v2-large      WER 0.107
    "seamless-medium",  # facebook/hf-seamless-m4t-medium     WER 0.134
    "whisper",          # nezamisafa/whisper-persian-v4       WER 0.137
]

# The three lightest LLMs by param count (plan.LLM_PARAMS, fp16) -- used for
# the `separate` pipeline, where the LLM only ever sees text (the STT
# transcript), so audio capability doesn't matter here. medgemma-1.5-4b has
# no audio input at all; it is still the right choice for `separate` because
# that pipeline never needs it. Note gemma-4-e4b's real weight is 7.85B
# despite the "E4B" name -- its "4B" refers to effective compute, not the
# on-disk parameter count.
#
# phi-4-multimodal is deliberately absent, from both this list and
# MULTIMODAL_LLM below. Its checkpoint remains registered in
# core_llm/model.py (Phi4MultimodalModel) for whoever eventually resolves
# this, but it cannot currently load at all: confirmed live across five
# rounds of real fixes (missing pip deps, a stale SlidingWindowCache import,
# a from_pretrained kwarg the checkpoint's config class silently ignores, a
# flash_attn dependency worked around via eager attention) that all
# succeeded in turn, ending on "RuntimeError: Tensor.item() cannot be
# called on meta tensors" inside the vendor's own
# speech_conformer_encoder.py -- its __init__ does real tensor computation,
# which is incompatible with the meta-device fast-init transformers uses by
# default, and low_cpu_mem_usage=False (the standard fix for exactly that
# error) did not change the outcome. That is a structural mismatch between
# this checkpoint's custom code and the installed transformers version, not
# something fixable from a caller's from_pretrained() kwargs.
TOP3_LLM = [
    "medgemma-1.5-4b",  # 4.3B  -> ~8.6 GB
    "gemma-4-e4b",       # 7.85B -> ~15.7 GB
    "aya-expanse-8b",    # 8.03B -> ~16.1 GB
]

# Every AUDIO-CAPABLE LLM that a 2x16 GB machine can hold at some precision,
# lightest first. The `multimodal` pipeline hands the recording straight to
# the LLM, so text-only models (medgemma-1.5-4b, aya-expanse-8b) cannot run
# here at all and the roster is not the same as TOP3_LLM.
#
# Five families, deliberately: Mistral, Google, Alibaba. Sharing one vendor's
# audio front-end across every row would make a family-wide weakness look
# like a property of the task.
#
# phi-4-multimodal would sit second in this list by weight and is excluded
# for the reason documented above TOP3_LLM.
#
# qwen3-omni-30b is ALSO excluded, for a live-confirmed reason of its own:
# nf4 across two cards routes through accelerate's device_map="auto", which
# bin-packs module-by-module rather than reasoning about total free memory --
# and Qwen3-Omni's Thinker carries real weight outside the quantized language
# model (an audio encoder, embeddings) that "auto" tried to place on
# whichever card had room left, came up short on both, and dispatched the
# remainder to CPU. bitsandbytes' 4-bit quantizer refuses that outright:
#
#   ValueError: Some modules are dispatched on the CPU or the disk. Make
#   sure you have enough GPU RAM to fit the quantized model...
#
# That raised cleanly. What actually cost a live Kaggle session a hard reload
# was upstream of it -- see core_llm/model.py's _quantization_config for the
# bitsandbytes logging-spam bug this shares with gemma-4-12b, now fixed
# there. The device_map failure above is unrelated and still open: it needs
# either a hand-built device_map (not "auto") or more GPU than 2x16 GB, and
# guessing at one from here without a live card to test against is how
# phi-4-multimodal cost five rounds of failed fixes. Left registered in
# core_llm/model.py for whoever picks it up with hardware to iterate on.
MULTIMODAL_LLM = [
    "voxtral-mini-3b",   # 4.7B  -> ~10.8 GB   fp16, one card
    "gemma-4-e4b",       # 7.85B -> ~18.1 GB   fp16, two cards
    "qwen2-audio-7b",    # 8.4B  -> ~19.3 GB   fp16, two cards
    "gemma-4-12b",       # 12B   -> ~13.8 GB   int8, two cards
]


def _vram_free() -> list[tuple[int, float, float]]:
    """(device index, free GB, total GB) for every visible GPU."""
    torch = bridge.torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return []
    report = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        report.append((index, free / 1024 ** 3, total / 1024 ** 3))
    return report


def _warn_if_cards_are_occupied(when: str = "before this run started") -> None:
    """Say so plainly when a card is already full before a load starts.

    A load that OOMs inside `from_pretrained` leaves its partial weights
    referenced by the exception traceback -- and in a notebook, IPython keeps
    the last traceback alive, so `unload()` has nothing to drop and
    `empty_cache()` cannot reclaim them. The next run in that kernel then
    fails on memory the previous failure is still holding, which reads like a
    fresh OOM and is really a stale one. Only a kernel restart clears it, so
    the useful thing is to name it before the run rather than after.
    """
    cards = _vram_free()
    if not cards:
        return
    print("  VRAM: " + " | ".join(
        f"cuda:{i} {free:.1f}/{total:.1f} GB free" for i, free, total in cards))
    occupied = [i for i, free, total in cards if free < 0.5 * total]
    if occupied:
        print(f"  WARNING: cuda:{','.join(str(i) for i in occupied)} already "
              f"more than half used {when}, and the start-of-run cleanup could "
              "not reclaim it. Something outside this process (or a reference "
              "this cannot reach) is holding it -- restart the kernel, because "
              "this run will not fit around it.")


def free_vram() -> None:
    """Clear GPU memory now, and say what that recovered.

    `run_one` already does this at the start of every run. This is the same
    thing as a one-liner, for when a cell has just failed and you want the
    cards back without starting another run -- `runner.free_vram()`.
    """
    before = _vram_free()
    _release_leaked_vram()
    after = _vram_free()
    if not before:
        print("no CUDA devices visible")
        return
    for (index, was_free, total), (_, now_free, _total) in zip(before, after):
        recovered = now_free - was_free
        print(f"cuda:{index}  {now_free:.1f}/{total:.1f} GB free"
              + (f"  (recovered {recovered:.1f} GB)" if recovered > 0.05 else ""))


def _peak_vram_gb() -> float:
    torch = bridge.torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return max((torch.cuda.max_memory_allocated(f"cuda:{i}")
               for i in range(torch.cuda.device_count())), default=0.0) / 1024 ** 3


def _reset_vram():
    torch = bridge.torch_or_none()
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(f"cuda:{i}")


def _release_leaked_vram() -> None:
    """Reclaim what a previous *failed* run is still holding, before this one
    allocates anything.

    Unloading at the end of a run only covers runs that reach their end. A
    load that dies inside `from_pretrained` never assigns the model anywhere
    this code can reach, so there is nothing for `unload()` to drop -- but
    the partially built weights stay reachable through the exception that
    carried them out. `sys.last_traceback` holds those frames, and in a
    notebook IPython additionally keeps the last result and the `Out`/`_`
    history, so the references outlive the cell indefinitely and
    `empty_cache()` reclaims nothing.

    Dropping those references first is what makes the collection actually
    free the memory. The cost is the notebook's `Out`/`_` history for the
    cells before this one, which is worth a card.
    """
    import gc
    import sys

    for name in ("last_traceback", "last_value", "last_type", "last_exc"):
        try:
            delattr(sys, name)
        except AttributeError:
            pass

    get_ipython = getattr(sys.modules.get("IPython"), "get_ipython", None)
    shell = get_ipython() if get_ipython else None
    if shell is not None:
        # Clears Out[...] and the _ / __ / ___ back-references, which are the
        # other thing holding a dead run's tensors alive in a notebook.
        try:
            shell.displayhook.flush()
        except Exception:  # noqa: BLE001 - best effort; never fail a run over it
            pass

    gc.collect()
    torch = bridge.torch_or_none()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _transcript_cache_path(results_dir: Path, prep_label: str, stt_key: str) -> Path:
    cache_dir = results_dir / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{prep_label}__{stt_key}.json"


def _load_cached_transcripts(cache_path: Path, asset_ids: set[str]) -> dict | None:
    """The cached `{asset_id: text}` map, or `None` if it does not cover every
    recording this call needs.

    A partial hit -- someone widened the dataset since the cache was written,
    say -- is treated as a miss rather than silently scoring the new
    recordings against nothing: this only ever saves time, never correctness.
    """
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not asset_ids <= set(cached):
        return None
    return cached


def _save_transcripts(cache_path: Path, transcripts_by_asset: dict) -> None:
    flat = {asset_id: slots.get("transcript_1", "")
           for asset_id, slots in transcripts_by_asset.items()}
    cache_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(stt_key: str | None, llm_key: str, pipeline_name: str, items, *,
            language: str = "en", label: str | None = None,
            precision: str = "fp16", cards: int = 1, devices=None,
            preprocessing: str | None = None, structure_guide: str | None = None,
            terms=None, results_dir=None, model_factory=None, llm_factory=None,
            use_cache: bool = True, max_new_tokens: int | None = None):
    """Run exactly one (stt, llm, pipeline) configuration end to end.

    `stt_key=None` is `multimodal`: no transcription stage, the LLM hears the
    recording. Every other pipeline needs at least one STT engine.

    `preprocessing` selects how the recording is windowed before it reaches
    the STT model -- one of `plan.PREPROCESSING`'s four keys, or `None` for
    fixed-length windows. It only matters when `stt_key` is given; a
    `multimodal` run hears the whole recording and has no windowing stage.

    `use_cache` reuses a transcript already produced by an earlier call with
    the same `(preprocessing, stt_key)` pair, regardless of which LLM or
    pipeline that call used -- the transcript is identical either way, only
    the STT stage produces it. Set to `False` to force a fresh transcription,
    which is also what happens automatically when `model_factory` is given: a
    caller supplying a stub wants it exercised, not skipped by a stale cache
    from a real run.

    `max_new_tokens` overrides `settings.LLM_MAX_NEW_TOKENS` for this call
    only. The `separate` pipeline asks the model to fill three full fields
    (`raw_transcript`, `corrected_transcript`, `final_text`), which is easy to
    run past the default 1536-token cap on a real report -- a run cut off
    mid-generation shows up as `no JSON object found` or a `JSONDecodeError`
    on the longer clips specifically, since the model never got to write
    the closing brace.

    Returns the per-clip DataFrame (with a trailing SUMMARY row) and also
    writes it to `results_dir/results__<label>.csv` -- the write happens
    before this returns, so the result is on disk even if the next cell is
    never run.
    """
    import pandas as pd  # notebook-facing; kept out of the module's import time

    results_dir = Path(results_dir or settings.OUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    prep_label = preprocessing or "fixed"
    label = label or f"{stt_key or 'multimodal'}__{llm_key}__{pipeline_name}__{prep_label}"
    terms = terms or bridge.ClinicalTerms()

    print(f"\n{'=' * 70}\n  RUN  {label}\n{'=' * 70}")
    # Before anything allocates: reclaim whatever an earlier run left behind,
    # including a run that died mid-load and whose weights are still pinned by
    # its traceback. Re-running a cell after a failure is the normal case here,
    # not the exception, so this belongs at the start of every run rather than
    # only at the end of the ones that finish.
    _release_leaked_vram()
    _reset_vram()
    _warn_if_cards_are_occupied()

    if not devices:
        # One replica, on whichever card has the most room *now*. A fixed
        # "cuda:0" keeps sending the STT model at the card that previous runs
        # left occupied, which is how a run OOMs mid-transcription with the
        # other card completely free. Deliberately not `resolve_devices()`
        # here: that returns every visible GPU, and transcribe_batch loads one
        # replica per device, which is the opposite of what a memory problem
        # needs.
        best = llm_module.emptiest_cuda_device()
        devices = [best] if best else None

    started = time.perf_counter()

    # --- STT stage -----------------------------------------------------
    transcripts_by_asset = {item.asset_id: {} for item in items}
    stt_run = None
    stt_cached = False
    if stt_key:
        cache_path = _transcript_cache_path(results_dir, prep_label, stt_key)
        cached = (_load_cached_transcripts(cache_path, {item.asset_id for item in items})
                  if use_cache and not model_factory else None)

        if cached is not None:
            stt_cached = True
            print(f"[stt] {stt_key} ({prep_label}) -- cached, skipping transcription")
            for item in items:
                transcripts_by_asset[item.asset_id]["transcript_1"] = cached[item.asset_id]
        else:
            print(f"[stt] {stt_key} ({prep_label})")
            stt_run = transcribe.transcribe_batch(
                stt_key, items, devices=devices, language=language,
                model_factory=model_factory, preprocessing=preprocessing,
                on_progress=lambda t: print(
                    f"    {t.asset_id:14} {t.real_time_factor:6.2f}x real-time"
                    + (f"   ERROR: {t.error}" if t.error else "")))
            for entry in stt_run.transcripts:
                transcripts_by_asset[entry.asset_id]["transcript_1"] = entry.text
            failed = any(entry.error for entry in stt_run.transcripts)
            if use_cache and not failed:
                # Written even when model_factory is a stub, matching the
                # asymmetry on the read side: only the read is gated on
                # "no override given" (below), so a stub's output is never
                # mistaken for a real transcript by a later, un-stubbed call.
                _save_transcripts(cache_path, transcripts_by_asset)
    elif pipeline_name == "separate":
        raise ValueError("separate needs an STT model; pass stt_key")

    # --- LLM stage -------------------------------------------------------
    print(f"[llm] {llm_key} ({precision}, {cards} card(s))")
    _warn_if_cards_are_occupied("with the STT stage already unloaded")
    llm_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens else {}
    model = (llm_factory or llm_module.build)(llm_key, precision=precision, cards=cards, **llm_kwargs)
    load_started = time.perf_counter()
    try:
        model.load()
    except BaseException:
        # BaseException, not Exception: a Kaggle interrupt during "Loading
        # weights..." is KeyboardInterrupt, which Exception does not catch.
        # Without this, interrupting mid-load leaves the (partially) loaded
        # model resident on GPU with nothing left to unload it -- the next
        # cell's load then OOMs against memory nothing in this process still
        # references but CUDA was never told to free.
        model.unload()
        raise
    load_seconds = time.perf_counter() - load_started

    try:
        reports = pipeline.build_reports(
            transcripts_by_asset, model, pipeline_name, structure_guide=structure_guide,
            items=items,
            on_progress=lambda r: print(
                f"    {r.asset_id:14} {r.elapsed_seconds:6.1f}s"
                + (f"   ERROR: {r.error}" if r.error else "")))
    finally:
        peak_vram = _peak_vram_gb()
        model.unload()

    # --- score -------------------------------------------------------------
    run_meta = {"llm_model": llm_key, "pipeline": pipeline_name,
               "stt_model": stt_key or "(none)", "precision": precision}
    results, summary = scoring.score_reports(reports, items, terms, run_meta)

    # Whether this run told the model the expected house-style structure
    # (report_structure.GUIDE) -- stamped on every row so a run made with it
    # and one made without are always distinguishable in a master CSV,
    # rather than silently blended into one ranking.
    rows = leaderboard.per_report_rows(results)
    for row in rows:
        row.update(stt_model=stt_key or "(none)", llm_model=llm_key,
                   pipeline=pipeline_name, precision=precision, preprocessing=prep_label,
                   stt_cached=stt_cached, structure_guided=bool(structure_guide))

    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    if summary["models"]:
        summary_row = leaderboard.rows(summary)[0]
        summary_row.update(
            asset_id="SUMMARY", stt_model=stt_key or "(none)", llm_model=llm_key,
            pipeline=pipeline_name, precision=precision, preprocessing=prep_label,
            stt_cached=stt_cached, structure_guided=bool(structure_guide),
            stt_load_seconds=round(stt_run.load_seconds, 1) if stt_run else 0.0,
            llm_load_seconds=round(load_seconds, 1),
            peak_vram_gb=round(peak_vram, 2),
            total_seconds=round(elapsed, 1),
        )
        frame = pd.concat([frame, pd.DataFrame([summary_row])], ignore_index=True)

    csv_path = results_dir / f"results__{label}.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _print_summary(label, frame, csv_path, elapsed)
    return frame


def _print_summary(label, frame, csv_path, elapsed_seconds):
    print(f"\n{'-' * 70}")
    print(f"  {label}")
    summary_rows = frame[frame.get("asset_id") == "SUMMARY"] if "asset_id" in frame else frame.iloc[0:0]
    if len(summary_rows):
        s = summary_rows.iloc[0]
        print(f"  Corpus WER          : {s.get('corpus_wer', float('nan')):.4f}"
              f"  (p50={s.get('wer_p50', float('nan')):.3f}"
              f"  p90={s.get('wer_p90', float('nan')):.3f})")
        print(f"  Medical term F1     : {s.get('medical_term_f1', float('nan')):.4f}")
        print(f"  Negation error rate : {s.get('negation_error_rate', float('nan')):.4f}")
        print(f"  Laterality err rate : {s.get('laterality_error_rate', float('nan')):.4f}")
        print(f"  Number error rate   : {s.get('number_error_rate', float('nan')):.4f}")
        print(f"  Review rate         : {s.get('review_rate', float('nan')):.1%}")
        print(f"  Peak VRAM           : {s.get('peak_vram_gb', 0):.2f} GB")
        if bool(s.get("stt_cached")):
            print(f"  STT stage           : cached -- no transcription this run")
    print(f"  Wall time           : {elapsed_seconds:.1f}s")
    print(f"  Saved -> {csv_path}")
    print(f"{'-' * 70}")


def build_master(results_dir=None, pattern: str = "results__*.csv"):
    """Merge every per-run CSV's SUMMARY row into one master table.

    Mirrors the previous notebook's master-results cell: read every result
    file, keep only its SUMMARY row, sort by corpus WER, save, print.
    """
    import pandas as pd

    results_dir = Path(results_dir or settings.OUT_DIR)
    paths = sorted(results_dir.glob(pattern))
    print(f"found {len(paths)} result file(s)")

    summaries = []
    for path in paths:
        frame = pd.read_csv(path)
        matches = frame[frame.get("asset_id") == "SUMMARY"] if "asset_id" in frame else frame.iloc[0:0]
        if not len(matches):
            print(f"  [skip] {path.name}: no SUMMARY row")
            continue
        row = matches.iloc[0].to_dict()
        row["source_file"] = path.name
        summaries.append(row)

    if not summaries:
        print("nothing to merge yet")
        return None

    master = pd.DataFrame(summaries).sort_values("corpus_wer").reset_index(drop=True)
    master.index += 1
    master.index.name = "rank"

    master_path = results_dir / "results_master.csv"
    master.to_csv(master_path, encoding="utf-8-sig")
    print(f"\nmaster results -> {master_path}\n")

    columns = [c for c in ("stt_model", "llm_model", "pipeline", "precision", "structure_guided",
                           "corpus_wer", "medical_term_f1", "negation_error_rate",
                           "laterality_error_rate", "number_error_rate",
                           "unit_error_rate", "review_rate", "peak_vram_gb")
              if c in master.columns]
    print(master[columns].to_string())
    return master
