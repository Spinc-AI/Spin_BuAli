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
"""
import time
from pathlib import Path

import bridge
import leaderboard
import llm as llm_module
import pipeline
import scoring
import settings
import transcribe

# The five models named in docs/STT_Models.pdf, in that document's own rank
# order (lowest WER first) on the FLEURS fa_ir benchmark. Keys are
# stt/app/config.py's MODEL_REGISTRY keys, not the checkpoint names -- the
# registry has ten entries; the other five (whisper-halakoo, mms-all,
# wav2vec2-xlsr53, whisper-large-v3, whisper-large-v3-turbo, and stock Whisper
# below large-v3) are not run here because that PDF already showed they lose.
TOP5_STT = [
    "seamless",         # facebook/seamless-m4t-v2-large      WER 0.107
    "seamless-medium",  # facebook/hf-seamless-m4t-medium     WER 0.134
    "whisper",          # nezamisafa/whisper-persian-v4       WER 0.137
    "mms-fl102",        # facebook/mms-1b-fl102               WER 0.146 (in-domain, optimistic -- see the PDF)
    "whisper-vhdm",     # vhdm/whisper-large-fa-v1            WER 0.150
]


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


def run_one(stt_key: str | None, llm_key: str, pipeline_name: str, items, *,
            language: str = "fa", label: str | None = None,
            precision: str = "fp16", cards: int = 1, devices=None,
            preprocessing: str | None = None, structure_guide: str | None = None,
            terms=None, results_dir=None, model_factory=None, llm_factory=None):
    """Run exactly one (stt, llm, pipeline) configuration end to end.

    `stt_key=None` is `multimodal`: no transcription stage, the LLM hears the
    recording. Every other pipeline needs at least one STT engine.

    `preprocessing` selects how the recording is windowed before it reaches
    the STT model -- one of `plan.PREPROCESSING`'s four keys, or `None` for
    fixed-length windows. It only matters when `stt_key` is given; a
    `multimodal` run hears the whole recording and has no windowing stage.

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
    _reset_vram()
    started = time.perf_counter()

    # --- STT stage -----------------------------------------------------
    transcripts_by_asset = {item.asset_id: {} for item in items}
    stt_run = None
    if stt_key:
        print(f"[stt] {stt_key}")
        stt_run = transcribe.transcribe_batch(
            stt_key, items, devices=devices, language=language,
            model_factory=model_factory, preprocessing=preprocessing,
            on_progress=lambda t: print(
                f"    {t.asset_id:14} {t.real_time_factor:6.2f}x real-time"
                + (f"   ERROR: {t.error}" if t.error else "")))
        for entry in stt_run.transcripts:
            transcripts_by_asset[entry.asset_id]["transcript_1"] = entry.text
    elif pipeline_name == "separate":
        raise ValueError("separate needs an STT model; pass stt_key")

    # --- LLM stage -------------------------------------------------------
    print(f"[llm] {llm_key} ({precision}, {cards} card(s))")
    model = (llm_factory or llm_module.build)(llm_key, precision=precision, cards=cards)
    load_started = time.perf_counter()
    model.load()
    load_seconds = time.perf_counter() - load_started

    try:
        reports = pipeline.build_reports(
            transcripts_by_asset, model, pipeline_name, structure_guide=structure_guide,
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

    rows = leaderboard.per_report_rows(results)
    for row in rows:
        row.update(stt_model=stt_key or "(none)", llm_model=llm_key,
                   pipeline=pipeline_name, precision=precision, preprocessing=prep_label)

    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    if summary["models"]:
        summary_row = leaderboard.rows(summary)[0]
        summary_row.update(
            asset_id="SUMMARY", stt_model=stt_key or "(none)", llm_model=llm_key,
            pipeline=pipeline_name, precision=precision, preprocessing=prep_label,
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

    columns = [c for c in ("stt_model", "llm_model", "pipeline", "precision",
                           "corpus_wer", "medical_term_f1", "negation_error_rate",
                           "laterality_error_rate", "number_error_rate",
                           "unit_error_rate", "review_rate", "peak_vram_gb")
              if c in master.columns]
    print(master[columns].to_string())
    return master
