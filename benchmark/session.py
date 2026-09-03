"""Working through the plan one run at a time, so a session can end anywhere.

Everything here exists because a Kaggle session is not yours to keep. Twelve
hours, no warning at the end, and `/kaggle/working` is all that survives. So:

* **A run is the unit of work.** It transcribes, scores, writes its CSV, and
  only then counts as done. Stop between any two and nothing in flight is lost
  because nothing is in flight.
* **The budget stops early, on purpose.** Better to leave forty minutes unused
  than to start a run that gets killed at minute thirty-nine with nothing
  written.
* **Speech recognition is cached across runs.** It is the expensive stage and
  it does not change when the prompt or the language model does, so the second
  run with the same (preprocessing, engines) pair skips it entirely.
"""
import time

import bridge
import leaderboard
import ledger as ledger_module
import llm as llm_module
import pipeline
import scoring
import transcribe


def transcript_key(run: dict) -> str:
    """Identifies a cached STT result: the preprocessing and the engines only.

    Deliberately not the whole run. The point of the cache is that the language
    model, the prompt and the pipeline can all change without re-running speech
    recognition.
    """
    engines = "+".join(run["stt_models"]) or "none"
    return f"{run['preprocessing']}__{engines}"


def execute(run: dict, items, store, terms=None, devices=None,
            model_factory=None, llm_factory=None, on_progress=None,
            **window_kwargs) -> dict:
    """Run one configuration end to end and persist it.

    Two stages. Speech recognition is cached across runs, because it is the
    expensive one and does not change when the prompt or the language model
    does. The report stage then runs once per recording with the model loaded
    once for the batch.

    A failure is recorded the same way a success is -- the run is finished
    either way, and a model that will not load is a result worth keeping
    rather than an error to retry forever.
    """
    terms = terms or bridge.ClinicalTerms()
    started = time.perf_counter()

    transcripts, stt_runs = _transcribe(run, items, store, devices,
                                        model_factory, on_progress, window_kwargs)

    model = (llm_factory or _build_llm)(run)
    try:
        model.load()
    except Exception as error:  # noqa: BLE001 - the plan expected it to fit; it did not
        return _record_failure(run, store, f"LLM load failed: {error}", started)

    try:
        reports = pipeline.build_reports(transcripts, model, run["pipeline"])
    finally:
        # Free the weights before the next run needs the cards.
        model.unload()

    results, summary = scoring.score_reports(reports, items, terms, run)
    summary["run"] = run
    summary["stt"] = [stt_run.summary() for stt_run in stt_runs]
    summary["llm"] = {"model": model.model_key, "precision": model.precision,
                      "load_seconds": round(model.load_seconds, 1)}
    elapsed = time.perf_counter() - started

    rows = leaderboard.per_report_rows(results)
    for row in rows:
        row.update(_stamp(run))
    store.record(run, rows, summary)
    return {"run_id": run["run_id"], "status": "ok", "seconds": round(elapsed, 1),
            "reports": len(results)}


def _transcribe(run, items, store, devices, model_factory, on_progress, window_kwargs):
    """The STT stage, reused from cache when this exact pair has been done.

    Keyed on the preprocessing and the engines only -- which is what makes
    trying five prompts cost five LLM passes and no speech recognition at all.
    """
    if not run["stt_models"]:
        return {item.asset_id: {} for item in items}, []   # multimodal: audio only

    key = transcript_key(run)
    cached = store.cached_transcripts(key)
    if cached is not None and not model_factory:
        return cached, []

    stt_runs = [transcribe.transcribe_batch(
        model, items, devices=devices, model_factory=model_factory,
        on_progress=on_progress, preprocessing=run["preprocessing"],
        **window_kwargs) for model in run["stt_models"]]

    transcripts: dict[str, dict[str, str]] = {item.asset_id: {} for item in items}
    for position, stt_run in enumerate(stt_runs, start=1):
        for entry in stt_run.transcripts:
            transcripts[entry.asset_id][f"transcript_{position}"] = entry.text
    store.cache_transcripts(key, transcripts)
    return transcripts, stt_runs


def _build_llm(run: dict):
    return llm_module.build(run["llm_model"], precision=run["precision"],
                            cards=run.get("cards", 1))


def _stamp(run: dict) -> dict:
    """What every row needs to say about the run that produced it, so the
    combined CSV is readable without the plan beside it."""
    return {"run_id": run["run_id"], "tier": run["tier"],
            "preprocessing": run["preprocessing"], "pipeline": run["pipeline"],
            "stt_models": "+".join(run["stt_models"]) or "(none)",
            "llm_model": run["llm_model"], "precision": run["precision"]}


def _record_failure(run: dict, store, reason: str, started: float) -> dict:
    """Write the run out as failed, so it is not retried forever and the gap
    shows up in the results rather than as an absence."""
    row = {**_stamp(run), "asset_id": "", "status": "failed", "error": reason}
    store.record(run, [row], {"models": [], "run": run, "error": reason})
    return {"run_id": run["run_id"], "status": "failed", "reason": reason,
            "seconds": round(time.perf_counter() - started, 1)}


def work_through(plan, items, out_dir, budget=None, tier=None, terms=None,
                 devices=None, model_factory=None, llm_factory=None, on_run=None,
                 **window_kwargs) -> dict:
    """Execute pending runs until the plan is done or the budget says stop.

    `tier` limits the session to one tier, which is how the three passes are
    kept separate: tier A first and completely, then B, and C never here.
    """
    store = ledger_module.Ledger(out_dir)
    budget = budget or ledger_module.Budget()
    terms = terms or bridge.ClinicalTerms()

    pending = store.pending(plan)
    if tier:
        pending = [run for run in pending if run["tier"] == tier]

    performed, stopped_because = [], ""
    for run in pending:
        allowed, reason = budget.allows_another()
        if not allowed:
            stopped_because = reason
            break
        outcome = execute(run, items, store, terms=terms, devices=devices,
                          model_factory=model_factory, llm_factory=llm_factory,
                          **window_kwargs)
        budget.record_run()
        performed.append(outcome)
        if on_run:
            on_run(run, outcome)

    return {
        "performed": performed,
        "stopped_because": stopped_because,
        "remaining": len(store.pending(plan)),
        "status": store.status(plan),
    }


def combine(out_dir) -> dict:
    """Rebuild the leaderboard from every completed run on disk.

    Rebuilt rather than accumulated, so it always reflects the results that
    actually exist -- including after a session was killed, or a bad run was
    deleted to be redone.
    """
    store = ledger_module.Ledger(out_dir)
    summaries = store.all_summaries()

    models = []
    for entry in summaries:
        run, summary = entry.get("run", {}), entry.get("summary", {})
        for bucket in summary.get("models", []):
            models.append({**bucket, "run_id": run.get("run_id"),
                           "tier": run.get("tier"), "precision": run.get("precision"),
                           "preprocessing": run.get("preprocessing"),
                           "pipeline": run.get("pipeline"),
                           "llm_model": run.get("llm_model")})

    rows = leaderboard.rows({"models": models})
    if rows:
        ledger_module.write_csv(store.root / "leaderboard.csv", rows)
        ledger_module.write_atomic(
            store.root / "leaderboard.md",
            lambda handle: handle.write(leaderboard.to_markdown({"models": models})))
    ledger_module.write_csv(store.root / "all_reports.csv", store.all_rows() or [{}])
    return {"runs": len(summaries), "rows": len(rows)}
