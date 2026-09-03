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
            model_factory=None, on_progress=None, **window_kwargs) -> dict:
    """Run one configuration end to end and persist it.

    Returns the row that goes into the session report. A failure is recorded
    the same way a success is -- the run is finished either way, and a model
    that cannot load is a result worth keeping rather than an error to retry
    forever.
    """
    terms = terms or bridge.ClinicalTerms()
    started = time.perf_counter()

    runs = []
    for model in run["stt_models"]:
        runs.append(transcribe.transcribe_batch(
            model, items, devices=devices, model_factory=model_factory,
            on_progress=on_progress, **window_kwargs))

    if not runs:
        # multimodal: no speech recognition stage at all. Until the language
        # model stage exists there is nothing to score, so the run is planned
        # and skipped rather than silently counted as measured.
        return {"run_id": run["run_id"], "status": "skipped",
                "reason": "multimodal needs the LLM stage", "seconds": 0.0}

    results, summary = scoring.score_all(runs, items, terms)
    summary["run"] = run
    elapsed = time.perf_counter() - started

    rows = leaderboard.per_report_rows(results)
    for row in rows:
        row.update(run_id=run["run_id"], tier=run["tier"],
                   preprocessing=run["preprocessing"], pipeline=run["pipeline"],
                   llm_model=run["llm_model"], precision=run["precision"])
    store.record(run, rows, summary)

    return {"run_id": run["run_id"], "status": "ok", "seconds": round(elapsed, 1),
            "reports": len(results), "models": len(runs)}


def work_through(plan, items, out_dir, budget=None, tier=None, terms=None,
                 devices=None, model_factory=None, on_run=None,
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
                          model_factory=model_factory, **window_kwargs)
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
