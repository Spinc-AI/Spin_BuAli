"""Scoring the transcripts -- by calling `evaluation/`, never by reimplementing it.

Every number here comes from `medical_metrics.evaluate()` and
`evaluate_results.summarize()`, the same two functions the live service and the
batch CLI use. That is the point: a benchmark that computed its own WER would
eventually disagree with production about which model is better, and there
would be no way to tell which of the two was wrong.

What this module adds is the join -- lining transcripts up with their
references, keeping the unlabelled ones out of the arithmetic, and carrying the
speed measurements through to the summary, since a model is only usable if it
is both accurate and fast enough.
"""
import bridge

# Where a long dictation stops behaving like a short one. Radiology audio spans
# a one-line finding to a multi-minute study, and a model that is fine on the
# first and collapses on the second has an average that says neither.
DURATION_BUCKETS = [("lt_30s", 0, 30), ("30s_2m", 30, 120), ("gt_2m", 120, float("inf"))]


def score_run(run, items, terms):
    """Score one model's transcripts against the labelled items.

    Returns result dicts in exactly the shape `evaluate_results.summarize()`
    consumes, so the aggregate step is the module's own. Semantic metrics are
    deliberately not computed here -- see `attach_semantic()`.
    """
    references = {item.asset_id: item.reference for item in items if item.labelled}

    results = []
    for transcript in run.transcripts:
        reference = references.get(transcript.asset_id)
        if reference is None:
            continue  # unlabelled: transcribed and timed, but nothing to score against
        if transcript.error is not None:
            # A failed transcription is an empty output, not a missing data
            # point. Dropping it would flatter the model that crashed.
            hypothesis = ""
        else:
            hypothesis = transcript.text

        results.append({
            "asset_id": transcript.asset_id,
            "model": run.model,
            "model_version": run.model_id,
            "pipeline": "stt_only",
            "transcription_error": transcript.error,
            "real_time_factor": round(transcript.real_time_factor, 4),
            "audio_seconds": round(transcript.audio_seconds, 2),
            "reference": reference,
            "hypothesis": hypothesis,
            **bridge.evaluate(hypothesis, reference, terms),
        })
    return results


def attach_semantic(results):
    """Add the two embedding metrics to every result, in one pass.

    Kept out of `score_run` on purpose. `evaluate(include_semantic=True)` scores
    one report at a time, which for a benchmark means a batch of one, several
    hundred times over, against models that take longer to load than to run.
    Here every pair goes through together.
    """
    if not results:
        return results

    scores = bridge.semantic_batch([(r["reference"], r["hypothesis"]) for r in results])
    for result, semantic in zip(results, scores):
        result["semantic"] = semantic
    return results


def add_semantic(results, summary):
    """Compute the embedding metrics after the fact and fold them into both
    levels -- per report, and as a per-model mean on the summary.

    Exists so the decision to spend the extra model load can be made after
    seeing the WER, rather than committed to before the run starts.
    """
    attach_semantic(results)
    for bucket in summary.get("models", []):
        bucket["semantic"] = _mean_semantic(
            [r for r in results if r["model"] == bucket["model"]])
    return results, summary


def duration_buckets(results):
    """WER by recording length -- summed edits over summed words, per bucket.

    Long audio is where windowing and stitching can go wrong, and where a model
    with a short attention span quietly degrades. One overall WER hides both.
    """
    summary = {}
    for name, low, high in DURATION_BUCKETS:
        inside = [r for r in results if low <= r.get("audio_seconds", 0) < high]
        if not inside:
            continue
        errors = sum(r["general"]["substitutions"] + r["general"]["insertions"]
                     + r["general"]["deletions"] for r in inside)
        words = sum(r["general"]["reference_words"] for r in inside)
        summary[name] = {
            "reports": len(inside),
            "wer": round(errors / words, 4) if words else 0.0,
            "audio_seconds": round(sum(r["audio_seconds"] for r in inside), 1),
        }
    return summary


def score_all(runs, items, terms=None, include_semantic=False):
    """Score every model's run and aggregate them into one comparison.

    The summary is `evaluate_results.summarize()` output with the speed figures
    folded in per model, so accuracy and cost are read off the same row.
    """
    terms = terms or bridge.ClinicalTerms()

    results = []
    speed = {}
    for run in runs:
        results.extend(score_run(run, items, terms))
        speed[run.model] = run.summary()

    if include_semantic:
        attach_semantic(results)

    summary = bridge.summarize(results, terms) if results else {
        "models": [], "reports": 0,
        "evaluation": {"metrics_version": bridge.METRICS_VERSION,
                       "terms_version": terms.version, "terms_sha": terms.sha},
    }
    for bucket in summary["models"]:
        model_results = [r for r in results if r["model"] == bucket["model"]]
        bucket["speed"] = speed.get(bucket["model"], {})
        bucket["by_duration"] = duration_buckets(model_results)
        if include_semantic:
            bucket["semantic"] = _mean_semantic(model_results)

    # Models whose every recording was unlabelled never reach summarize(),
    # because there was nothing to score. Report their speed anyway rather than
    # letting them vanish from the run.
    scored = {bucket["model"] for bucket in summary["models"]}
    summary["unscored_models"] = [
        {"model": model, "speed": figures, "reason": "no labelled recordings"}
        for model, figures in speed.items() if model not in scored
    ]
    summary["labelled"] = sum(1 for item in items if item.labelled)
    summary["unlabelled"] = sum(1 for item in items if not item.labelled)
    return results, summary


def _mean_semantic(results):
    """Both embedding scores are similarities in [0, 1], so a plain mean is the
    right aggregate -- unlike an error rate, there is no denominator to sum."""
    scored = [r["semantic"] for r in results if r.get("semantic")]
    if not scored:
        return {}
    return {f"{field}_mean": round(sum(s[field] for s in scored) / len(scored), 4)
            for field in ("bertscore_f1", "semantic_similarity")}
