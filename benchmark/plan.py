"""The run matrix: every configuration to be measured, in the order to run it.

A run is one tuple of (preprocessing, pipeline, STT engines, LLM). The plan is
written to disk before anything executes, so the whole experiment can be read,
argued with and edited as a table rather than inferred from what a script
happened to loop over.

Order matters, and it is not arbitrary:

    tier A before B before C     cheap and trustworthy results first, so a
                                 session that only gets an hour still produces
                                 something worth reading
    within a tier, small first   a model that will not load should say so in
                                 two minutes rather than forty

Tier C runs are planned but never executed here. They are the same models at
full precision, which this hardware cannot hold; keeping them in the plan is
what turns "we did not measure that" into a row with a reason.
"""
import itertools

import ledger
import tiers

# Parameter counts, in billions. Only used to decide what fits before the
# weights exist locally; the runner reports what was actually used.
LLM_PARAMS = {
    "aya-expanse-8b": 8.03,
    "aya-expanse-32b": 32.3,
    "gemma-4-31b": 31.0,
    "gemma-4-e4b": 7.85,
    "gemma-4-12b": 12.0,
    "qwen3-omni-30b": 30.5,
}

# Which local LLMs can take audio. `separate` sends text only, so it can use
# any of them; `multimodal` and `hybrid` cannot.
AUDIO_CAPABLE = {"gemma-4-e4b", "gemma-4-12b", "qwen3-omni-30b"}

# Preprocessing variants, as (id, description). Chunking is what actually
# differs; VAD is included because it moves the adaptive chunk boundaries even
# though it never cuts audio itself.
PREPROCESSING = {
    "adaptive-vad": "adaptive chunking, VAD-guided boundaries",
    "adaptive": "adaptive chunking, no VAD",
    "uniform": "even chunks, chosen from duration alone",
    "fixed": "strict target-length windows",
}

PIPELINES = ("separate", "multimodal", "hybrid")


def cloud_llms(models: list[str] | None) -> list[str]:
    """Cloud models are named by the caller -- they have no parameter count to
    place and no VRAM cost, so they sit outside the tier system entirely."""
    return list(models or [])


def stt_combinations(models: list[str], max_slots: int) -> list[tuple[str, ...]]:
    """Every 1..max_slots subset of the STT engines.

    Order-independent: `separate` reconciles its slots, so whisper+seamless is
    the same experiment as seamless+whisper and running both would double the
    cost for one number.
    """
    combinations: list[tuple[str, ...]] = []
    for size in range(1, max_slots + 1):
        combinations.extend(itertools.combinations(models, size))
    return combinations


def build(stt_models: list[str], llm_models: list[str], usable_gb: float, cards: int,
          pipelines=PIPELINES, preprocessing=tuple(PREPROCESSING),
          max_slots: int = 1, cloud=None) -> list[dict]:
    """Every run to be measured, each with its tier and placement decided."""
    placements = {}
    for placement in tiers.plan_placements(
            {name: LLM_PARAMS[name] for name in llm_models if name in LLM_PARAMS},
            usable_gb, cards):
        # A model appears twice when it was quantized: once as it will run
        # here, once deferred at full precision. Keyed so both survive.
        key = (placement.model, placement.tier)
        placements[key] = placement

    runs = []
    slot_sets = stt_combinations(stt_models, max_slots)

    for prep in preprocessing:
        for pipeline in pipelines:
            for (model, tier), placement in sorted(placements.items()):
                if pipeline != "separate" and model not in AUDIO_CAPABLE:
                    continue  # the LLM has to hear the audio for these two
                slots = [()] if pipeline == "multimodal" else slot_sets
                for engines in slots:
                    runs.append(_run(prep, pipeline, engines, model, placement))

            for model in cloud_llms(cloud):
                slots = [()] if pipeline == "multimodal" else slot_sets
                for engines in slots:
                    runs.append(_run(prep, pipeline, engines, model, None))

    return sorted(runs, key=_order)


def _run(prep: str, pipeline: str, engines: tuple, llm: str, placement) -> dict:
    spec = {
        "preprocessing": prep,
        "pipeline": pipeline,
        "stt_models": list(engines),
        "llm_model": llm,
        "precision": placement.precision if placement else "cloud",
        "cards": placement.cards if placement else 0,
    }
    return {
        "run_id": ledger.run_id(spec),
        **spec,
        "tier": placement.tier if placement else "A",
        "params_b": placement.params_b if placement else 0.0,
        "vram_gb": round(placement.vram_gb, 1) if placement else 0.0,
        "placement": placement.describe() if placement else "cloud API, no local VRAM",
    }


def _order(run: dict) -> tuple:
    """Tier first, then cheapest within a tier."""
    return (run["tier"], run["params_b"], len(run["stt_models"]),
            run["preprocessing"], run["pipeline"], run["llm_model"])


def to_rows(plan: list[dict]) -> list[dict]:
    """The plan flattened for a CSV -- the STT list becomes one readable cell."""
    return [{**run, "stt_models": "+".join(run["stt_models"]) or "(none)"}
            for run in plan]


def summarize(plan: list[dict]) -> dict:
    by_tier: dict[str, int] = {}
    for run in plan:
        by_tier[run["tier"]] = by_tier.get(run["tier"], 0) + 1
    return {
        "runs": len(plan),
        "runnable": sum(1 for run in plan if run["tier"] != "C"),
        "deferred": by_tier.get("C", 0),
        "by_tier": dict(sorted(by_tier.items())),
    }
