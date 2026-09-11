"""The one place the benchmark reaches into the other modules.

Every other service in this repo is import-isolated: it talks to its neighbours
over HTTP and never imports them. A benchmark cannot honour that -- comparing
ten models over hundreds of recordings means loading model weights in-process
(no HTTP round trip per chunk) and calling the metrics directly (no server to
stand up on a Kaggle runtime). So the coupling exists, and it is confined here:
delete this file and nothing else in `benchmark/` knows the other modules
exist.

Two rules keep it from spreading:

* `evaluation/` is imported for the metrics only -- never reimplemented. If a
  metric changes there, the benchmark changes with it.
* `stt/` is imported for its model classes only. Adding a model to
  `stt/app/config.py` is enough to make it benchmarkable; nothing here lists
  models by name.
* `core_llm/` is imported for its **audio-capable** model classes only --
  GemmaAudioModel, QwenOmniModel, and any future class whose `load()` needs
  something other than a plain `AutoModelForCausalLM`. Text-only models
  still load through `llm.LocalLLM`'s own loader, which additionally
  supports the quantized tiers (int8/nf4) `core_llm/` deliberately does not.
  Reusing `core_llm/`'s classes here means a model's audio-handling code is
  written once, not once in production and once (differently) for the
  benchmark.
* `controller/` is imported for its **prompts**, and for nothing else. The
  prompt is the experiment: a benchmark that used its own wording would be
  measuring a system nobody ships. Those strings are tuned and the file that
  holds them says the duplication in it is deliberate -- so they are read, not
  copied.
"""
import sys

import settings


def _ensure_on_path(directory):
    """Put a sibling module's directory on `sys.path`, once.

    Appended rather than inserted: `benchmark/` must keep winning for its own
    module names, whatever the siblings happen to call theirs.
    """
    path = str(directory)
    if path not in sys.path:
        sys.path.append(path)


# --- evaluation ------------------------------------------------------------
# Imported eagerly: pure text processing, no weights, no torch. Keeping it
# eager means `import scoring` fails loudly at import time if the evaluation
# module has moved, rather than half way through a two-hour benchmark run.
_ensure_on_path(settings.EVALUATION_DIR)

from evaluate_results import summarize  # noqa: E402
from extractors import ClinicalTerms  # noqa: E402
from medical_metrics import METRICS_VERSION, evaluate  # noqa: E402
from semantic_metrics import available as semantic_available  # noqa: E402
from semantic_metrics import compute_batch as semantic_batch  # noqa: E402

# --- controller ------------------------------------------------------------
# Prompts only. Eager, and text-only, so it costs nothing and fails loudly if
# the controller moves.
_ensure_on_path(settings.CONTROLLER_DIR)

from prompts import (  # noqa: E402
    RECONCILE,
    TRANSCRIBE_FROM_AUDIO,
    extract_json,
    with_template,
)

__all__ = [
    "ClinicalTerms", "METRICS_VERSION", "evaluate", "summarize",
    "semantic_available", "semantic_batch",
    "RECONCILE", "TRANSCRIBE_FROM_AUDIO", "with_template", "extract_json",
    "model_registry", "build_stt_model", "torch_or_none", "chunk_planner", "speech_regions",
    "build_llm_model",
]



# --- stt -------------------------------------------------------------------
# Imported lazily. `stt/app/config.py` imports torch at module level, and the
# dataset and scoring halves of this package have no business needing a
# multi-hundred-megabyte import to run their tests.
def _stt():
    _ensure_on_path(settings.STT_DIR)
    from app import config as stt_config
    from app import model as stt_model

    return stt_config, stt_model


def model_registry():
    """The models stt knows how to load, in stt's own preference order."""
    stt_config, _ = _stt()
    return dict(stt_config.MODEL_REGISTRY)


def build_stt_model(key, device):
    """Instantiate (without loading) one registered model, pinned to `device`.

    stt's own `build_model()` reads a single process-wide `config.DEVICE`, which
    is right for a service holding one model and wrong here: the whole point of
    two GPUs is two replicas on two different devices at once. Same registry,
    same classes, explicit device.
    """
    stt_config, stt_model = _stt()
    if key not in stt_config.MODEL_REGISTRY:
        raise KeyError(f"unknown model {key!r}; registered: {sorted(stt_config.MODEL_REGISTRY)}")
    spec = dict(stt_config.MODEL_REGISTRY[key])
    cls = stt_model._MODEL_TYPES[spec.pop("type")]
    return cls(model_id=spec.pop("model_id"), device=device, **spec)


def _core_llm():
    """core_llm/model.py, imported lazily and defensively.

    Two problems `_ensure_on_path` (append) does not solve here, unlike for
    `stt/` and `controller/`: core_llm's `model.py` does a bare `import
    config`, and `evaluation/`, `controller/` and `stt/app/` each already
    have their own `config.py` on the path by the time this runs -- exactly
    the collision `_ensure_on_path`'s append-not-insert rule exists to avoid
    for the *other* siblings, but core_llm's flat layout (no subpackage the
    way `stt/app/` has) walks straight into it regardless of append order.
    Mirrors `llm.py`'s `_controller_llm_client()`: insert core_llm's
    directory at the *front* so it wins, drop any already-cached `config` /
    `model` from sys.modules first, import, then restore both so this
    doesn't leave core_llm's `config` shadowing anyone else's afterward.
    """
    path = str(settings.CORE_LLM_DIR)
    original = list(sys.path)
    sys.path.insert(0, path)
    for name in ("config", "model"):
        sys.modules.pop(name, None)
    try:
        import model as core_llm_model

        return core_llm_model
    finally:
        sys.path[:] = original
        for name in ("config", "model"):
            sys.modules.pop(name, None)


def build_llm_model(key):
    """One of core_llm/model.py's registered classes, not yet loaded.

    Only for audio-capable keys -- `llm.build()` is the one caller, and it
    only reaches here for a key in `plan.AUDIO_CAPABLE`. A text-only key
    still goes through `llm.LocalLLM`, which this deliberately does not
    replace: `core_llm/` classes never quantize, and several of the
    text-only tier ladder's placements depend on being able to.

    The returned instance carries a `_core_llm_config` attribute -- core_llm's
    own `config` module, still bound inside the class's `__module__` namespace
    even after this function's sys.modules cleanup -- so a caller can override
    `MAX_NEW_TOKENS` for one generation the same way `settings.LLM_MAX_NEW_TOKENS`
    already can for the text-only path (see `llm.CoreLLMAdapter`).
    """
    core_llm_model = _core_llm()
    if key not in core_llm_model.MODEL_REGISTRY:
        raise KeyError(f"unknown model {key!r}; registered in core_llm/model.py: "
                       f"{sorted(core_llm_model.MODEL_REGISTRY)}")
    cls, model_id = core_llm_model.MODEL_REGISTRY[key]
    instance = cls(model_id)
    instance._core_llm_config = core_llm_model.config
    return instance


# --- preprocessing ---------------------------------------------------------
def chunk_planner():
    """`preprocessing.plan_chunks` and its default configuration.

    Lazy: `preprocessing/vad.py` reaches for silero and torch, and the caller
    only wants the chunker. Imported rather than reimplemented because the
    adaptive strategy -- which nudges every boundary onto the quietest moment
    nearby, so a cut lands between words instead of through one -- is the whole
    reason the preprocessing dimension is worth measuring.
    """
    _ensure_on_path(settings.PREPROCESSING_DIR)
    import chunking
    from common import load_config

    return chunking.plan_chunks, load_config()


def speech_regions(audio, sample_rate):
    """Where preprocessing's VAD thinks the speech is, as [(start, end)].

    Returns None when VAD is unavailable or found nothing, which the caller
    treats as "chunk the whole recording" -- the same non-destructive stance
    preprocessing itself takes, where a wrong VAD call costs a line of metadata
    and never a syllable of audio.
    """
    _ensure_on_path(settings.PREPROCESSING_DIR)
    try:
        import vad
        from common import load_config

        report = vad.run_vad(audio, sample_rate, load_config())
    except Exception:
        return None
    spans = [(float(segment["start_sec"]), float(segment["end_sec"]))
             for segment in report.get("segments") or []]
    return spans or None


def torch_or_none():
    """torch if it is installed, else None -- so CPU-only tests can skip."""
    try:
        import torch
    except ImportError:
        return None
    return torch
