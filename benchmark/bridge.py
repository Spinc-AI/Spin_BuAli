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
    "model_registry", "build_stt_model", "torch_or_none",
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


def torch_or_none():
    """torch if it is installed, else None -- so CPU-only tests can skip."""
    try:
        import torch
    except ImportError:
        return None
    return torch
