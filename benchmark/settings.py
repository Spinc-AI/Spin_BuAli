"""Benchmark knobs, from the environment or `benchmark/.env`.

Deliberately **not** called `config.py`. `bridge.py` puts `evaluation/` on
`sys.path` so this module can reuse the real metrics, and `evaluation/` has a
top-level `config.py` of its own -- two files with that name on the same path
means whichever was inserted first silently wins. The name is the fix.
"""
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# --- Long audio ---
# Whisper's encoder takes a fixed 30-second window: hand it four minutes of
# dictation and it transcribes the first thirty seconds and silently drops the
# rest. Every model here is therefore fed the same overlapping windows, so a
# long-audio penalty never gets mistaken for a model being worse.
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "28"))
OVERLAP_SEC = float(os.getenv("OVERLAP_SEC", "3"))

# When two consecutive windows end and begin with the same words, that is the
# overlap being transcribed twice. Look back at most this many words for it.
MAX_STITCH_OVERLAP_WORDS = int(os.getenv("MAX_STITCH_OVERLAP_WORDS", "40"))

# --- Runtime ---
TARGET_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = os.getenv("BENCHMARK_LANGUAGE", "fa")

# Comma-separated torch devices, or "auto" for every visible GPU (falling back
# to CPU). On Kaggle's dual T4 this resolves to cuda:0,cuda:1 and one replica of
# the model is loaded per GPU, with the audio split between them.
DEVICES = os.getenv("BENCHMARK_DEVICES", "auto")

# Models to run when the caller names none. Keys come from stt's MODEL_REGISTRY.
DEFAULT_MODELS = [
    key for key in os.getenv("BENCHMARK_MODELS", "whisper,seamless").split(",") if key.strip()
]

# --- Output ---
def _default_out_dir():
    """Somewhere writable, which is not always next to the code.

    A Kaggle notebook usually reads the repo from `/kaggle/input`, a read-only
    mount -- so the obvious default of "beside the source" fails at mkdir, at
    the end of a run, after the GPU time has already been spent. `/kaggle/working`
    is the writable half of that runtime and the only directory Kaggle keeps.
    """
    explicit = os.getenv("BENCHMARK_OUT")
    if explicit:
        return pathlib.Path(explicit)
    kaggle_working = pathlib.Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "benchmark_results"
    return REPO_ROOT / "benchmark" / "results"


OUT_DIR = _default_out_dir()

# Where the sibling modules live. Set these if the benchmark is run from a copy
# of the repo that has been rearranged (a Kaggle dataset mount, say).
EVALUATION_DIR = pathlib.Path(os.getenv("EVALUATION_DIR", str(REPO_ROOT / "evaluation")))
STT_DIR = pathlib.Path(os.getenv("STT_DIR", str(REPO_ROOT / "stt")))
CONTROLLER_DIR = pathlib.Path(os.getenv("CONTROLLER_DIR", str(REPO_ROOT / "controller")))
PREPROCESSING_DIR = pathlib.Path(
    os.getenv("PREPROCESSING_DIR", str(REPO_ROOT / "preprocessing")))

# --- The LLM stage ---
# How much the model may write. A radiology report plus the JSON wrapper fits
# well inside this; the cap exists so a looping model ends rather than filling
# the session.
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1536"))

# Optional: score through a running evaluation service instead of in-process.
EVALUATION_URL = os.getenv("EVALUATION_URL") or None
