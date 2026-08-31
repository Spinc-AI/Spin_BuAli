"""Evaluation service configuration, from the environment or `evaluation/.env`."""
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))

# The concept vocabulary. Point this elsewhere to score against a different
# term set without touching the code.
CLINICAL_TERMS_PATH = os.getenv(
    "CLINICAL_TERMS_PATH",
    str(pathlib.Path(__file__).parent / "clinical_terms.json"),
)

# --- Optional embedding metrics (semantic_metrics.py) ---
# Only loaded when a request asks for them. See requirements-semantic.txt.
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "HooshvareLab/bert-base-parsbert-uncased")
BERTSCORE_LAYERS = int(os.getenv("BERTSCORE_LAYERS", "8"))
SIMILARITY_MODEL = os.getenv(
    "SIMILARITY_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# A report worse than this counts as catastrophic in the batch summary.
CATASTROPHIC_WER = float(os.getenv("CATASTROPHIC_WER", "1.0"))

# --- Degenerate-output thresholds ---
# A looping or wildly padded output is unusable regardless of how the clinical
# counters read, so these also raise requires_medical_review.
MAX_REPETITION = float(os.getenv("MAX_REPETITION", "0.5"))
MAX_LENGTH_RATIO = float(os.getenv("MAX_LENGTH_RATIO", "2.0"))
MIN_LENGTH_RATIO = float(os.getenv("MIN_LENGTH_RATIO", "0.5"))
