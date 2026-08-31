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
