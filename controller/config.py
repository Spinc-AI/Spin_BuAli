"""Controller configuration, read from the environment or `controller/.env`."""
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

# --- This service ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9002"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))

# --- Sibling services in this repo, reached over HTTP ---
STT_URL = os.getenv("STT_URL", "http://localhost:8000").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001").rstrip("/")
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://localhost:8002").rstrip("/")

# --- Cloud providers, selected per model by prefix (see providers.py) ---
OPENAI_PREFIX = "openai:"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")

GEMINI_PREFIX = "gemini:"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).rstrip("/")

MAX_STT_SLOTS = 3

# --- Backend -> controller authentication ---
# Every route except GET / requires this in an X-Internal-Token header. Empty
# means the controller refuses work rather than accepting it unauthenticated:
# a missing secret is a deployment mistake, not permission to skip the check.
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

# Largest audio upload accepted. Enforced while reading the stream, so an
# oversized file is refused before it is held in memory, not after.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


# --- Version pinning -------------------------------------------------------
# A job declares which preprocessing, pipeline and prompt it expects, and the
# controller refuses anything it does not implement. Without this a result
# cannot be traced back to the code that produced it: the same request would
# quietly mean something different after a deploy.
def _version_set(name: str, default: str) -> set[str]:
    return {value.strip() for value in os.getenv(name, default).split(",") if value.strip()}


SUPPORTED_PREPROCESSING_VERSIONS = _version_set(
    "SUPPORTED_PREPROCESSING_VERSIONS", "legacy-v1")
SUPPORTED_PIPELINE_VERSIONS = _version_set("SUPPORTED_PIPELINE_VERSIONS", "buali-v1")
SUPPORTED_PROMPT_VERSIONS = _version_set("SUPPORTED_PROMPT_VERSIONS", "radiology-v1")
