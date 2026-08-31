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
