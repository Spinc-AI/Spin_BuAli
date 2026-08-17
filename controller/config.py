"""Config for the BuAli controller (override via environment / .env)."""
import os
import pathlib

from dotenv import load_dotenv

HERE_DIR = pathlib.Path(__file__).parent
load_dotenv(HERE_DIR / ".env")

# STT and Core_LLM (from Spin_Medical_Assistant_Project) are reached over HTTP.
STT_URL = os.getenv("STT_URL", "http://localhost:8000").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001").rstrip("/")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9002"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))

# OpenAI-compatible external provider fallback (STT and/or LLM).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
API_PREFIX = "openai:"

# Gemini's own (non-OpenAI-shaped) generateContent API, for audio input.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
GEMINI_PREFIX = "gemini:"

MAX_STT_SLOTS = 3
