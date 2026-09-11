"""Central configuration, loaded from the environment or a .env file.

Keeping it here means nothing else hard-codes a server address or model name.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; no-op otherwise

# --- HTTP server ---
# Port 8001 keeps this clear of the STT service on 8000.
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Local models ---
# Lazy-loaded on first request, one at a time (see model.LLMManager).
AYA_8B_MODEL_ID = os.getenv("AYA_8B_MODEL_ID", "CohereLabs/aya-expanse-8b")
AYA_32B_MODEL_ID = os.getenv("AYA_32B_MODEL_ID", "CohereLabs/aya-expanse-32b")
# Gemma 4 only accepts audio on the E2B/E4B/12B "Unified" (encoder-free) tier.
# The 26B-A4B and 31B dense variants are image/video/text only -- 31B is still
# here as Gemma 4's strongest text model.
GEMMA_31B_MODEL_ID = os.getenv("GEMMA_31B_MODEL_ID", "google/gemma-4-31B-it")
GEMMA_E4B_MODEL_ID = os.getenv("GEMMA_E4B_MODEL_ID", "google/gemma-4-E4B-it")
GEMMA_12B_MODEL_ID = os.getenv("GEMMA_12B_MODEL_ID", "google/gemma-4-12B-it")
QWEN_OMNI_MODEL_ID = os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
MEDGEMMA_4B_MODEL_ID = os.getenv("MEDGEMMA_4B_MODEL_ID", "google/medgemma-1.5-4b-it")
PHI4_MULTIMODAL_MODEL_ID = os.getenv("PHI4_MULTIMODAL_MODEL_ID", "microsoft/Phi-4-multimodal-instruct")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "aya-expanse-8b")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# transformers' device_map for every from_pretrained() call. "cuda" pins the
# model to the single GPU with no CPU offload; "auto" lets accelerate decide,
# which can be unnecessarily conservative on unified-memory hardware (an
# NVIDIA GB10 reports its full ~130GB pool, yet "auto" still offloaded part of
# a 24GB model to CPU). Use "auto" only when a model genuinely doesn't fit.
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
