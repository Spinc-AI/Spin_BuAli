"""Central configuration, loaded from environment / .env file.

Keeping this in one place means the rest of the code never hard-codes the
server address or model name — handy when you point at different servers.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads a .env file if present; no-op otherwise

# --- HTTP server (the FastAPI wrapper in main.py) ---
# Where this service listens. The orchestrator and other modules call it here.
# Port 8001 keeps it clear of the STT service (which uses 8000).
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Local models — served directly via transformers, NOT Ollama. ---
# Ollama can't accept audio input at all, so keeping it would've meant two
# separate serving paths side by side even though several of these models can
# do both text and audio; this is the one unified path instead (see model.py
# for the registry/manager). Trade-off: no automatic quantization like Ollama
# gave you, so these load at full precision -- higher VRAM, simpler code.
# Lazy-loaded on first request for a given model, not at startup; only one is
# held in memory at a time (see model.LLMManager).
AYA_8B_MODEL_ID = os.getenv("AYA_8B_MODEL_ID", "CohereLabs/aya-expanse-8b")
AYA_32B_MODEL_ID = os.getenv("AYA_32B_MODEL_ID", "CohereLabs/aya-expanse-32b")
# Gemma 4's audio support only exists on E2B/E4B/12B ("Unified", encoder-free)
# -- NOT the 26B-A4B or 31B dense variants, which are image/video/text only,
# no audio input. 31B is still worth having as Gemma 4's strongest *text*
# model even though it can't do the audio role the E4B/12B entries can.
GEMMA_31B_MODEL_ID = os.getenv("GEMMA_31B_MODEL_ID", "google/gemma-4-31B-it")
GEMMA_E4B_MODEL_ID = os.getenv("GEMMA_E4B_MODEL_ID", "google/gemma-4-E4B-it")
GEMMA_12B_MODEL_ID = os.getenv("GEMMA_12B_MODEL_ID", "google/gemma-4-12B-it")  # largest audio-capable Gemma 4
QWEN_OMNI_MODEL_ID = os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "aya-expanse-8b")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# transformers' device_map passed to every from_pretrained() call. "cuda" pins
# everything to the single GPU (no automatic CPU offload); "auto" lets
# accelerate decide, which can be unnecessarily conservative on unified-memory
# hardware (e.g. NVIDIA GB10 -- CUDA correctly reports the full ~130GB pool,
# but "auto" still partially offloaded a 24GB model to CPU there). Override to
# "auto" if you're on a server where the model genuinely doesn't fit in one
# GPU's VRAM and offload is actually needed.
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
