"""Constants for the demo client."""

# --- Controller connection ---
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9002
TIMEOUT_SHORT = 15    # health checks, model lists, unload
TIMEOUT_RUN = 900     # a run can load a model and transcribe a long recording

# Mirrors the controller's config.MAX_STT_SLOTS -- the slot rows are built at
# startup, before any connection exists to ask.
MAX_STT_SLOTS = 3

# --- Audio ---
AUDIO_FILETYPES = [("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")]
MIC_SAMPLE_RATE = 16000

# --- Model source selection ---
LOCAL_LABEL = "Remote Local Model"
CLOUD_LABEL = "Custom Cloud API"
DEFAULT_CLOUD_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CLOUD_STT_MODEL = "whisper-1"
DEFAULT_CLOUD_LLM_MODEL = "gpt-4o-mini"

# --- Pipelines, as shown in the dropdown ---
PIPELINE_LABELS = {
    "Separate": "separate",
    "Multimodal": "multimodal",
    "Hybrid (STT + Multimodal LLM)": "hybrid",
}

# The versions this client declares on every job. They must be among the
# controller's SUPPORTED_*_VERSIONS or the job is refused -- which is the point:
# a result can always be traced to the implementation that produced it.
PREPROCESSING_VERSION = "legacy-v1"
PIPELINE_VERSION = "buali-v1"
PROMPT_VERSION = "radiology-v1"

GEMINI_HINT = (
    'Multimodal/Hybrid: prefix the cloud model with "gemini:" to use Gemini\'s native audio '
    "API (e.g. gemini:gemini-2.5-pro) instead of the OpenAI-compatible shape."
)
