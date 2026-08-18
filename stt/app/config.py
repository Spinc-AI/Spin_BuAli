'''Central configuration for the STT service.'''

import torch


# ============================================================
# Runtime
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOST = "0.0.0.0"
PORT = 8000

ALLOWED_ORIGINS = ["*"]


# ============================================================
# Audio
# ============================================================

TARGET_SAMPLE_RATE = 16000


# ============================================================
# Languages
# ============================================================

# Codes accepted by POST /transcribe's `language` field, mapped to a display
# name. Each model class maps these onto whatever codes it actually needs
# (see WHISPER_LANGUAGE_NAMES / SEAMLESS_LANGUAGE_CODES in model.py).
SUPPORTED_LANGUAGES = {
    "fa": "Persian",
    "en": "English",
}
DEFAULT_LANGUAGE = "fa"


# ============================================================
# Models
# ============================================================

PRELOAD_MODEL = None

# Ordered best -> worst by clinic-realistic-noise WER, per
# ../benchmark/benchmark_summary.pdf ("Results - clean vs. clinic-realistic").
# This order is what the UI dropdowns show, so don't reorder without also
# reordering the benchmark's ranking.
MODEL_REGISTRY = {
    "seamless": {
        "type": "seamless_v2",
        "model_id": "facebook/seamless-m4t-v2-large",
        "tgt_lang": "pes",
    },
    "whisper": {
        "type": "whisper",
        "model_id": "nezamisafa/whisper-persian-v4",
    },
    "seamless-medium": {
        "type": "seamless_v1",
        "model_id": "facebook/hf-seamless-m4t-medium",
        "tgt_lang": "pes",
    },
    "whisper-halakoo": {
        "type": "whisper",
        "model_id": "MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch",
    },
    "mms-all": {
        "type": "mms",
        "model_id": "facebook/mms-1b-all",
        "target_lang": "fas",
    },
    "mms-fl102": {
        "type": "mms",
        "model_id": "facebook/mms-1b-fl102",
        "target_lang": "fas",
    },
    "whisper-vhdm": {
        "type": "whisper",
        "model_id": "vhdm/whisper-large-fa-v1",
    },
    "wav2vec2-xlsr53": {
        "type": "ctc",
        "model_id": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    },
    "whisper-large-v3": {
        "type": "whisper",
        "model_id": "openai/whisper-large-v3",
    },
    "whisper-large-v3-turbo": {
        "type": "whisper",
        "model_id": "openai/whisper-large-v3-turbo",
    },
}
