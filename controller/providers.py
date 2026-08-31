"""Where a model name routes to, and what that provider needs.

A model name carries its provider as a prefix:

    "whisper-large"          -> LOCAL   (this repo's stt/ or core_llm/ service)
    "openai:gpt-4o-mini"     -> OPENAI  (any OpenAI-compatible API)
    "gemini:gemini-2.5-pro"  -> GEMINI  (Gemini's native generateContent API)

Everything that varies by provider lives here: the prefix rules, credential
resolution, and which audio containers each one accepts.
"""
from enum import Enum

import config


class Provider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
    GEMINI = "gemini"


_PREFIXES = {
    config.OPENAI_PREFIX: Provider.OPENAI,
    config.GEMINI_PREFIX: Provider.GEMINI,
}


def provider_of(model: str | None) -> Provider:
    """Which provider `model` selects."""
    for prefix, provider in _PREFIXES.items():
        if model and model.startswith(prefix):
            return provider
    return Provider.LOCAL


def bare_model(model: str | None) -> str:
    """`model` with its provider prefix stripped, as the provider expects it."""
    for prefix in _PREFIXES:
        if model and model.startswith(prefix):
            return model[len(prefix):]
    return model or ""


def is_cloud(model: str | None) -> bool:
    """True if `model` goes to an external provider rather than our own service."""
    return provider_of(model) is not Provider.LOCAL


def credentials(provider: Provider, api_key: str | None = None,
                base_url: str | None = None) -> tuple[str, str]:
    """Resolve (api_key, base_url) for a cloud call.

    A value passed per request wins; otherwise fall back to controller/.env.
    Raises RuntimeError if no key is available from either source.
    """
    if provider is Provider.GEMINI:
        key = api_key or config.GEMINI_API_KEY
        url = base_url or config.GEMINI_BASE_URL
        env_var = "GEMINI_API_KEY"
    else:
        key = api_key or config.OPENAI_API_KEY
        url = base_url or config.OPENAI_BASE_URL
        env_var = "OPENAI_API_KEY"
    if not key:
        raise RuntimeError(
            f"no API key available for this {provider.value} call — pass one in "
            f"POST /session or /run, or set {env_var} in controller/.env"
        )
    return key, url.rstrip("/")


# --- Audio containers ------------------------------------------------------
# Gemini's inline_data accepts this set; our local core_llm/ path is treated
# the same. OpenAI's chat-completions input_audio part takes only wav/mp3.
GEMINI_AUDIO_MIME_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mp3", "aac": "audio/aac",
    "ogg": "audio/ogg", "flac": "audio/flac", "aiff": "audio/aiff",
}

_ACCEPTED_AUDIO = {
    Provider.OPENAI: {"wav", "mp3"},
    Provider.GEMINI: set(GEMINI_AUDIO_MIME_TYPES),
    Provider.LOCAL: set(GEMINI_AUDIO_MIME_TYPES),
}


def audio_format(filename: str, model: str | None) -> str:
    """The uploaded file's container extension, checked against `model`'s provider.

    Raises ValueError if the provider can't accept that container.
    """
    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    accepted = _ACCEPTED_AUDIO[provider_of(model)]
    if ext not in accepted:
        raise ValueError(
            f"this provider only accepts {sorted(accepted)} audio (got "
            f"'{ext or 'unknown'}') — a provider API restriction, not ours"
        )
    return ext
