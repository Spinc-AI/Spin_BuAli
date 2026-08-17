"""Thin HTTP clients for STT and Core_LLM (from Spin_Medical_Assistant_Project),
plus local/cloud dispatch helpers. Ported unchanged from Orchestrator's
orchestrator.py -- same behavior, just living in its own module here.
"""
from __future__ import annotations

import base64
import pathlib
from typing import Any, Optional

import httpx
from fastapi import HTTPException

import config

# ---------------------------------------------------------------------------
# Local <-> API dispatch helpers
# ---------------------------------------------------------------------------
def is_api_model(model: Optional[str]) -> bool:
    """True if `model` selects the external OpenAI-compatible API."""
    return bool(model) and model.startswith(config.API_PREFIX)


def strip_api_prefix(model: Optional[str]) -> str:
    return model[len(config.API_PREFIX):] if model else ""


def is_gemini_model(model: Optional[str]) -> bool:
    """True if `model` selects Gemini's own (non-OpenAI-shaped) API directly."""
    return bool(model) and model.startswith(config.GEMINI_PREFIX)


def strip_gemini_prefix(model: Optional[str]) -> str:
    return model[len(config.GEMINI_PREFIX):] if model else ""


def is_cloud_model(model: Optional[str]) -> bool:
    """True if `model` routes to any external provider (OpenAI-shaped or Gemini),
    as opposed to a local model."""
    return is_api_model(model) or is_gemini_model(model)


# ---------------------------------------------------------------------------
# Module clients
# ---------------------------------------------------------------------------
def _client() -> httpx.Client:
    return httpx.Client(timeout=config.HTTP_TIMEOUT)


def _resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No API key available for this external call — pass it in POST /session or /run, "
            "or set OPENAI_API_KEY in the controller's .env"
        )
    return key


def _resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or config.OPENAI_BASE_URL).rstrip("/")


def stt_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{config.STT_URL}/models").status_code == 200
    except httpx.HTTPError:
        return False


def stt_models() -> dict:
    with _client() as c:
        r = c.get(f"{config.STT_URL}/models")
    r.raise_for_status()
    return r.json()


def stt_load(model: str) -> None:
    with _client() as c:
        r = c.post(f"{config.STT_URL}/models/{model}/load")
    if r.status_code != 200:
        raise RuntimeError(f"STT load failed ({r.status_code}): {r.text}")


def stt_transcribe(audio: bytes, filename: str = "audio.wav", language: Optional[str] = None) -> str:
    """Transcribe with our own (local) STT service. Caller must have loaded the model already."""
    data = {"language": language} if language else None
    with _client() as c:
        r = c.post(f"{config.STT_URL}/transcribe", files={"file": (filename, audio)}, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"STT transcribe failed ({r.status_code}): {r.text}")
    return r.json()["text"]


def stt_api_transcribe(audio: bytes, filename: str = "audio.wav", language: Optional[str] = None,
                        api_key: Optional[str] = None, base_url: Optional[str] = None,
                        model: Optional[str] = None) -> str:
    """Transcribe with an external STT API (OpenAI-compatible /audio/transcriptions)."""
    key = _resolve_api_key(api_key)
    url = _resolve_base_url(base_url)
    data = {"model": model or config.OPENAI_STT_MODEL}
    if language:
        data["language"] = language
    with _client() as c:
        r = c.post(
            f"{url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename, audio)},
            data=data,
        )
    if r.status_code != 200:
        raise RuntimeError(f"API STT failed ({r.status_code}): {r.text}")
    return r.json()["text"]


def stt_languages() -> dict:
    with _client() as c:
        r = c.get(f"{config.STT_URL}/languages")
    r.raise_for_status()
    return r.json()


def stt_unload() -> None:
    with _client() as c:
        c.post(f"{config.STT_URL}/models/unload")


def llm_health() -> bool:
    try:
        with _client() as c:
            return c.get(f"{config.LLM_URL}/").status_code == 200
    except httpx.HTTPError:
        return False


def llm_chat(messages: list[dict], model: Optional[str] = None,
             response_format: Optional[dict] = None) -> str:
    """Chat with our own (local) Core_LLM service."""
    payload: dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(f"{config.LLM_URL}/chat", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["reply"]


def llm_api_chat(messages: list[dict], model: str, response_format: Optional[dict] = None,
                  api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with an external LLM API (OpenAI-compatible /chat/completions)."""
    key = _resolve_api_key(api_key)
    url = _resolve_base_url(base_url)
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if response_format:
        payload["response_format"] = response_format
    with _client() as c:
        r = c.post(
            f"{url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
    if r.status_code != 200:
        raise RuntimeError(f"API LLM chat failed ({r.status_code}): {r.text}")
    return r.json()["choices"][0]["message"]["content"]


# OpenAI's chat-completions audio input (input_audio content-part) only
# accepts these two container formats. Gemini's generateContent (inline_data)
# accepts a wider set. The local path (Core_LLM's /chat_audio) is treated the
# same as Gemini's set (unverified exact boundary).
OPENAI_AUDIO_FORMATS = {"wav", "mp3"}
GEMINI_AUDIO_MIME_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mp3", "aac": "audio/aac",
    "ogg": "audio/ogg", "flac": "audio/flac", "aiff": "audio/aiff",
}
LOCAL_AUDIO_FORMATS = set(GEMINI_AUDIO_MIME_TYPES)


def audio_format_from_filename(filename: str, model: Optional[str] = None) -> str:
    """Validate the uploaded audio's container format against whichever
    provider `model` selects, and return its extension (lowercase, no dot)."""
    ext = pathlib.Path(filename or "").suffix.lstrip(".").lower()
    if is_gemini_model(model):
        allowed = set(GEMINI_AUDIO_MIME_TYPES)
    elif is_api_model(model):
        allowed = OPENAI_AUDIO_FORMATS
    else:
        allowed = LOCAL_AUDIO_FORMATS
    if ext not in allowed:
        raise HTTPException(
            400,
            f"multimodal LLM mode only accepts {sorted(allowed)} audio for this provider "
            f"(got '{ext or 'unknown'}') — this is a provider API restriction, not ours.",
        )
    return ext


def llm_api_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                        model: str, response_format: Optional[dict] = None,
                        api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with an OpenAI-compatible external LLM API, feeding it audio directly (no STT step)."""
    content: list[dict] = [{
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": audio_format},
    }]
    if user_text:
        content.append({"type": "text", "text": user_text})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return llm_api_chat(messages, model, response_format=response_format, api_key=api_key, base_url=base_url)


def _resolve_gemini_api_key(api_key: Optional[str]) -> str:
    key = api_key or config.GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "No Gemini API key available for this call — pass it in POST /session or /run "
            "(llm_api_key), or set GEMINI_API_KEY in the controller's .env"
        )
    return key


def _resolve_gemini_base_url(base_url: Optional[str]) -> str:
    return (base_url or config.GEMINI_BASE_URL).rstrip("/")


def llm_gemini_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                          model: str, json_response: bool = True,
                          api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Chat with Gemini's own generateContent API, feeding it audio directly.

    Uses Gemini's native request shape (contents/parts/inline_data +
    systemInstruction), not the OpenAI-compatible shape. The instructions go
    in BOTH system_instruction AND the first text part of the user turn --
    some proxies drop system_instruction silently.
    """
    key = _resolve_gemini_api_key(api_key)
    url = _resolve_gemini_base_url(base_url)
    mime_type = GEMINI_AUDIO_MIME_TYPES.get(audio_format, f"audio/{audio_format}")
    instructions = system_prompt + (f"\n\n{user_text}" if user_text else "")
    parts: list[dict] = [
        {"text": instructions},
        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio).decode("ascii")}},
    ]
    payload: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
    }
    if json_response:
        payload["generationConfig"] = {"response_mime_type": "application/json"}
    with _client() as c:
        r = c.post(f"{url}/models/{model}:generateContent",
                   headers={"x-goog-api-key": key}, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini LLM call failed ({r.status_code}): {r.text}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected Gemini response shape: {data}")


def llm_local_chat_audio(audio: bytes, audio_format: str, system_prompt: str,
                         user_text: Optional[str], model: Optional[str] = None,
                         filename: str = "audio.wav") -> str:
    """Chat with our own (local) Core_LLM service's audio-capable model
    (Core_LLM's /chat_audio endpoint, served via transformers)."""
    data = {"system_prompt": system_prompt, "text": user_text or ""}
    if model:
        data["model"] = model
    with _client() as c:
        r = c.post(f"{config.LLM_URL}/chat_audio", files={"file": (filename, audio)}, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"Local multimodal LLM call failed ({r.status_code}): {r.text}")
    return r.json()["reply"]


def llm_chat_audio(audio: bytes, audio_format: str, system_prompt: str, user_text: Optional[str],
                   model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Dispatch an audio-input LLM call to whichever provider `model` selects:
    "openai:..." (OpenAI-shaped), "gemini:..." (Gemini's own shape), or
    anything else -> our local Core_LLM service's audio-capable model."""
    if is_gemini_model(model):
        return llm_gemini_chat_audio(audio, audio_format, system_prompt, user_text,
                                     strip_gemini_prefix(model), api_key=api_key, base_url=base_url)
    if is_api_model(model):
        return llm_api_chat_audio(audio, audio_format, system_prompt, user_text,
                                  strip_api_prefix(model), response_format={"type": "json_object"},
                                  api_key=api_key, base_url=base_url)
    return llm_local_chat_audio(audio, audio_format, system_prompt, user_text,
                                model=model, filename=f"audio.{audio_format}")


def chat(messages: list[dict], model: Optional[str], response_format: Optional[dict] = None,
         api_key: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Dispatch to the local Core_LLM or an external API, based on `model`."""
    if is_api_model(model):
        return llm_api_chat(messages, strip_api_prefix(model), response_format=response_format,
                            api_key=api_key, base_url=base_url)
    return llm_chat(messages, model=model, response_format=response_format)


def llm_unload(model: Optional[str] = None) -> None:
    if is_api_model(model):
        return  # external API — nothing local to unload
    params = {"model": model} if model else {}
    with _client() as c:
        c.post(f"{config.LLM_URL}/unload", params=params)


# ---------------------------------------------------------------------------
# STT slots — up to MAX_STT_SLOTS independently-configurable STT engines
# ---------------------------------------------------------------------------
def transcribe_slot(audio: bytes, slot, default_language: Optional[str]) -> str:
    """Run one STT slot: loads+calls the local service, or calls the external API.
    `slot` is a schemas.SttSlotConfig."""
    language = slot.language or default_language
    if is_api_model(slot.model):
        return stt_api_transcribe(audio, language=language, api_key=slot.api_key,
                                  base_url=slot.base_url, model=strip_api_prefix(slot.model) or None)
    stt_load(slot.model)  # STT holds one model at a time — (re)load right before use
    return stt_transcribe(audio, language=language)


def extract_json(text: str):
    """Pull a JSON object out of the LLM reply (tolerant of code fences / prose)."""
    import json
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM reply")
    return json.loads(text[start:end + 1])
