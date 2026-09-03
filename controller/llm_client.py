"""HTTP client for the language model.

One entry point, `complete()`, sends a single-turn prompt (optionally with an
audio attachment) to whichever provider the model name selects, and returns
the raw reply text. Every call site here wants JSON back, so JSON mode is
requested wherever the provider supports it; `prompts.extract_json()` parses the reply
tolerantly for the providers that don't.
"""
import base64
import json

import httpx

import config
import providers
from providers import Provider


def _ok(response: httpx.Response, what: str) -> httpx.Response:
    if response.status_code != 200:
        raise RuntimeError(f"{what} failed ({response.status_code}): {response.text}")
    return response


def health() -> bool:
    try:
        with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
            return c.get(f"{config.LLM_URL}/").status_code == 200
    except httpx.HTTPError:
        return False


def list_models() -> dict:
    """The local LLM service's registry, plus which of those accept audio."""
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        every = _ok(c.get(f"{config.LLM_URL}/models"), "LLM model list").json()
        audio = _ok(c.get(f"{config.LLM_URL}/chat_audio/models"), "LLM audio model list").json()
    return {
        "available": every.get("available", []),
        "audio_capable": audio.get("available", []),
        "loaded": every.get("loaded"),
    }


def unload(model: str | None = None) -> None:
    """Free the local service's VRAM. No-op for cloud models."""
    if providers.is_cloud(model):
        return
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        c.post(f"{config.LLM_URL}/unload")


def complete(system_prompt: str, user_text: str | None, model: str,
             api_key: str | None = None, base_url: str | None = None,
             audio: bytes | None = None, audio_format: str | None = None) -> str:
    """Send one system+user turn (optionally with audio) and return the reply.

    Routes to the provider that `model` selects. Passing `audio` requires a
    model that can accept it -- validate the container with
    providers.audio_format() first.
    """
    provider = providers.provider_of(model)
    name = providers.bare_model(model)
    if provider is Provider.GEMINI:
        return _gemini(system_prompt, user_text, name, api_key, base_url, audio, audio_format)
    if provider is Provider.OPENAI:
        return _openai(system_prompt, user_text, name, api_key, base_url, audio, audio_format)
    return _local(system_prompt, user_text, name, audio, audio_format)


def _local(system_prompt: str, user_text: str | None, model: str,
           audio: bytes | None, audio_format: str | None) -> str:
    """This repo's core_llm/ service. It has no JSON mode -- the prompt asks
    for JSON and prompts.extract_json() copes with whatever comes back."""
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        if audio is None:
            response = _ok(
                c.post(f"{config.LLM_URL}/chat",
                       json={"model": model,
                             "messages": [{"role": "system", "content": system_prompt},
                                          {"role": "user", "content": user_text or ""}]}),
                "LLM chat",
            )
        else:
            response = _ok(
                c.post(f"{config.LLM_URL}/chat_audio",
                       files={"file": (f"audio.{audio_format}", audio)},
                       data={"model": model, "system_prompt": system_prompt,
                             "text": user_text or ""}),
                "local multimodal LLM chat",
            )
    return response.json()["reply"]


def _openai(system_prompt: str, user_text: str | None, model: str,
            api_key: str | None, base_url: str | None,
            audio: bytes | None, audio_format: str | None) -> str:
    """Any OpenAI-compatible /chat/completions endpoint."""
    key, url = providers.credentials(Provider.OPENAI, api_key, base_url)
    if audio is None:
        user_content: str | list[dict] = user_text or ""
    else:
        user_content = [{
            "type": "input_audio",
            "input_audio": {"data": _b64(audio), "format": audio_format},
        }]
        if user_text:
            user_content.append({"type": "text", "text": user_text})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}],
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        response = _ok(
            c.post(f"{url}/chat/completions",
                   headers={"Authorization": f"Bearer {key}"}, json=payload),
            "cloud LLM chat",
        )
    return response.json()["choices"][0]["message"]["content"]


def _gemini(system_prompt: str, user_text: str | None, model: str,
            api_key: str | None, base_url: str | None,
            audio: bytes | None, audio_format: str | None) -> str:
    """Gemini's native generateContent shape (not the OpenAI-compatible one)."""
    key, url = providers.credentials(Provider.GEMINI, api_key, base_url)
    # The instructions go in BOTH system_instruction and the user turn's text
    # part -- some proxies drop system_instruction silently.
    parts: list[dict] = [
        {"text": system_prompt + (f"\n\n{user_text}" if user_text else "")}
    ]
    if audio is not None:
        mime = providers.GEMINI_AUDIO_MIME_TYPES.get(audio_format, f"audio/{audio_format}")
        parts.append({"inline_data": {"mime_type": mime, "data": _b64(audio)}})
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        response = _ok(
            c.post(f"{url}/models/{model}:generateContent",
                   headers={"x-goog-api-key": key}, json=payload),
            "Gemini LLM call",
        )
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected Gemini response shape: {data}")


def _b64(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")
