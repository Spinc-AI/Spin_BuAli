"""HTTP client for speech-to-text.

Talks to this repo's `stt/` service for local models, or to an
OpenAI-compatible `/audio/transcriptions` endpoint for `openai:` models.
Which one a slot uses is decided by its model prefix (see providers.py).
"""
import httpx

import config
import providers
from providers import Provider
from schemas import RuntimeSttSlotConfig, reveal


def _ok(response: httpx.Response, what: str) -> httpx.Response:
    if response.status_code != 200:
        raise RuntimeError(f"{what} failed ({response.status_code}): {response.text}")
    return response


def health() -> bool:
    """True if the local STT service answers. (It has no dedicated health
    route, so the model list doubles as one.)"""
    try:
        with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
            return c.get(f"{config.STT_URL}/models").status_code == 200
    except httpx.HTTPError:
        return False


def list_models() -> dict:
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        return _ok(c.get(f"{config.STT_URL}/models"), "STT model list").json()


def list_languages() -> dict:
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        return _ok(c.get(f"{config.STT_URL}/languages"), "STT language list").json()


def unload() -> None:
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        c.post(f"{config.STT_URL}/models/unload")


def transcribe(audio: bytes, slot: RuntimeSttSlotConfig, default_language: str | None,
               filename: str = "audio.wav") -> str:
    """Run one STT slot and return its transcript.

    The slot's own language wins over the job default. `filename` is passed
    through rather than hardcoded because a cloud endpoint reads the container
    format off the extension -- send an mp3 called "audio.wav" and it is
    rejected, or worse, misdecoded.
    """
    language = slot.language or default_language
    if providers.provider_of(slot.model) is Provider.LOCAL:
        return _transcribe_local(audio, slot.model, language, filename)
    return _transcribe_api(audio, slot, language, filename)


def _transcribe_local(audio: bytes, model: str, language: str | None,
                      filename: str = "audio.wav") -> str:
    # The STT service holds one model at a time, so (re)load right before use:
    # consecutive slots may each want a different one.
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        _ok(c.post(f"{config.STT_URL}/models/{model}/load"), f"STT load of '{model}'")
        response = _ok(
            c.post(f"{config.STT_URL}/transcribe",
                   files={"file": (filename, audio)},
                   data={"language": language} if language else None),
            "STT transcribe",
        )
    return response.json()["text"]


def _transcribe_api(audio: bytes, slot: RuntimeSttSlotConfig, language: str | None,
                    filename: str = "audio.wav") -> str:
    key, base_url = providers.credentials(
        providers.provider_of(slot.model), reveal(slot.api_key), slot.base_url
    )
    data = {"model": providers.bare_model(slot.model) or config.OPENAI_STT_MODEL}
    if language:
        data["language"] = language
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as c:
        response = _ok(
            c.post(f"{base_url}/audio/transcriptions",
                   headers={"Authorization": f"Bearer {key}"},
                   files={"file": (filename, audio)},
                   data=data),
            "cloud STT transcribe",
        )
    return response.json()["text"]
