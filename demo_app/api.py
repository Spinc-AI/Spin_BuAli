"""HTTP client for the BuAli controller.

Every controller call goes through here, so the UI never touches `requests`
directly and every failure arrives as one exception type carrying the
controller's own `detail` message.
"""
import os
from typing import Callable

import requests

from config import TIMEOUT_RUN, TIMEOUT_SHORT


class ControllerError(Exception):
    """A controller call failed. The message is the controller's own detail."""


def _detail(exc: requests.RequestException) -> str:
    """The `detail` FastAPI puts in an error body, or the next best thing."""
    if exc.response is None:
        return str(exc)
    try:
        body = exc.response.json()
    except ValueError:
        return exc.response.text or str(exc)
    return body.get("detail", str(body))


class BuAliClient:
    """Calls the controller at whatever address `base_url()` currently returns."""

    def __init__(self, base_url: Callable[[], str]):
        self._base_url = base_url

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = requests.request(method, self._base_url() + path, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ControllerError(_detail(exc)) from exc
        return response.json()

    def is_reachable(self) -> bool:
        try:
            return requests.get(self._base_url() + "/", timeout=TIMEOUT_SHORT).status_code == 200
        except requests.RequestException:
            return False

    def stt_models(self) -> list[str]:
        return self._request("GET", "/models", timeout=TIMEOUT_SHORT).get("available", [])

    def llm_models(self) -> tuple[list[str], list[str]]:
        """(every local LLM, just the audio-capable ones)."""
        body = self._request("GET", "/llm/models", timeout=TIMEOUT_SHORT)
        return body.get("available", []), body.get("audio_capable", [])

    def start_session(self, payload: dict) -> dict:
        return self._request("POST", "/session", json=payload, timeout=TIMEOUT_RUN)

    def unload_session(self) -> dict:
        return self._request("POST", "/session/unload", timeout=TIMEOUT_SHORT)

    def run(self, audio_path: str, overrides: dict | None = None) -> dict:
        with open(audio_path, "rb") as audio:
            return self._request(
                "POST", "/run",
                files={"file": (os.path.basename(audio_path), audio)},
                data=overrides or None,
                timeout=TIMEOUT_RUN,
            )
