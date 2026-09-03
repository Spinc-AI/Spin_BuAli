"""HTTP client for the BuAli controller.

Every controller call goes through here, so the UI never touches `requests`
directly and every failure arrives as one exception type carrying the
controller's own message.

The controller is stateless, so there is nothing to start and nothing to keep:
`execute()` sends the configuration, the credentials and the recording together
and gets the whole report back in one call.
"""
import json
import os
import uuid
from typing import Callable

import requests

from config import TIMEOUT_RUN, TIMEOUT_SHORT


class ControllerError(Exception):
    """A controller call failed, carrying the controller's own explanation."""

    def __init__(self, message: str, error_code: str = "", retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


def _detail(exc: requests.RequestException) -> ControllerError:
    """Unpack the controller's structured error body.

    Errors arrive as {error_code, message, retryable}. Older or unexpected
    shapes still produce something readable rather than a traceback.
    """
    if exc.response is None:
        return ControllerError(str(exc))
    try:
        detail = exc.response.json().get("detail", {})
    except ValueError:
        return ControllerError(exc.response.text or str(exc))

    if isinstance(detail, dict) and "message" in detail:
        message = detail["message"]
        if detail.get("retryable"):
            message += "  (worth retrying)"
        return ControllerError(message, detail.get("error_code", ""),
                               bool(detail.get("retryable")))
    return ControllerError(str(detail) or str(exc))


class BuAliClient:
    """Calls the controller at whatever `base_url()` currently returns.

    `token()` supplies the internal API token; every route but the health check
    requires it.
    """

    def __init__(self, base_url: Callable[[], str], token: Callable[[], str] = lambda: ""):
        self._base_url = base_url
        self._token = token

    def _headers(self) -> dict:
        return {"X-Internal-Token": self._token()}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = requests.request(method, self._base_url() + path,
                                        headers=self._headers(), **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise _detail(exc) from exc
        return response.json()

    def is_reachable(self) -> bool:
        """Health needs no token, so this works before one is entered."""
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

    def execute(self, audio_path: str, processing_config: dict,
                credentials: dict | None = None, job_id: str | None = None) -> dict:
        """Run one recording through one pipeline.

        The job id is invented here because this is a demo client with no
        backend behind it. In production the backend owns the job and passes
        its own id; the controller only echoes it back.
        """
        job_id = job_id or f"demo-{uuid.uuid4().hex[:12]}"
        data = {"config_json": json.dumps(processing_config, ensure_ascii=False)}
        if credentials:
            data["credentials_json"] = json.dumps(credentials, ensure_ascii=False)
        with open(audio_path, "rb") as audio:
            return self._request(
                "POST", f"/internal/jobs/{job_id}/execute",
                files={"file": (os.path.basename(audio_path), audio)},
                data=data, timeout=TIMEOUT_RUN)

    def unload_models(self, llm_model: str | None = None) -> dict:
        """Free the local services' weights. An admin action, not a session."""
        return self._request("POST", "/internal/admin/models/unload",
                             data={"llm_model": llm_model} if llm_model else None,
                             timeout=TIMEOUT_SHORT)
