"""The controller client.

Two things matter here and neither involves a window: the request carries the
token and keeps configuration separate from credentials, and the controller's
structured errors arrive as something a person can read.
"""
import json

import pytest
import requests

from api import BuAliClient, ControllerError


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


@pytest.fixture
def sent(monkeypatch):
    """Capture the request instead of making it."""
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return captured.get("_reply") or FakeResponse(body={"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    return captured


@pytest.fixture
def client():
    return BuAliClient(lambda: "http://localhost:9002", lambda: "the-token")


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"fake-audio")
    return str(path)


CONFIG = {"pipeline": "separate", "llm_model": "whisper",
          "stt_slots": [{"slot_id": "stt_1", "model": "whisper"}],
          "preprocessing_version": "legacy-v1", "pipeline_version": "buali-v1",
          "prompt_version": "radiology-v1"}


class TestRequestShape:
    def test_the_token_goes_on_every_call(self, client, sent):
        client.stt_models()
        assert sent["headers"]["X-Internal-Token"] == "the-token"

    def test_a_job_posts_to_the_execute_route(self, client, sent, audio):
        client.execute(audio, CONFIG, job_id="job-7")
        assert sent["method"] == "POST"
        assert sent["url"].endswith("/internal/jobs/job-7/execute")

    def test_configuration_and_credentials_travel_separately(self, client, sent, audio):
        """The whole point of the split: one half is storable, the other is not."""
        client.execute(audio, CONFIG, {"llm": {"api_key": "sk-secret"}})
        assert json.loads(sent["data"]["config_json"]) == CONFIG
        assert "sk-secret" not in sent["data"]["config_json"]
        assert json.loads(sent["data"]["credentials_json"])["llm"]["api_key"] == "sk-secret"

    def test_credentials_are_omitted_when_there_are_none(self, client, sent, audio):
        client.execute(audio, CONFIG)
        assert "credentials_json" not in sent["data"]

    def test_the_real_filename_is_sent(self, client, sent, audio):
        """A cloud endpoint reads the container format off the extension."""
        assert sent is not None
        client.execute(audio, CONFIG)
        assert sent["files"]["file"][0] == "recording.mp3"

    def test_a_job_id_is_invented_when_none_is_given(self, client, sent, audio):
        """The demo has no backend to own one; production passes its own."""
        client.execute(audio, CONFIG)
        assert "/internal/jobs/demo-" in sent["url"]

    def test_health_is_checked_without_a_token(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(requests, "get",
                            lambda url, **kw: seen.update(url=url, kw=kw) or FakeResponse())
        assert client.is_reachable() is True
        assert "headers" not in seen["kw"], "health must work before a token is entered"


class TestErrors:
    def _fails_with(self, monkeypatch, status, body):
        def fake_request(method, url, **kwargs):
            return FakeResponse(status, body)
        monkeypatch.setattr(requests, "request", fake_request)

    def test_a_structured_error_becomes_a_readable_message(self, client, monkeypatch):
        self._fails_with(monkeypatch, 422, {"detail": {
            "error_code": "UNSUPPORTED_PROMPT_VERSION",
            "message": "Unsupported version 'radiology-v9'.", "retryable": False}})
        with pytest.raises(ControllerError) as caught:
            client.stt_models()
        assert caught.value.error_code == "UNSUPPORTED_PROMPT_VERSION"
        assert "radiology-v9" in str(caught.value)

    def test_a_retryable_error_says_so(self, client, monkeypatch):
        """The one thing a user needs from an error: is it worth trying again."""
        self._fails_with(monkeypatch, 503, {"detail": {
            "error_code": "GPU_OUT_OF_MEMORY", "message": "The GPU is full.",
            "retryable": True}})
        with pytest.raises(ControllerError) as caught:
            client.stt_models()
        assert caught.value.retryable is True
        assert "retrying" in str(caught.value)

    def test_a_non_json_error_still_reads_sensibly(self, client, monkeypatch):
        self._fails_with(monkeypatch, 502, None)
        monkeypatch.setattr(requests, "request",
                            lambda *a, **k: FakeResponse(502, None, "Bad Gateway"))
        with pytest.raises(ControllerError, match="Bad Gateway"):
            client.stt_models()

    def test_an_unreachable_controller_is_not_a_crash(self, client, monkeypatch):
        def refuse(*args, **kwargs):
            raise requests.ConnectionError("connection refused")
        monkeypatch.setattr(requests, "request", refuse)
        with pytest.raises(ControllerError):
            client.stt_models()
