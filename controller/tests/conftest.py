"""Test fixtures.

The controller is a pure HTTP client of the stt/ and core_llm/ services, so
every test here stubs those two out -- nothing loads a model or touches the
network.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import evaluation_client  # noqa: E402
import llm_client  # noqa: E402
import stt_client  # noqa: E402

TOKEN = "test-internal-token"

# A minimally valid job configuration. Tests override the parts they are about.
BASE_CONFIG = {
    "pipeline": "separate",
    "language": "fa",
    "stt_slots": [{"slot_id": "stt_1", "model": "whisper"}],
    "llm_model": "gemini:gemini-2.5-pro",
    "preprocessing_version": "legacy-v1",
    "pipeline_version": "buali-v1",
    "prompt_version": "radiology-v1",
}


@pytest.fixture
def services_up(monkeypatch):
    """Sibling services reachable, with recording stubs for their calls.

    Every outbound client is stubbed, so no test ever opens a socket.
    """
    calls = {"transcribe": [], "complete": [], "unloaded": []}

    def fake_transcribe(audio, slot, language, filename="audio.wav"):
        calls["transcribe"].append((slot.model, language, filename))
        return f"transcript from {slot.model}"

    def fake_complete(system_prompt, user_text, model, api_key=None, base_url=None,
                      audio=None, audio_format=None):
        calls["complete"].append({
            "system_prompt": system_prompt, "user_text": user_text, "model": model,
            "api_key": api_key, "base_url": base_url,
            "audio": audio, "audio_format": audio_format,
        })
        return json.dumps({"raw_transcript": "raw", "corrected_transcript": "corrected",
                           "final_text": "report"})

    monkeypatch.setattr(stt_client, "health", lambda: True)
    monkeypatch.setattr(evaluation_client, "health", lambda: True)
    monkeypatch.setattr(llm_client, "health", lambda: True)
    monkeypatch.setattr(stt_client, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(stt_client, "unload", lambda: calls["unloaded"].append("stt"))
    monkeypatch.setattr(llm_client, "unload", lambda model=None: calls["unloaded"].append("llm"))
    return calls


@pytest.fixture
def token(monkeypatch):
    """A configured internal token, since every route but / requires one."""
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", TOKEN)
    return TOKEN


@pytest.fixture
def client(services_up, token):
    """A TestClient that sends the internal token on every request."""
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app, headers={"X-Internal-Token": token}) as test_client:
        yield test_client


@pytest.fixture
def run_job(client):
    """POST a job, with `overrides` merged into the base configuration."""
    def execute(overrides=None, credentials=None, job_id="job-1",
                filename="recording.wav", audio=b"fake-audio-bytes"):
        payload = {**BASE_CONFIG, **(overrides or {})}
        data = {"config_json": json.dumps(payload)}
        if credentials is not None:
            data["credentials_json"] = json.dumps(credentials)
        return client.post(f"/internal/jobs/{job_id}/execute",
                           files={"file": (filename, audio)}, data=data)
    return execute
