"""Test fixtures.

The controller is a pure HTTP client of the stt/ and core_llm/ services, so
every test here stubs those two out -- nothing loads a model or touches the
network.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import evaluation_client  # noqa: E402
import llm_client  # noqa: E402
import stt_client  # noqa: E402


@pytest.fixture
def services_up(monkeypatch):
    """Sibling services reachable, with recording stubs for their calls.

    Every outbound client is stubbed, so no test ever opens a socket.
    """
    calls = {"transcribe": [], "complete": [], "unloaded": []}

    def fake_transcribe(audio, slot, language):
        calls["transcribe"].append((slot.model, language))
        return f"transcript from {slot.model}"

    def fake_complete(system_prompt, user_text, model, api_key=None, base_url=None,
                      audio=None, audio_format=None):
        calls["complete"].append({
            "system_prompt": system_prompt, "user_text": user_text, "model": model,
            "api_key": api_key, "base_url": base_url,
            "audio": audio, "audio_format": audio_format,
        })
        return '{"final_text": "report"}'

    monkeypatch.setattr(stt_client, "health", lambda: True)
    monkeypatch.setattr(evaluation_client, "health", lambda: True)
    monkeypatch.setattr(llm_client, "health", lambda: True)
    monkeypatch.setattr(stt_client, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(stt_client, "unload", lambda: calls["unloaded"].append("stt"))
    monkeypatch.setattr(llm_client, "unload", lambda model=None: calls["unloaded"].append("llm"))
    return calls


@pytest.fixture
def client(services_up):
    """A TestClient with a fresh, empty session."""
    from fastapi.testclient import TestClient

    import main

    main._session = None
    with TestClient(main.app) as test_client:
        yield test_client
    main._session = None
