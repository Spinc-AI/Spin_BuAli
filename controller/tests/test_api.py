"""The HTTP surface: session lifecycle, validation, and secret handling."""
import json

import pytest

import llm_client
import stt_client


def start(client, **overrides):
    body = {"pipeline": "separate", "llm_model": "aya-expanse-8b",
            "stt_slots": [{"model": "whisper"}]}
    body.update(overrides)
    return client.post("/session", json=body)


class TestSessionLifecycle:
    def test_no_session_until_one_is_started(self, client):
        assert client.get("/status").json() == {"active": False}

    def test_status_reports_the_started_session(self, client):
        assert start(client).status_code == 200
        body = client.get("/status").json()
        assert body["active"] is True
        assert body["pipeline"] == "separate"
        assert body["llm_model"] == "aya-expanse-8b"

    def test_unload_clears_the_session_and_frees_both_services(self, client, services_up):
        start(client)
        assert client.post("/session/unload").json() == {"active": False}
        assert client.get("/status").json() == {"active": False}
        assert set(services_up["unloaded"]) == {"stt", "llm"}

    def test_unload_survives_a_service_being_down(self, client, monkeypatch):
        start(client)

        def boom(*args, **kwargs):
            raise RuntimeError("service is down")

        monkeypatch.setattr(stt_client, "unload", boom)
        monkeypatch.setattr(llm_client, "unload", boom)
        assert client.post("/session/unload").status_code == 200

    def test_running_without_a_session_is_rejected(self, client):
        response = client.post("/run", files={"file": ("r.wav", b"audio")})
        assert response.status_code == 409


class TestSessionValidation:
    def test_unknown_pipeline_is_rejected(self, client):
        assert start(client, pipeline="telepathy").status_code == 422

    def test_more_slots_than_allowed_is_rejected(self, client):
        response = start(client, stt_slots=[{"model": "whisper"}] * 4)
        assert response.status_code == 422

    @pytest.mark.parametrize("pipeline", ["separate", "hybrid"])
    def test_stt_pipelines_need_a_slot(self, client, pipeline):
        response = start(client, pipeline=pipeline, stt_slots=[None, None])
        assert response.status_code == 400
        assert "STT slot" in response.json()["detail"]

    def test_multimodal_needs_no_slots(self, client):
        assert start(client, pipeline="multimodal", stt_slots=None).status_code == 200

    def test_unreachable_stt_blocks_a_local_slot(self, client, monkeypatch):
        monkeypatch.setattr(stt_client, "health", lambda: False)
        assert start(client).status_code == 503

    def test_unreachable_stt_is_fine_for_cloud_slots(self, client, monkeypatch):
        monkeypatch.setattr(stt_client, "health", lambda: False)
        response = start(client, stt_slots=[{"model": "openai:whisper-1"}],
                         llm_model="openai:gpt-4o")
        assert response.status_code == 200

    def test_unreachable_llm_blocks_a_local_model(self, client, monkeypatch):
        monkeypatch.setattr(llm_client, "health", lambda: False)
        assert start(client).status_code == 503

    @pytest.mark.parametrize("model", ["openai:gpt-4o", "gemini:gemini-2.5-pro"])
    def test_cloud_llm_does_not_need_the_local_service(self, client, monkeypatch, model):
        """A gemini: model in `separate` used to demand a local LLM and then
        fail at run time -- both cloud prefixes must bypass the health check."""
        monkeypatch.setattr(llm_client, "health", lambda: False)
        assert start(client, llm_model=model).status_code == 200


class TestSecrets:
    def test_api_keys_never_appear_in_status(self, client):
        start(client, llm_api_key="sk-super-secret",
              stt_slots=[{"model": "openai:whisper-1", "api_key": "sk-slot-secret"}])
        body = json.dumps(client.get("/status").json())
        assert "sk-super-secret" not in body
        assert "sk-slot-secret" not in body

    def test_session_keys_still_reach_the_llm(self, client, services_up):
        start(client, llm_api_key="sk-session", llm_model="openai:gpt-4o")
        client.post("/run", files={"file": ("r.wav", b"audio")})
        assert services_up["complete"][0]["api_key"] == "sk-session"


class TestRun:
    def test_returns_the_pipeline_and_result(self, client):
        start(client)
        body = client.post("/run", files={"file": ("r.wav", b"audio")}).json()
        assert body["pipeline"] == "separate"
        assert body["result"]["final_text"] == "report"

    def test_per_run_credentials_override_the_session(self, client, services_up):
        start(client, llm_api_key="sk-session", llm_model="openai:gpt-4o")
        client.post("/run", files={"file": ("r.wav", b"audio")},
                    data={"llm_api_key": "sk-per-run"})
        assert services_up["complete"][0]["api_key"] == "sk-per-run"

    def test_per_run_language_overrides_the_session(self, client, services_up):
        start(client, language="en")
        client.post("/run", files={"file": ("r.wav", b"audio")}, data={"language": "fa"})
        assert services_up["transcribe"][0] == ("whisper", "fa")

    def test_per_run_slots_override_the_session(self, client, services_up):
        start(client)
        client.post("/run", files={"file": ("r.wav", b"audio")},
                    data={"stt_slots_json": json.dumps([{"model": "seamless"}])})
        assert services_up["transcribe"] == [("seamless", None)]

    def test_malformed_slot_json_is_rejected(self, client):
        start(client)
        response = client.post("/run", files={"file": ("r.wav", b"audio")},
                               data={"stt_slots_json": "{not json"})
        assert response.status_code == 400


class TestDiscovery:
    def test_health_reports_both_services(self, client):
        body = client.get("/").json()
        assert body["buali_controller"] == "ok"
        assert body["stt"] is True and body["llm"] is True

    def test_model_lookups_are_proxied(self, client, monkeypatch):
        monkeypatch.setattr(stt_client, "list_models", lambda: {"available": ["whisper"]})
        monkeypatch.setattr(llm_client, "list_models",
                            lambda: {"available": ["aya"], "audio_capable": ["gemma"]})
        assert client.get("/models").json()["available"] == ["whisper"]
        assert client.get("/llm/models").json()["audio_capable"] == ["gemma"]

    def test_an_unreachable_service_is_a_bad_gateway(self, client, monkeypatch):
        def boom():
            raise RuntimeError("connection refused")

        monkeypatch.setattr(stt_client, "list_models", boom)
        assert client.get("/models").status_code == 502
