"""The HTTP surface of a stateless controller.

Three things carry the design and are tested hardest: nothing is remembered
between requests, no route runs without a token, and every failure names
itself with a code and says whether a retry could help.
"""
import json

import pytest

import config
import execution
import main
from conftest import BASE_CONFIG, TOKEN


class TestAuthentication:
    def test_health_needs_no_token(self, client):
        """A load balancer has no secret, and this leaks nothing."""
        assert client.get("/", headers={}).status_code == 200

    @pytest.mark.parametrize("path", ["/models", "/languages", "/llm/models"])
    def test_a_missing_token_is_rejected(self, client, path):
        response = client.get(path, headers={"X-Internal-Token": ""})
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "UNAUTHORIZED_INTERNAL_CALL"

    def test_a_wrong_token_is_rejected(self, client):
        response = client.get("/models", headers={"X-Internal-Token": "not-the-token"})
        assert response.status_code == 401

    def test_running_a_job_needs_a_token(self, run_job, client):
        client.headers.pop("X-Internal-Token")
        assert run_job().status_code == 401

    def test_an_unconfigured_service_fails_closed(self, client, monkeypatch):
        """No secret set is a deployment mistake, not permission to skip the
        check -- the alternative leaves the service open unnoticed."""
        monkeypatch.setattr(config, "INTERNAL_API_TOKEN", "")
        response = client.get("/models")
        assert response.status_code == 503
        assert response.json()["detail"]["error_code"] == "INTERNAL_AUTH_NOT_CONFIGURED"

    def test_the_token_is_compared_in_constant_time(self):
        """A plain == leaks the secret one character at a time."""
        import inspect
        assert "compare_digest" in inspect.getsource(main.require_internal_token)


class TestHealth:
    def test_reports_the_siblings_and_what_it_supports(self, client):
        body = client.get("/").json()
        assert body["stateless"] is True
        assert (body["stt"], body["llm"], body["evaluation"]) == (True, True, True)
        assert body["supported_pipeline_versions"] == ["buali-v1"]
        assert body["max_upload_bytes"] == config.MAX_UPLOAD_BYTES

    def test_says_whether_authentication_is_configured(self, client):
        assert client.get("/").json()["internal_auth_configured"] is True


class TestRunningAJob:
    def test_returns_the_report_with_the_job_id_echoed_back(self, run_job):
        body = run_job(job_id="job-42").json()
        assert body["job_id"] == "job-42"
        assert body["status"] == "completed"
        assert body["result"]["final_text"] == "report"

    def test_transcripts_are_returned_beside_the_report(self, run_job):
        body = run_job().json()
        assert body["result"]["stt_transcripts"] == {"transcript_1": "transcript from whisper"}

    def test_the_configuration_travels_with_the_result(self, run_job):
        metadata = run_job().json()["result"]["model_metadata"]
        assert metadata["pipeline"] == "separate"
        assert metadata["stt_models"] == ["whisper"]
        assert metadata["prompt_version"] == "radiology-v1"

    def test_elapsed_time_is_measured(self, run_job):
        assert run_job().json()["result"]["processing_metrics"]["total_seconds"] >= 0

    def test_the_real_filename_reaches_stt(self, run_job, services_up):
        """A cloud endpoint reads the container format off the extension."""
        run_job(filename="dictation.mp3")
        assert services_up["transcribe"][0][2] == "dictation.mp3"

    def test_multimodal_needs_no_stt_slot(self, run_job, services_up):
        response = run_job({"pipeline": "multimodal", "stt_slots": []})
        assert response.status_code == 200
        assert services_up["transcribe"] == []


class TestStatelessness:
    def test_two_jobs_do_not_share_configuration(self, run_job, services_up):
        """The failure this prevents: one caller's models silently applying to
        the next caller's recording."""
        run_job({"stt_slots": [{"slot_id": "stt_1", "model": "whisper"}]})
        run_job({"stt_slots": [{"slot_id": "stt_1", "model": "seamless"}]})
        assert [call[0] for call in services_up["transcribe"]] == ["whisper", "seamless"]

    def test_no_session_route_exists(self, client):
        for path in ("/session", "/run", "/status"):
            assert client.post(path).status_code == 404, f"{path} should be gone"

    def test_the_module_holds_no_job_state(self):
        assert not hasattr(main, "_session")


class TestConfigHash:
    def test_the_same_configuration_hashes_the_same(self, run_job):
        first = run_job(job_id="a").json()["config_hash"]
        second = run_job(job_id="b").json()["config_hash"]
        assert first == second, "the job id must not affect the hash"

    def test_a_different_configuration_hashes_differently(self, run_job):
        first = run_job().json()["config_hash"]
        second = run_job({"llm_model": "openai:gpt-4o"}).json()["config_hash"]
        assert first != second

    def test_key_order_in_the_request_does_not_change_the_hash(self, client):
        reordered = dict(reversed(list(BASE_CONFIG.items())))
        bodies = [
            client.post("/internal/jobs/j/execute",
                        files={"file": ("r.wav", b"audio")},
                        data={"config_json": json.dumps(payload)}).json()["config_hash"]
            for payload in (BASE_CONFIG, reordered)
        ]
        assert bodies[0] == bodies[1]

    def test_the_hash_carries_no_credential(self, run_job):
        """It is published in every result, so it must be safe to publish."""
        with_key = run_job(credentials={"llm": {"api_key": "sk-secret"}}).json()
        without = run_job().json()
        assert with_key["config_hash"] == without["config_hash"]


class TestCredentials:
    def test_a_per_request_key_reaches_the_llm(self, run_job, services_up):
        run_job({"pipeline": "multimodal", "stt_slots": []},
                credentials={"llm": {"api_key": "sk-from-request"}})
        assert services_up["complete"][0]["api_key"] == "sk-from-request"

    def test_a_slot_key_is_matched_by_slot_id(self, run_job, monkeypatch):
        import stt_client

        seen = {}

        def capture(audio, slot, language, filename="audio.wav"):
            seen[slot.slot_id] = slot.api_key.get_secret_value() if slot.api_key else None
            return "text"

        monkeypatch.setattr(stt_client, "transcribe", capture)
        run_job({"stt_slots": [{"slot_id": "stt_1", "model": "openai:whisper-1"},
                               {"slot_id": "stt_2", "model": "whisper"}]},
                credentials={"stt": {"stt_2": {"api_key": "sk-for-slot-2"}}})
        assert seen == {"stt_1": None, "stt_2": "sk-for-slot-2"}

    def test_a_credential_never_appears_in_the_response(self, run_job):
        body = run_job(credentials={"llm": {"api_key": "sk-secret"}}).text
        assert "sk-secret" not in body

    def test_duplicate_slot_ids_are_refused(self, run_job):
        """Credentials are addressed by slot_id, so a duplicate would send one
        slot's key to another."""
        response = run_job({"stt_slots": [{"slot_id": "same", "model": "whisper"},
                                          {"slot_id": "same", "model": "seamless"}]})
        assert response.status_code == 400
        assert response.json()["detail"]["error_code"] == "INVALID_PROCESSING_CONFIG"


class TestValidation:
    def test_an_unsupported_version_is_named_in_the_error(self, run_job):
        response = run_job({"prompt_version": "radiology-v99"})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error_code"] == "UNSUPPORTED_PROMPT_VERSION"
        assert detail["retryable"] is False

    @pytest.mark.parametrize("field,code", [
        ("preprocessing_version", "UNSUPPORTED_PREPROCESSING_VERSION"),
        ("pipeline_version", "UNSUPPORTED_PIPELINE_VERSION"),
        ("prompt_version", "UNSUPPORTED_PROMPT_VERSION"),
    ])
    def test_every_version_is_pinned(self, run_job, field, code):
        assert run_job({field: "nope"}).json()["detail"]["error_code"] == code

    def test_separate_without_a_slot_is_refused(self, run_job):
        response = run_job({"stt_slots": []})
        assert response.json()["detail"]["error_code"] == "INVALID_PIPELINE_CONFIG"

    def test_multimodal_with_a_slot_is_refused(self, run_job):
        """The model hears the recording itself; a slot means the caller
        misunderstood which pipeline they asked for."""
        response = run_job({"pipeline": "multimodal"})
        assert response.json()["detail"]["error_code"] == "INVALID_PIPELINE_CONFIG"

    def test_malformed_config_json_is_a_400(self, client):
        response = client.post("/internal/jobs/j/execute",
                               files={"file": ("r.wav", b"audio")},
                               data={"config_json": "{not json"})
        assert response.status_code == 400
        assert response.json()["detail"]["error_code"] == "INVALID_PROCESSING_CONFIG"

    def test_malformed_credentials_json_is_a_400(self, client):
        response = client.post("/internal/jobs/j/execute",
                               files={"file": ("r.wav", b"audio")},
                               data={"config_json": json.dumps(BASE_CONFIG),
                                     "credentials_json": "{not json"})
        assert response.json()["detail"]["error_code"] == "INVALID_EXECUTION_CREDENTIALS"

    def test_empty_audio_is_refused(self, run_job):
        assert run_job(audio=b"").json()["detail"]["error_code"] == "INVALID_AUDIO"


class TestUploadLimit:
    def test_an_oversized_upload_is_refused(self, run_job, monkeypatch):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
        response = run_job(audio=b"x" * 5000)
        assert response.status_code == 413
        assert response.json()["detail"]["error_code"] == "AUDIO_TOO_LARGE"

    def test_a_file_at_the_limit_is_accepted(self, run_job, monkeypatch):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 5000)
        assert run_job(audio=b"x" * 5000).status_code == 200


class TestErrorContract:
    def test_a_retryable_failure_is_a_503(self, run_job, monkeypatch):
        import llm_client

        def out_of_memory(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(llm_client, "complete", out_of_memory)
        response = run_job()
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error_code"] == "GPU_OUT_OF_MEMORY"
        assert detail["retryable"] is True

    def test_an_unfixable_failure_is_a_422(self, run_job):
        assert run_job({"prompt_version": "nope"}).status_code == 422

    def test_malformed_llm_output_is_named_not_passed_on(self, run_job, monkeypatch):
        import llm_client

        monkeypatch.setattr(llm_client, "complete",
                            lambda *a, **k: '{"final_text": "report"}')
        response = run_job()
        assert response.json()["detail"]["error_code"] == "INVALID_MODEL_OUTPUT"

    def test_an_upstream_error_does_not_leak_its_detail(self, run_job, monkeypatch):
        """The caller gets a code and a sentence, never our internals."""
        import llm_client

        def leaky(*args, **kwargs):
            raise RuntimeError("connection refused to http://internal-host:8001/v1/chat")

        monkeypatch.setattr(llm_client, "complete", leaky)
        assert "internal-host" not in run_job().text

    def test_every_error_body_has_the_same_three_fields(self, run_job):
        detail = run_job({"stt_slots": []}).json()["detail"]
        assert set(detail) == {"error_code", "message", "retryable"}


class TestAdminUnload:
    def test_unloads_both_services(self, client, services_up):
        assert client.post("/internal/admin/models/unload").json() == {"stt": True, "llm": True}
        assert sorted(services_up["unloaded"]) == ["llm", "stt"]

    def test_a_service_already_down_is_not_an_error(self, client, monkeypatch):
        import stt_client

        def refuse():
            raise RuntimeError("connection refused")

        monkeypatch.setattr(stt_client, "unload", refuse)
        body = client.post("/internal/admin/models/unload").json()
        assert body == {"stt": False, "llm": True}

    def test_it_needs_a_token(self, client):
        client.headers.pop("X-Internal-Token")
        assert client.post("/internal/admin/models/unload").status_code == 401
