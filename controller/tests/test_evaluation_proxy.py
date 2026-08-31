"""POST /evaluate — the gateway in front of the evaluation service."""
import pytest

import evaluation_client


def body(**overrides):
    payload = {
        "asset_id": "DPM89130",
        "model": "whisper-large-v3",
        "hypothesis": "There is a 6 mm stone in the distal right ureter.",
        "reference": "There is a 7 mm calculus in the distal right ureter.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def evaluation_service(monkeypatch):
    """Stand-in for the evaluation service, recording what it was sent."""
    calls = []
    reply = {"status": 200, "body": {"asset_id": "DPM89130", "requires_medical_review": True}}

    def fake_evaluate(payload):
        calls.append(payload)
        return reply["status"], reply["body"]

    monkeypatch.setattr(evaluation_client, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluation_client, "health", lambda: True)
    return {"calls": calls, "reply": reply}


class TestPassThrough:
    def test_result_is_returned_to_the_caller(self, client, evaluation_service):
        response = client.post("/evaluate", json=body())
        assert response.status_code == 200
        assert response.json()["requires_medical_review"] is True

    def test_the_request_reaches_the_service_intact(self, client, evaluation_service):
        client.post("/evaluate", json=body(pipeline="hybrid", model_version="v3"))
        sent = evaluation_service["calls"][0]
        assert sent["asset_id"] == "DPM89130"
        assert sent["hypothesis"].startswith("There is a 6 mm")
        assert sent["pipeline"] == "hybrid"
        assert sent["model_version"] == "v3"

    def test_unknown_fields_are_forwarded_rather_than_dropped(self, client, evaluation_service):
        """The evaluation service owns the contract, so it can add fields
        without the controller needing to know about them."""
        client.post("/evaluate", json=body(future_field="something"))
        assert evaluation_service["calls"][0]["future_field"] == "something"


class TestErrors:
    @pytest.mark.parametrize("missing", ["asset_id", "model", "hypothesis", "reference"])
    def test_missing_required_fields_fail_at_the_gateway(self, client, evaluation_service,
                                                          missing):
        payload = body()
        del payload[missing]
        assert client.post("/evaluate", json=payload).status_code == 422
        assert evaluation_service["calls"] == [], "must not forward an invalid request"

    def test_service_rejection_is_relayed_not_masked(self, client, evaluation_service):
        """A 422 from the service must not surface as a generic 502."""
        evaluation_service["reply"]["status"] = 422
        evaluation_service["reply"]["body"] = {"detail": "reference must not be empty"}
        response = client.post("/evaluate", json=body())
        assert response.status_code == 422
        assert response.json()["detail"] == "reference must not be empty"

    def test_unreachable_service_is_a_bad_gateway(self, client, monkeypatch):
        def boom(payload):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(evaluation_client, "evaluate", boom)
        response = client.post("/evaluate", json=body())
        assert response.status_code == 502
        assert "evaluation service" in response.json()["detail"]


class TestHealth:
    def test_evaluation_is_reported_alongside_the_others(self, client, evaluation_service):
        health = client.get("/").json()
        assert health["evaluation"] is True
        assert set(health) >= {"stt", "llm", "evaluation"}
