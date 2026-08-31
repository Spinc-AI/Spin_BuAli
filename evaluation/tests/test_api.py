"""The HTTP surface."""
import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def request_body(**overrides):
    body = {
        "asset_id": "DPM89130",
        "model": "whisper-large-v3",
        "hypothesis": "There is a 6 mm stone in the distal right ureter.",
        "reference": "There is a 7 mm calculus in the distal right ureter.",
    }
    body.update(overrides)
    return body


class TestHealth:
    def test_reports_the_active_versions(self, client):
        body = client.get("/").json()
        assert body["evaluation_service"] == "ok"
        assert body["metrics_version"] and body["terms_version"] and body["terms_sha"]


class TestEvaluate:
    def test_returns_the_full_report(self, client):
        body = client.post("/evaluate", json=request_body()).json()
        assert body["asset_id"] == "DPM89130"
        assert body["model"] == "whisper-large-v3"
        assert body["clinical_counts"]["number_errors"] == 1
        assert body["requires_medical_review"] is True

    def test_identifying_fields_come_back(self, client):
        """Results must be attributable per file and per model."""
        body = client.post("/evaluate", json=request_body(
            pipeline="hybrid", model_version="v3-turbo-2026-08")).json()
        assert body["pipeline"] == "hybrid"
        assert body["model_version"] == "v3-turbo-2026-08"

    def test_version_stamp_is_present(self, client):
        body = client.post("/evaluate", json=request_body()).json()
        assert set(body["evaluation"]) == {"metrics_version", "terms_version", "terms_sha"}

    def test_a_clean_report_needs_no_review(self, client):
        text = "There is a 6 mm stone in the distal right ureter."
        body = client.post("/evaluate", json=request_body(
            hypothesis=text, reference=text)).json()
        assert body["requires_medical_review"] is False
        assert body["general"]["wer"] == 0.0

    @pytest.mark.parametrize("missing", ["asset_id", "model", "hypothesis", "reference"])
    def test_required_fields_are_enforced(self, client, missing):
        body = request_body()
        del body[missing]
        assert client.post("/evaluate", json=body).status_code == 422

    def test_empty_texts_are_accepted_without_crashing(self, client):
        response = client.post("/evaluate", json=request_body(hypothesis="", reference=""))
        assert response.status_code == 200
        assert response.json()["general"]["wer"] == 0.0
