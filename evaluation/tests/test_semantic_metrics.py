"""The optional embedding metrics.

These load a model, so the suite must pass whether or not the extra
dependencies are installed. Anything that needs them is skipped when they are
absent; the opt-in behaviour itself is tested either way.
"""
import pytest
from fastapi.testclient import TestClient

import main
import semantic_metrics

INSTALLED, REASON = semantic_metrics.available()
needs_extras = pytest.mark.skipif(not INSTALLED, reason=f"optional extras absent: {REASON}")


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def body(**overrides):
    payload = {
        "asset_id": "DPM89130", "model": "whisper-large-v3",
        "hypothesis": "a 6 mm stone", "reference": "a 6 mm calculus",
    }
    payload.update(overrides)
    return payload


class TestAvailability:
    def test_availability_is_reported_as_a_reason_not_a_crash(self):
        usable, reason = semantic_metrics.available()
        assert isinstance(usable, bool)
        assert usable or reason, "an unusable state must explain itself"

    def test_health_advertises_the_capability(self, client):
        """A caller should be able to find out before asking."""
        assert client.get("/").json()["semantic_metrics_available"] == INSTALLED


class TestOptIn:
    def test_omitted_by_default(self, client):
        assert client.post("/evaluate", json=body()).json().get("semantic") is None

    @pytest.mark.skipif(INSTALLED, reason="extras are installed here")
    def test_asking_without_the_extras_is_a_clear_503(self, client):
        response = client.post("/evaluate", json=body(include_semantic=True))
        assert response.status_code == 503
        assert "semantic metrics unavailable" in response.json()["detail"]


@needs_extras
class TestComputation:
    def test_identical_text_scores_high(self):
        scores = semantic_metrics.compute("the kidney is normal", "the kidney is normal")
        assert scores["semantic_similarity"] > 0.95
        assert scores["bertscore_f1"] > 0.95

    def test_returned_through_the_api_when_requested(self, client):
        semantic = client.post("/evaluate", json=body(include_semantic=True)).json()["semantic"]
        assert set(semantic) == {"bertscore_f1", "semantic_similarity"}
