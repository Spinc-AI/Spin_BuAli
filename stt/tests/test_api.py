'''Lightweight API tests that do not load any heavy model weights.'''

from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# Model listing
# ============================================================

def test_list_models_reports_registry():
    with TestClient(app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert "whisper" in body["available"]
    assert "seamless" in body["available"]
    assert body["loaded"] is None


# ============================================================
# Guard rails
# ============================================================

def test_unknown_model_is_rejected():
    with TestClient(app) as client:
        response = client.post("/models/does-not-exist/load")
    assert response.status_code == 404


def test_transcribe_requires_a_loaded_model():
    with TestClient(app) as client:
        response = client.post(
            "/transcribe", files={"file": ("clip.wav", b"not-real-audio", "audio/wav")}
        )
    assert response.status_code == 409
