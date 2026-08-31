"""Spin BuAli controller -- the HTTP API for turning a spoken radiology
report into a corrected transcript.

Configure a session (which pipeline, which models), then POST recordings to
/run. The pipelines themselves live in pipelines.py; this module is routing,
validation, and the one active session.

Only one session exists at a time: this drives a single radiologist's
dictation workstation, not concurrent users.
"""
import json
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import config
import evaluation_client
import llm_client
import pipelines
import providers
import stt_client
from schemas import EvaluationRequest, LlmTarget, SessionConfig, SttSlotConfig, reveal

app = FastAPI(title="Spin BuAli Controller")

_session: SessionConfig | None = None


def _require_session() -> SessionConfig:
    if _session is None:
        raise HTTPException(409, "no active session — call POST /session first")
    return _session


def _proxy(service: str, fetch: Callable[[], dict]) -> dict:
    """Forward a lookup to a sibling service, reporting unreachability as 502."""
    try:
        return fetch()
    except Exception as exc:
        raise HTTPException(502, f"could not reach the {service} service: {exc}")


# --- health and discovery --------------------------------------------------
@app.get("/")
def health() -> dict:
    return {
        "buali_controller": "ok",
        "stt": stt_client.health(),
        "llm": llm_client.health(),
        "evaluation": evaluation_client.health(),
        "openai_key_configured": bool(config.OPENAI_API_KEY),
        "gemini_key_configured": bool(config.GEMINI_API_KEY),
    }


@app.get("/models")
def stt_models() -> dict:
    """The local STT service's model registry."""
    return _proxy("STT", stt_client.list_models)


@app.get("/languages")
def stt_languages() -> dict:
    """The language codes the local STT service accepts."""
    return _proxy("STT", stt_client.list_languages)


@app.get("/llm/models")
def llm_models() -> dict:
    """The local LLM service's registry, and which of those accept audio."""
    return _proxy("LLM", llm_client.list_models)


# --- session ---------------------------------------------------------------
@app.get("/status")
def status() -> dict:
    if _session is None:
        return {"active": False}
    return {"active": True, **_session.model_dump(mode="json")}


@app.post("/session")
def start_session(request: SessionConfig) -> dict:
    """Pick the pipeline and models for the runs that follow.

    Nothing has to be configured server-side: credentials can be passed here,
    or per call in /run. Local STT models are loaded during /run rather than
    here, since consecutive slots may each need a different one.
    """
    global _session
    _check_reachable(request)
    _session = request
    return status()


@app.post("/session/unload")
def unload_session() -> dict:
    """Free the local services' models, then drop the session."""
    global _session
    model = _session.llm_model if _session else None
    for release in (stt_client.unload, lambda: llm_client.unload(model)):
        try:
            release()
        except Exception:
            pass  # best effort: the service may already be down
    _session = None
    return {"active": False}


def _check_reachable(request: SessionConfig) -> None:
    """Fail fast on a session that cannot possibly run."""
    slots = request.active_slots
    if request.pipeline.uses_stt and not slots:
        raise HTTPException(
            400,
            f"the '{request.pipeline.value}' pipeline needs at least one configured STT slot"
            + (" — use 'multimodal' if you don't want one"
               if request.pipeline.uses_audio_llm else ""),
        )
    if any(not providers.is_cloud(slot.model) for slot in slots) and not stt_client.health():
        raise HTTPException(503, f"STT service not reachable at {config.STT_URL}")
    if not providers.is_cloud(request.llm_model) and not llm_client.health():
        raise HTTPException(503, f"LLM service not reachable at {config.LLM_URL}")


# --- run -------------------------------------------------------------------
@app.post("/run")
def run(file: UploadFile = File(...),
        language: str | None = Form(default=None),
        llm_api_key: str | None = Form(default=None),
        llm_base_url: str | None = Form(default=None),
        stt_slots_json: str | None = Form(default=None)) -> dict:
    """Run the active pipeline over one recording.

    Every form field except `file` overrides the session's setting for this
    call only. `stt_slots_json` is a JSON array in the same shape as
    POST /session's `stt_slots`.
    """
    session = _require_session()
    slots = _slots_for_run(session, stt_slots_json)
    llm = LlmTarget(
        model=session.llm_model,
        api_key=llm_api_key or reveal(session.llm_api_key),
        base_url=llm_base_url or session.llm_base_url,
    )
    result = pipelines.run(
        session.pipeline, file.file.read(), file.filename or "audio.wav",
        slots, language or session.language, llm,
    )
    return {"pipeline": session.pipeline.value, "result": result}


def _slots_for_run(session: SessionConfig,
                   stt_slots_json: str | None) -> list[SttSlotConfig | None]:
    if not stt_slots_json:
        return session.stt_slots or []
    try:
        return [SttSlotConfig(**slot) if slot is not None else None
                for slot in json.loads(stt_slots_json)]
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"invalid stt_slots_json: {exc}")


# --- evaluation ------------------------------------------------------------
@app.post("/evaluate")
def evaluate(request: EvaluationRequest) -> dict:
    """Score a transcript against a radiologist-verified reference.

    A pass-through to the evaluation service, which owns the metric contract.
    Callers reach it here because the controller is the only entry point --
    the scoring module is never addressed directly.
    """
    status_code, body = _forward(evaluation_client.evaluate, request.model_dump())
    if status_code != 200:
        # Surface the service's own complaint rather than a blanket 502.
        raise HTTPException(status_code, body.get("detail", body))
    return body


def _forward(call, payload):
    try:
        return call(payload)
    except Exception as exc:
        raise HTTPException(502, f"could not reach the evaluation service: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
