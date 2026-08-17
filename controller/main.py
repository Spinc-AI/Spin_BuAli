"""BuAli controller — a standalone FastAPI service for turning a spoken
radiology report into a corrected transcript, in one of three pipelines:

  separate   -- up to 3 independently-configured STT engines transcribe the
                audio; an LLM reconciles whichever transcripts were produced.
  multimodal -- STT is skipped; an audio-capable LLM (local or cloud) is
                given the audio directly.
  hybrid     -- both at once: STT slot(s) run AND the audio-capable LLM
                hears the audio, with the STT transcript(s) folded in as
                reference material.

Extracted from Spin_Medical_Assistant_Project's Orchestrator (which drove
this through a generic JSON-instruction engine) — same behavior, hardcoded.
STT and Core_LLM stay in the main project; this is only an HTTP client of
them (config.STT_URL / config.LLM_URL).
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import config
import pipelines
import stt_llm_client as client
from schemas import Session, SessionRequest, SttSlotConfig

app = FastAPI(title="Spin BuAli Controller")

SESSION: Optional[Session] = None
_SECRET_FIELDS = {"stt_api_key", "llm_api_key"}


def _redact_session(session: Session) -> dict:
    data = session.model_dump(exclude=_SECRET_FIELDS)
    if data.get("stt_slots"):
        data["stt_slots"] = [
            ({k: v for k, v in slot.items() if k != "api_key"} if slot is not None else None)
            for slot in data["stt_slots"]
        ]
    return data


@app.get("/")
def health():
    return {"buali_controller": "ok", "stt": client.stt_health(), "llm": client.llm_health(),
            "openai_default_key_configured": bool(config.OPENAI_API_KEY),
            "gemini_default_key_configured": bool(config.GEMINI_API_KEY)}


@app.get("/models")
def list_models():
    """Proxy STT's available local models (for building a model picker)."""
    try:
        return client.stt_models()
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch models from STT: {exc}")


@app.get("/languages")
def list_languages():
    try:
        return client.stt_languages()
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch languages from STT: {exc}")


@app.get("/status")
def status():
    if SESSION is None:
        return {"active": False}
    return {"active": True, **_redact_session(SESSION)}


@app.post("/session")
def start_session(req: SessionRequest):
    """Choose the pipeline + models, validate reachability, report status.

    No credential has to be configured on the server — pass it here (or
    per-call in /run) instead. Local STT models are (re)loaded per-slot
    during /run, not eagerly here (different slots may need different
    models in sequence).
    """
    global SESSION
    if req.pipeline not in ("separate", "multimodal", "hybrid"):
        raise HTTPException(400, f"unknown pipeline '{req.pipeline}' — use "
                                 "'separate', 'multimodal', or 'hybrid'")

    slots = req.stt_slots or []
    if len(slots) > config.MAX_STT_SLOTS:
        raise HTTPException(400, f"at most {config.MAX_STT_SLOTS} STT slots are supported")
    any_slot_configured = any(s is not None for s in slots)
    any_local_slot = any(not client.is_api_model(s.model) for s in slots if s is not None)

    if req.pipeline in ("multimodal", "hybrid"):
        if not client.is_cloud_model(req.llm_model) and not client.llm_health():
            raise HTTPException(503, f"LLM server not reachable at {config.LLM_URL} (needed for the "
                                     "local multimodal model's /chat_audio endpoint)")
        if req.pipeline == "hybrid":
            if not any_slot_configured:
                raise HTTPException(400, "hybrid mode needs at least one configured STT slot "
                                         "(stt_slots) as a reference transcript for the LLM — use "
                                         "'multimodal' instead if you don't want one")
            if any_local_slot and not client.stt_health():
                raise HTTPException(503, f"STT server not reachable at {config.STT_URL}")
    else:  # separate
        if not any_slot_configured:
            raise HTTPException(400, "separate mode needs at least one configured STT slot (stt_slots)")
        if any_local_slot and not client.stt_health():
            raise HTTPException(503, f"STT server not reachable at {config.STT_URL}")
        if not client.is_api_model(req.llm_model) and not client.llm_health():
            raise HTTPException(503, f"LLM server not reachable at {config.LLM_URL}")

    SESSION = Session(llm_model=req.llm_model, language=req.language,
                      stt_api_key=req.stt_api_key, stt_base_url=req.stt_base_url,
                      llm_api_key=req.llm_api_key, llm_base_url=req.llm_base_url,
                      stt_slots=req.stt_slots, pipeline=req.pipeline,
                      stt_ready=True, llm_ready=True)
    return status()


@app.post("/run")
def run(file: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
        stt_api_key: Optional[str] = Form(default=None),
        stt_base_url: Optional[str] = Form(default=None),
        llm_api_key: Optional[str] = Form(default=None),
        llm_base_url: Optional[str] = Form(default=None),
        stt_slots_json: Optional[str] = Form(default=None)):
    """Run the active pipeline on an audio recording.

    `language`, `stt_api_key`/`stt_base_url`, and `llm_api_key`/`llm_base_url`
    override the session's defaults for this call. `stt_slots_json` (a
    JSON-encoded array of {model, api_key?, base_url?, language?}, same shape
    as POST /session's `stt_slots`) overrides the session's slot configs for
    this call.
    """
    if SESSION is None:
        raise HTTPException(409, "no active session - call POST /session first")

    audio = file.file.read()

    stt_slots = SESSION.stt_slots
    if stt_slots_json:
        try:
            stt_slots = [SttSlotConfig(**s) if s is not None else None
                        for s in json.loads(stt_slots_json)]
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid stt_slots_json: {exc}")

    effective_language = language or SESSION.language
    effective_llm_api_key = llm_api_key or SESSION.llm_api_key
    effective_llm_base_url = llm_base_url or SESSION.llm_base_url

    try:
        if SESSION.pipeline == "separate":
            result = pipelines.run_separate(
                audio, stt_slots or [], effective_language,
                SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
            )
        elif SESSION.pipeline == "multimodal":
            result = pipelines.run_multimodal(
                audio, file.filename or "audio.wav",
                SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
            )
        else:  # hybrid
            result = pipelines.run_hybrid(
                audio, file.filename or "audio.wav", stt_slots or [], effective_language,
                SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, str(exc))
    return {"pipeline": SESSION.pipeline, "result": result}


@app.post("/session/unload")
def unload():
    """Unload the models from the modules, then drop the active session."""
    global SESSION
    llm_model = SESSION.llm_model if SESSION else None
    try:
        client.stt_unload()
    except Exception:
        pass  # best-effort: module may already be down
    try:
        client.llm_unload(llm_model)
    except Exception:
        pass
    SESSION = None
    return {"active": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
