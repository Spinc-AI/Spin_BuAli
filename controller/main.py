# """BuAli controller — a standalone FastAPI service for turning a spoken
# radiology report into a corrected transcript, in one of three pipelines:

#   separate   -- up to 3 independently-configured STT engines transcribe the
#                 audio; an LLM reconciles whichever transcripts were produced.
#   multimodal -- STT is skipped; an audio-capable LLM (local or cloud) is
#                 given the audio directly.
#   hybrid     -- both at once: STT slot(s) run AND the audio-capable LLM
#                 hears the audio, with the STT transcript(s) folded in as
#                 reference material.

# Extracted from Spin_Medical_Assistant_Project's Orchestrator (which drove
# this through a generic JSON-instruction engine) — same behavior, hardcoded.
# STT and Core_LLM stay in the main project; this is only an HTTP client of
# them (config.STT_URL / config.LLM_URL).
# """
# from __future__ import annotations

# import json
# from typing import Optional

# from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# import config
# import pipelines
# import stt_llm_client as client
# from schemas import Session, SessionRequest, SttSlotConfig

# app = FastAPI(title="Spin BuAli Controller")

# # SESSION: Optional[Session] = None
# _SECRET_FIELDS = {"stt_api_key", "llm_api_key"}


# def _redact_session(session: Session) -> dict:
#     data = session.model_dump(exclude=_SECRET_FIELDS)
#     if data.get("stt_slots"):
#         data["stt_slots"] = [
#             ({k: v for k, v in slot.items() if k != "api_key"} if slot is not None else None)
#             for slot in data["stt_slots"]
#         ]
#     return data


# @app.get("/")
# def health():
#     return {"buali_controller": "ok", "stt": client.stt_health(), "llm": client.llm_health(),
#             "openai_default_key_configured": bool(config.OPENAI_API_KEY),
#             "gemini_default_key_configured": bool(config.GEMINI_API_KEY)}


# @app.get("/models")
# def list_models():
#     """Proxy STT's available local models (for building a model picker)."""
#     try:
#         return client.stt_models()
#     except Exception as exc:
#         raise HTTPException(502, f"Could not fetch models from STT: {exc}")


# @app.get("/languages")
# def list_languages():
#     try:
#         return client.stt_languages()
#     except Exception as exc:
#         raise HTTPException(502, f"Could not fetch languages from STT: {exc}")


# @app.get("/status")
# def status():
#     if SESSION is None:
#         return {"active": False}
#     return {"active": True, **_redact_session(SESSION)}


# @app.post("/session")
# def start_session(req: SessionRequest):
#     """Choose the pipeline + models, validate reachability, report status.

#     No credential has to be configured on the server — pass it here (or
#     per-call in /run) instead. Local STT models are (re)loaded per-slot
#     during /run, not eagerly here (different slots may need different
#     models in sequence).
#     """
#     global SESSION
#     if req.pipeline not in ("separate", "multimodal", "hybrid"):
#         raise HTTPException(400, f"unknown pipeline '{req.pipeline}' — use "
#                                  "'separate', 'multimodal', or 'hybrid'")

#     slots = req.stt_slots or []
#     if len(slots) > config.MAX_STT_SLOTS:
#         raise HTTPException(400, f"at most {config.MAX_STT_SLOTS} STT slots are supported")
#     any_slot_configured = any(s is not None for s in slots)
#     any_local_slot = any(not client.is_api_model(s.model) for s in slots if s is not None)

#     if req.pipeline in ("multimodal", "hybrid"):
#         if not client.is_cloud_model(req.llm_model) and not client.llm_health():
#             raise HTTPException(503, f"LLM server not reachable at {config.LLM_URL} (needed for the "
#                                      "local multimodal model's /chat_audio endpoint)")
#         if req.pipeline == "hybrid":
#             if not any_slot_configured:
#                 raise HTTPException(400, "hybrid mode needs at least one configured STT slot "
#                                          "(stt_slots) as a reference transcript for the LLM — use "
#                                          "'multimodal' instead if you don't want one")
#             if any_local_slot and not client.stt_health():
#                 raise HTTPException(503, f"STT server not reachable at {config.STT_URL}")
#     else:  # separate
#         if not any_slot_configured:
#             raise HTTPException(400, "separate mode needs at least one configured STT slot (stt_slots)")
#         if any_local_slot and not client.stt_health():
#             raise HTTPException(503, f"STT server not reachable at {config.STT_URL}")
#         if not client.is_api_model(req.llm_model) and not client.llm_health():
#             raise HTTPException(503, f"LLM server not reachable at {config.LLM_URL}")

#     SESSION = Session(llm_model=req.llm_model, language=req.language,
#                       stt_api_key=req.stt_api_key, stt_base_url=req.stt_base_url,
#                       llm_api_key=req.llm_api_key, llm_base_url=req.llm_base_url,
#                       stt_slots=req.stt_slots, pipeline=req.pipeline,
#                       stt_ready=True, llm_ready=True)
#     return status()


# @app.post("/run")
# def run(file: UploadFile = File(...),
#         language: Optional[str] = Form(default=None),
#         stt_api_key: Optional[str] = Form(default=None),
#         stt_base_url: Optional[str] = Form(default=None),
#         llm_api_key: Optional[str] = Form(default=None),
#         llm_base_url: Optional[str] = Form(default=None),
#         stt_slots_json: Optional[str] = Form(default=None)):
#     """Run the active pipeline on an audio recording.

#     `language`, `stt_api_key`/`stt_base_url`, and `llm_api_key`/`llm_base_url`
#     override the session's defaults for this call. `stt_slots_json` (a
#     JSON-encoded array of {model, api_key?, base_url?, language?}, same shape
#     as POST /session's `stt_slots`) overrides the session's slot configs for
#     this call.
#     """
#     if SESSION is None:
#         raise HTTPException(409, "no active session - call POST /session first")

#     audio = file.file.read()

#     stt_slots = SESSION.stt_slots
#     if stt_slots_json:
#         try:
#             stt_slots = [SttSlotConfig(**s) if s is not None else None
#                         for s in json.loads(stt_slots_json)]
#         except (ValueError, TypeError) as exc:
#             raise HTTPException(400, f"invalid stt_slots_json: {exc}")

#     effective_language = language or SESSION.language
#     effective_llm_api_key = llm_api_key or SESSION.llm_api_key
#     effective_llm_base_url = llm_base_url or SESSION.llm_base_url

#     try:
#         if SESSION.pipeline == "separate":
#             result = pipelines.run_separate(
#                 audio, stt_slots or [], effective_language,
#                 SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
#             )
#         elif SESSION.pipeline == "multimodal":
#             result = pipelines.run_multimodal(
#                 audio, file.filename or "audio.wav",
#                 SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
#             )
#         else:  # hybrid
#             result = pipelines.run_hybrid(
#                 audio, file.filename or "audio.wav", stt_slots or [], effective_language,
#                 SESSION.llm_model, effective_llm_api_key, effective_llm_base_url,
#             )
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(502, str(exc))
#     return {"pipeline": SESSION.pipeline, "result": result}


# @app.post("/session/unload")
# def unload():
#     """Unload the models from the modules, then drop the active session."""
#     global SESSION
#     llm_model = SESSION.llm_model if SESSION else None
#     try:
#         client.stt_unload()
#     except Exception:
#         pass  # best-effort: module may already be down
#     try:
#         client.llm_unload(llm_model)
#     except Exception:
#         pass
#     SESSION = None
#     return {"active": False}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host=config.HOST, port=config.PORT)

"""Stateless BuAli AI Controller.


This service executes one immutable AI processing request.
"""

from __future__ import annotations

import hmac
import json
import logging

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

import config
import stt_llm_client as client

from execution import (
    AIExecutionError,
    calculate_config_hash,
    execute_job,
)
from schemas import (
    ExecutionCredentials,
    JobExecutionResponse,
    ProcessingConfig,
)


logger = logging.getLogger("buali.ai_controller")

app = FastAPI(title="Spin BuAli AI Controller")


def require_internal_token(
    x_internal_token: str | None = Header(default=None),
) -> None:
    """Allow only Backend or an authorized development client."""

    if not config.INTERNAL_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "INTERNAL_AUTH_NOT_CONFIGURED",
                "message": "Internal authentication is not configured.",
                "retryable": False,
            },
        )

    if not x_internal_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "UNAUTHORIZED_INTERNAL_CALL",
                "message": "Internal service token is missing.",
                "retryable": False,
            },
        )

    if not hmac.compare_digest(
        x_internal_token,
        config.INTERNAL_API_TOKEN,
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "UNAUTHORIZED_INTERNAL_CALL",
                "message": "Internal service token is invalid.",
                "retryable": False,
            },
        )


async def read_upload_limited(
    upload: UploadFile,
    max_bytes: int,
) -> bytes:
    """Read uploaded audio while enforcing a maximum size."""

    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 1024 * 1024

    while True:
        chunk = await upload.read(chunk_size)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": "AUDIO_TOO_LARGE",
                    "message": (
                        f"Audio exceeds maximum size of "
                        f"{max_bytes} bytes."
                    ),
                    "retryable": False,
                },
            )

        chunks.append(chunk)

    return b"".join(chunks)


@app.get("/")
def health():
    """Service health; this does not load any model."""

    return {
        "service": "buali-ai-controller",
        "status": "ok",
        "stateless": True,
        "stt_reachable": client.stt_health(),
        "llm_reachable": client.llm_health(),
        "supported_preprocessing_versions": sorted(
            config.SUPPORTED_PREPROCESSING_VERSIONS
        ),
        "supported_pipeline_versions": sorted(
            config.SUPPORTED_PIPELINE_VERSIONS
        ),
        "supported_prompt_versions": sorted(
            config.SUPPORTED_PROMPT_VERSIONS
        ),
    }


@app.get(
    "/models",
    dependencies=[Depends(require_internal_token)],
)
def list_models():
    try:
        return client.stt_models()

    except Exception as exc:
        logger.exception("Could not fetch STT models")

        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "STT_UNAVAILABLE",
                "message": "Could not fetch available STT models.",
                "retryable": True,
            },
        ) from exc


@app.get(
    "/languages",
    dependencies=[Depends(require_internal_token)],
)
def list_languages():
    try:
        return client.stt_languages()

    except Exception as exc:
        logger.exception("Could not fetch STT languages")

        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "STT_UNAVAILABLE",
                "message": "Could not fetch supported STT languages.",
                "retryable": True,
            },
        ) from exc


@app.post(
    "/internal/jobs/{job_id}/execute",
    response_model=JobExecutionResponse,
    dependencies=[Depends(require_internal_token)],
)
async def execute_internal_job(
    job_id: str,
    file: UploadFile = File(...),
    config_json: str = Form(...),
    credentials_json: str | None = Form(default=None),
):
    """Execute a Job created and owned by Backend.

    This endpoint does not create, save or update the Job.
    """

    try:
        config_data = json.loads(config_json)
        processing_config = ProcessingConfig.model_validate(config_data)

    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_PROCESSING_CONFIG",
                "message": "config_json is invalid.",
                "retryable": False,
            },
        ) from exc

    if credentials_json:
        try:
            credentials_data = json.loads(credentials_json)

            credentials = ExecutionCredentials.model_validate(
                credentials_data
            )

        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_EXECUTION_CREDENTIALS",
                    "message": "credentials_json is invalid.",
                    "retryable": False,
                },
            ) from exc
    else:
        credentials = ExecutionCredentials()

    audio = await read_upload_limited(
        file,
        config.MAX_UPLOAD_BYTES,
    )

    try:
        result = await run_in_threadpool(
            execute_job,
            audio,
            file.filename or "audio.wav",
            processing_config,
            credentials,
        )

    except AIExecutionError as exc:
        logger.exception(
            "AI Job failed: job_id=%s error_code=%s",
            job_id,
            exc.error_code,
        )

        status_code = 503 if exc.retryable else 422

        raise HTTPException(
            status_code=status_code,
            detail={
                "error_code": exc.error_code,
                "message": exc.safe_message,
                "retryable": exc.retryable,
            },
        ) from exc

    return JobExecutionResponse(
        job_id=job_id,
        status="completed",
        config_hash=calculate_config_hash(processing_config),
        result=result,
    )


@app.post(
    "/internal/admin/models/unload",
    dependencies=[Depends(require_internal_token)],
)
async def admin_unload_models():
    """Administrative model unload, unrelated to user Sessions."""

    stt_unloaded = False
    llm_unloaded = False

    try:
        await run_in_threadpool(client.stt_unload)
        stt_unloaded = True
    except Exception:
        logger.exception("Could not unload STT model")

    try:
        await run_in_threadpool(client.llm_unload, None)
        llm_unloaded = True
    except Exception:
        logger.exception("Could not unload LLM model")

    return {
        "stt_unloaded": stt_unloaded,
        "llm_unloaded": llm_unloaded,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
    )
