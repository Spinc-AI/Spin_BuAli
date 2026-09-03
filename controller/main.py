"""Spin BuAli controller -- the only door into the system.

Stateless. Every request carries everything it needs: the recording, the
configuration, and the credentials for that one call. The controller stores
nothing between requests and owns no session, so two callers cannot interfere
with each other and a restart loses nothing.

    POST /internal/jobs/{job_id}/execute     run one recording through a pipeline
    POST /evaluate                           score a transcript against a reference
    GET  /  /models  /languages  /llm/models

`job_id` is the caller's identifier, echoed back on the result. The controller
does not create it, store it, or look it up -- it exists so a result can be
matched to the request that asked for it.

Routing and validation live here; the work lives in execution.py.
"""
import hmac
import json
import logging
from typing import Callable

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

import config
import evaluation_client
import execution
import llm_client
import stt_client
from execution import AIExecutionError, calculate_config_hash, execute_job
from schemas import (
    EvaluationRequest,
    ExecutionCredentials,
    JobExecutionResponse,
    ProcessingConfig,
)

logger = logging.getLogger("buali.controller")

app = FastAPI(title="Spin BuAli Controller")


def _error(code: str, message: str, retryable: bool = False) -> dict:
    """The body shape every failure uses, so a caller parses one thing."""
    return {"error_code": code, "message": message, "retryable": retryable}


# --- Authentication --------------------------------------------------------
def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Allow only the backend, or a developer holding the same secret.

    An unset token fails closed with 503: no secret configured is a deployment
    mistake, and treating it as "no check needed" would leave the service open
    at exactly the moment nobody noticed.
    """
    if not config.INTERNAL_API_TOKEN:
        raise HTTPException(503, _error(
            "INTERNAL_AUTH_NOT_CONFIGURED",
            "Internal authentication is not configured on this service."))
    if not x_internal_token:
        raise HTTPException(401, _error(
            "UNAUTHORIZED_INTERNAL_CALL", "Internal service token is missing."))
    # Constant-time: a plain == leaks the secret one character at a time.
    if not hmac.compare_digest(x_internal_token, config.INTERNAL_API_TOKEN):
        raise HTTPException(401, _error(
            "UNAUTHORIZED_INTERNAL_CALL", "Internal service token is invalid."))


authenticated = [Depends(require_internal_token)]


# --- Health and discovery --------------------------------------------------
@app.get("/")
def health() -> dict:
    """Unauthenticated on purpose: a load balancer has no token, and this
    reveals nothing beyond which services are up."""
    return {
        "buali_controller": "ok",
        "stateless": True,
        "stt": stt_client.health(),
        "llm": llm_client.health(),
        "evaluation": evaluation_client.health(),
        "internal_auth_configured": bool(config.INTERNAL_API_TOKEN),
        "openai_key_configured": bool(config.OPENAI_API_KEY),
        "gemini_key_configured": bool(config.GEMINI_API_KEY),
        "supported_preprocessing_versions": sorted(config.SUPPORTED_PREPROCESSING_VERSIONS),
        "supported_pipeline_versions": sorted(config.SUPPORTED_PIPELINE_VERSIONS),
        "supported_prompt_versions": sorted(config.SUPPORTED_PROMPT_VERSIONS),
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
    }


def _proxy(service: str, fetch: Callable[[], dict]) -> dict:
    try:
        return fetch()
    except Exception as exc:
        logger.exception("Could not reach the %s service", service)
        raise HTTPException(502, _error(
            f"{service.upper()}_UNAVAILABLE",
            f"Could not reach the {service} service.", retryable=True)) from exc


@app.get("/models", dependencies=authenticated)
def stt_models() -> dict:
    """The local STT service's model registry."""
    return _proxy("stt", stt_client.list_models)


@app.get("/languages", dependencies=authenticated)
def stt_languages() -> dict:
    """The language codes the local STT service accepts."""
    return _proxy("stt", stt_client.list_languages)


@app.get("/llm/models", dependencies=authenticated)
def llm_models() -> dict:
    """The local LLM service's registry, and which of those accept audio."""
    return _proxy("llm", llm_client.list_models)


# --- Running a job ---------------------------------------------------------
async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload, refusing it the moment it grows too large.

    Checked while streaming rather than after: `await file.read()` on a
    multi-gigabyte upload has already cost the memory by the time its size
    could be inspected.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, _error(
                "AUDIO_TOO_LARGE", f"Audio exceeds the {max_bytes} byte maximum."))
        chunks.append(chunk)
    return b"".join(chunks)


def _parse(model, raw: str, code: str, what: str):
    try:
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(400, _error(code, f"{what} is invalid: {exc}")) from exc


@app.post("/internal/jobs/{job_id}/execute",
          response_model=JobExecutionResponse, dependencies=authenticated)
async def execute_internal_job(
    job_id: str,
    file: UploadFile = File(...),
    config_json: str = Form(...),
    credentials_json: str | None = Form(default=None),
) -> JobExecutionResponse:
    """Run one recording through one pipeline.

    The job is the caller's; this endpoint neither creates nor stores it.
    `config_json` is a ProcessingConfig and is safe to keep. `credentials_json`
    is an ExecutionCredentials, is used for this call only, and is never
    written anywhere.
    """
    processing_config = _parse(ProcessingConfig, config_json,
                               "INVALID_PROCESSING_CONFIG", "config_json")
    credentials = (_parse(ExecutionCredentials, credentials_json,
                          "INVALID_EXECUTION_CREDENTIALS", "credentials_json")
                   if credentials_json else ExecutionCredentials())

    audio = await read_upload_limited(file, config.MAX_UPLOAD_BYTES)

    try:
        # Synchronous and CPU/GPU-bound, so it must not block the event loop.
        result = await run_in_threadpool(
            execute_job, audio, file.filename or "audio.wav",
            processing_config, credentials)
    except AIExecutionError as exc:
        logger.warning("job %s failed: %s", job_id, exc.error_code)
        # 503 invites a retry, 422 says the request itself was the problem.
        raise HTTPException(503 if exc.retryable else 422,
                            _error(exc.error_code, exc.safe_message, exc.retryable))

    return JobExecutionResponse(
        job_id=job_id, status="completed",
        config_hash=calculate_config_hash(processing_config), result=result)


@app.post("/internal/admin/models/unload", dependencies=authenticated)
async def admin_unload_models(llm_model: str | None = Form(default=None)) -> dict:
    """Free the local services' weights.

    An operational lever, not part of any job's lifecycle -- which is why it
    sits under /internal/admin rather than being tied to a session that no
    longer exists. Best effort: a service already down is not an error here.
    """
    released = {}
    for name, release in (("stt", stt_client.unload),
                          ("llm", lambda: llm_client.unload(llm_model))):
        try:
            await run_in_threadpool(release)
            released[name] = True
        except Exception:
            logger.exception("Could not unload the %s model", name)
            released[name] = False
    return released


# --- Evaluation ------------------------------------------------------------
@app.post("/evaluate", dependencies=authenticated)
def evaluate(request: EvaluationRequest) -> dict:
    """Score a transcript against a radiologist-verified reference.

    A pass-through to the evaluation service, which owns the metric contract.
    Callers reach it here because the controller is the only entry point --
    the scoring module is never addressed directly.
    """
    try:
        status_code, body = evaluation_client.evaluate(request.model_dump())
    except Exception as exc:
        logger.exception("Could not reach the evaluation service")
        raise HTTPException(502, _error(
            "EVALUATION_UNAVAILABLE", "Could not reach the evaluation service.",
            retryable=True)) from exc

    if status_code != 200:
        # Surface the service's own complaint rather than a blanket 502.
        raise HTTPException(status_code, _error(
            "EVALUATION_REJECTED", str(body.get("detail", body))))
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
