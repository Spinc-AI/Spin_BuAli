"""Running one job, from validated config to a stable result.

Everything between "the request parsed" and "here is the report" lives here,
so `main.py` stays routing and this stays logic.

Three jobs, in order:

1. **Refuse early.** A pipeline with no STT slot, or a version this build does
   not implement, fails before any model is loaded rather than after.
2. **Merge the credentials in.** Config arrives without secrets and keys arrive
   separately; they meet here, in memory, for the length of one call.
3. **Turn every failure into a code.** A caller needs to know whether to retry.
   An out-of-memory GPU and a malformed prompt are both "it didn't work", and
   only one of them is worth trying again.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

from fastapi import HTTPException
from pydantic import ValidationError

import config
import pipelines
from schemas import (
    AIJobResult,
    ExecutionCredentials,
    LLMReportOutput,
    LlmTarget,
    ModelMetadata,
    Pipeline,
    ProcessingConfig,
    ProcessingMetrics,
    RuntimeSttSlotConfig,
    reveal,
)

ProgressCallback = Callable[[str, int], None]


class AIExecutionError(Exception):
    """A failure the caller can act on.

    `safe_message` is what crosses the wire: a code and a sentence, never a
    stack trace or an upstream URL. `retryable` separates a transient
    resource problem from a request that will fail the same way forever.
    """

    def __init__(self, error_code: str, safe_message: str, retryable: bool = False):
        super().__init__(safe_message)
        self.error_code = error_code
        self.safe_message = safe_message
        self.retryable = retryable


# --- Identity --------------------------------------------------------------
def calculate_config_hash(processing_config: ProcessingConfig) -> str:
    """A stable SHA-256 over the job's configuration.

    Canonical JSON -- sorted keys, no incidental whitespace -- so the same
    settings always hash the same however the request happened to be written.
    The hash goes into the result, which is what makes a stored report
    traceable to the exact configuration that produced it.

    Safe to publish: `ProcessingConfig` holds no credentials by construction.
    """
    canonical = json.dumps(
        processing_config.model_dump(mode="json"),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Validation ------------------------------------------------------------
def validate_processing_config(cfg: ProcessingConfig) -> None:
    """Reject a job this build cannot honour, before it costs anything."""
    for version, supported, code in (
        (cfg.preprocessing_version, config.SUPPORTED_PREPROCESSING_VERSIONS,
         "UNSUPPORTED_PREPROCESSING_VERSION"),
        (cfg.pipeline_version, config.SUPPORTED_PIPELINE_VERSIONS,
         "UNSUPPORTED_PIPELINE_VERSION"),
        (cfg.prompt_version, config.SUPPORTED_PROMPT_VERSIONS,
         "UNSUPPORTED_PROMPT_VERSION"),
    ):
        if version not in supported:
            raise AIExecutionError(
                code, f"Unsupported version {version!r}; this build implements "
                      f"{sorted(supported)}.")

    slots = cfg.active_slots
    if cfg.pipeline.uses_stt and not slots:
        raise AIExecutionError(
            "INVALID_PIPELINE_CONFIG",
            f"The {cfg.pipeline.value} pipeline requires at least one STT slot.")
    if cfg.pipeline is Pipeline.MULTIMODAL and slots:
        raise AIExecutionError(
            "INVALID_PIPELINE_CONFIG",
            "The multimodal pipeline must not be given STT slots — the model "
            "hears the recording itself.")


# --- Credentials -----------------------------------------------------------
def build_runtime_stt_slots(cfg: ProcessingConfig,
                            credentials: ExecutionCredentials
                            ) -> list[Optional[RuntimeSttSlotConfig]]:
    """Pair each slot with its credential, by `slot_id`.

    Empty positions are preserved as None: the slot number is part of the
    result (`transcript_2` means the second slot), so the list cannot be
    compacted.
    """
    runtime: list[Optional[RuntimeSttSlotConfig]] = []
    for slot in cfg.stt_slots:
        if slot is None:
            runtime.append(None)
            continue
        credential = credentials.stt.get(slot.slot_id)
        runtime.append(RuntimeSttSlotConfig(
            slot_id=slot.slot_id,
            model=slot.model,
            language=slot.language,
            base_url=(credential.base_url if credential and credential.base_url
                      else slot.base_url),
            api_key=credential.api_key if credential else None,
        ))
    return runtime


def build_llm_target(cfg: ProcessingConfig, credentials: ExecutionCredentials) -> LlmTarget:
    """The LLM to call, with the per-request credential folded in."""
    credential = credentials.llm
    return LlmTarget(
        model=cfg.llm_model,
        api_key=reveal(credential.api_key) if credential else None,
        base_url=(credential.base_url if credential and credential.base_url
                  else cfg.llm_base_url),
    )


# --- Result ----------------------------------------------------------------
def normalize_pipeline_result(raw_result: dict, cfg: ProcessingConfig,
                              elapsed_seconds: float) -> AIJobResult:
    """Check what the LLM returned, then shape it into the output contract.

    The pipelines return the LLM's JSON with the slot transcripts merged in.
    Validating here means a model that answered with prose, or dropped
    `final_text`, produces one clear error rather than a half-formed report
    that only fails somewhere downstream.
    """
    transcripts = {key: value for key, value in raw_result.items()
                   if key.startswith("transcript_") and isinstance(value, str)}
    try:
        report = LLMReportOutput.model_validate({
            field: raw_result.get(field) for field in
            ("raw_transcript", "corrected_transcript", "final_text",
             "discrepancies_found", "notes")
        } | {"discrepancies_found": raw_result.get("discrepancies_found") or []})
    except ValidationError as exc:
        raise AIExecutionError(
            "INVALID_MODEL_OUTPUT",
            "The language model did not return a usable report structure.") from exc

    return AIJobResult(
        stt_transcripts=transcripts,
        raw_transcript=report.raw_transcript,
        corrected_transcript=report.corrected_transcript,
        final_text=report.final_text,
        discrepancies_found=report.discrepancies_found,
        notes=report.notes,
        model_metadata=ModelMetadata(
            pipeline=cfg.pipeline,
            stt_models=[slot.model for slot in cfg.active_slots],
            llm_model=cfg.llm_model,
            preprocessing_version=cfg.preprocessing_version,
            pipeline_version=cfg.pipeline_version,
            prompt_version=cfg.prompt_version,
            dictionary_version=cfg.dictionary_version,
        ),
        processing_metrics=ProcessingMetrics(total_seconds=round(elapsed_seconds, 3)),
    )


# --- The job ---------------------------------------------------------------
def execute_job(audio: bytes, audio_filename: str, processing_config: ProcessingConfig,
                credentials: Optional[ExecutionCredentials] = None,
                progress_callback: Optional[ProgressCallback] = None) -> AIJobResult:
    """Run one job end to end. Holds no state and returns no secret."""
    credentials = credentials or ExecutionCredentials()

    def progress(stage: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, percent)

    progress("validating_config", 5)
    validate_processing_config(processing_config)
    if not audio:
        raise AIExecutionError("INVALID_AUDIO", "The uploaded audio file is empty.")

    slots = build_runtime_stt_slots(processing_config, credentials)
    llm = build_llm_target(processing_config, credentials)

    progress({Pipeline.SEPARATE: "transcribing",
              Pipeline.MULTIMODAL: "multimodal_processing"}.get(
                  processing_config.pipeline, "hybrid_processing"), 20)

    started = time.perf_counter()
    try:
        raw_result = pipelines.run(
            processing_config.pipeline, audio, audio_filename,
            slots, processing_config.language, llm)
    except AIExecutionError:
        raise
    except Exception as exc:
        raise _classify(exc) from exc
    elapsed = time.perf_counter() - started

    progress("validating_output", 95)
    result = normalize_pipeline_result(raw_result, processing_config, elapsed)
    progress("completed", 100)
    return result


def _classify(exc: Exception) -> AIExecutionError:
    """Turn whatever went wrong into a code and a retry verdict.

    The resource cases are matched on the message text, because torch raises no
    distinct type for them. That has to happen *before* the HTTPException
    branch: `pipelines` wraps every upstream failure into a 502, so by the time
    the exception arrives here an out-of-memory GPU and an unplugged network
    cable look identical from the outside. The text is the only thing that
    still separates them.

    A weak signal used carefully -- it only ever adds information. Anything
    unrecognised falls through to a retryable pipeline failure rather than
    being reported as something specific it might not be.
    """
    message = str(exc.detail if isinstance(exc, HTTPException) else exc).lower()

    if isinstance(exc, MemoryError):
        return AIExecutionError("RESOURCE_EXHAUSTED",
                                "The AI service ran out of memory.", retryable=True)
    if "out of memory" in message or ("cuda" in message and "memory" in message):
        return AIExecutionError("GPU_OUT_OF_MEMORY",
                                "The GPU has insufficient free memory.", retryable=True)
    if "timeout" in message or "timed out" in message:
        return AIExecutionError("PROCESSING_TIMEOUT", "AI processing timed out.",
                                retryable=True)

    if isinstance(exc, HTTPException):
        # The pipelines already distinguish a bad request from a dead upstream.
        if exc.status_code == 400:
            return AIExecutionError("INVALID_PIPELINE_CONFIG", str(exc.detail))
        return AIExecutionError("AI_PIPELINE_FAILED", "The AI pipeline failed.",
                                retryable=exc.status_code >= 500)
    return AIExecutionError("AI_PIPELINE_FAILED",
                            "The AI pipeline failed unexpectedly.", retryable=True)
