

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

from fastapi import HTTPException
from pydantic import ValidationError

import config
import pipelines
import stt_llm_client as client

from schemas import (
    AIJobResult,
    ExecutionCredentials,
    LLMReportOutput,
    ModelMetadata,
    ProcessingConfig,
    ProcessingMetrics,
    RuntimeSttSlotConfig,
)


ProgressCallback = Callable[[str, int], None]


class AIExecutionError(Exception):
    """Safe, structured error returned to Backend."""

    def __init__(
        self,
        error_code: str,
        safe_message: str,
        retryable: bool = False,
    ):
        super().__init__(safe_message)
        self.error_code = error_code
        self.safe_message = safe_message
        self.retryable = retryable


def calculate_config_hash(processing_config: ProcessingConfig) -> str:
    """Calculate a stable SHA-256 over the non-secret Job configuration."""

    canonical_json = json.dumps(
        processing_config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def validate_processing_config(cfg: ProcessingConfig) -> None:
    """Validate relations between pipeline, models and STT slots."""

    if cfg.preprocessing_version not in config.SUPPORTED_PREPROCESSING_VERSIONS:
        raise AIExecutionError(
            error_code="UNSUPPORTED_PREPROCESSING_VERSION",
            safe_message=(
                f"Unsupported preprocessing version: "
                f"{cfg.preprocessing_version}"
            ),
        )

    if cfg.pipeline_version not in config.SUPPORTED_PIPELINE_VERSIONS:
        raise AIExecutionError(
            error_code="UNSUPPORTED_PIPELINE_VERSION",
            safe_message=(
                f"Unsupported pipeline version: {cfg.pipeline_version}"
            ),
        )

    if cfg.prompt_version not in config.SUPPORTED_PROMPT_VERSIONS:
        raise AIExecutionError(
            error_code="UNSUPPORTED_PROMPT_VERSION",
            safe_message=f"Unsupported prompt version: {cfg.prompt_version}",
        )

    configured_slots = [
        slot for slot in cfg.stt_slots if slot is not None
    ]

    if cfg.pipeline == "separate":
        if not configured_slots:
            raise AIExecutionError(
                error_code="INVALID_PIPELINE_CONFIG",
                safe_message=(
                    "Separate pipeline requires at least one STT slot."
                ),
            )

        # در کد فعلی Gemini متنی برای Separate پیاده‌سازی نشده است.
        if client.is_gemini_model(cfg.llm_model):
            raise AIExecutionError(
                error_code="INVALID_PIPELINE_CONFIG",
                safe_message=(
                    "Gemini text mode is not implemented for "
                    "the separate pipeline."
                ),
            )

    elif cfg.pipeline == "hybrid":
        if not configured_slots:
            raise AIExecutionError(
                error_code="INVALID_PIPELINE_CONFIG",
                safe_message=(
                    "Hybrid pipeline requires at least one STT slot."
                ),
            )

    elif cfg.pipeline == "multimodal":
        if configured_slots:
            raise AIExecutionError(
                error_code="INVALID_PIPELINE_CONFIG",
                safe_message=(
                    "Multimodal pipeline must not contain STT slots."
                ),
            )


def build_runtime_stt_slots(
    cfg: ProcessingConfig,
    credentials: ExecutionCredentials,
) -> list[Optional[RuntimeSttSlotConfig]]:
    """Merge persistent STT config with temporary API credentials."""

    runtime_slots: list[Optional[RuntimeSttSlotConfig]] = []

    for slot in cfg.stt_slots:
        if slot is None:
            runtime_slots.append(None)
            continue

        credential = credentials.stt.get(slot.slot_id)

        runtime_slots.append(
            RuntimeSttSlotConfig(
                slot_id=slot.slot_id,
                model=slot.model,
                language=slot.language,
                base_url=(
                    credential.base_url
                    if credential and credential.base_url
                    else slot.base_url
                ),
                api_key=(
                    credential.api_key
                    if credential
                    else None
                ),
            )
        )

    return runtime_slots


def normalize_pipeline_result(
    raw_result: dict,
    cfg: ProcessingConfig,
    elapsed_seconds: float,
) -> AIJobResult:
    """Validate the current pipeline result and convert it to stable output."""

    stt_transcripts = {
        key: value
        for key, value in raw_result.items()
        if key.startswith("transcript_") and isinstance(value, str)
    }

    try:
        report = LLMReportOutput.model_validate(
            {
                "raw_transcript": raw_result.get("raw_transcript"),
                "corrected_transcript": raw_result.get(
                    "corrected_transcript"
                ),
                "final_text": raw_result.get("final_text"),
                "discrepancies_found": raw_result.get(
                    "discrepancies_found",
                    [],
                ),
                "notes": raw_result.get("notes"),
            }
        )

    except ValidationError as exc:
        raise AIExecutionError(
            error_code="INVALID_MODEL_OUTPUT",
            safe_message=(
                "The language model returned an invalid report structure."
            ),
            retryable=False,
        ) from exc

    return AIJobResult(
        stt_transcripts=stt_transcripts,
        raw_transcript=report.raw_transcript,
        corrected_transcript=report.corrected_transcript,
        final_text=report.final_text,
        discrepancies_found=report.discrepancies_found,
        notes=report.notes,
        model_metadata=ModelMetadata(
            pipeline=cfg.pipeline,
            stt_models=[
                slot.model
                for slot in cfg.stt_slots
                if slot is not None
            ],
            llm_model=cfg.llm_model,
            preprocessing_version=cfg.preprocessing_version,
            pipeline_version=cfg.pipeline_version,
            prompt_version=cfg.prompt_version,
            dictionary_version=cfg.dictionary_version,
        ),
        processing_metrics=ProcessingMetrics(
            total_seconds=elapsed_seconds,
        ),
    )


def execute_job(
    audio: bytes,
    audio_filename: str,
    processing_config: ProcessingConfig,
    credentials: Optional[ExecutionCredentials] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> AIJobResult:
    

    credentials = credentials or ExecutionCredentials()

    validate_processing_config(processing_config)

    if not audio:
        raise AIExecutionError(
            error_code="INVALID_AUDIO",
            safe_message="The uploaded audio file is empty.",
        )

    runtime_slots = build_runtime_stt_slots(
        processing_config,
        credentials,
    )

    llm_credential = credentials.llm

    llm_api_key = (
        llm_credential.api_key
        if llm_credential
        else None
    )

    llm_base_url = (
        llm_credential.base_url
        if llm_credential and llm_credential.base_url
        else processing_config.llm_base_url
    )

    def report_progress(stage: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, percent)

    started_at = time.perf_counter()

    report_progress("validating_config", 5)

    try:
        if processing_config.pipeline == "separate":
            report_progress("transcribing", 20)

            raw_result = pipelines.run_separate(
                audio=audio,
                audio_filename=audio_filename,
                stt_slots=runtime_slots,
                language=processing_config.language,
                llm_model=processing_config.llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        elif processing_config.pipeline == "multimodal":
            report_progress("multimodal_processing", 30)

            raw_result = pipelines.run_multimodal(
                audio=audio,
                audio_filename=audio_filename,
                llm_model=processing_config.llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

        else:
            report_progress("hybrid_processing", 20)

            raw_result = pipelines.run_hybrid(
                audio=audio,
                audio_filename=audio_filename,
                stt_slots=runtime_slots,
                language=processing_config.language,
                llm_model=processing_config.llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )

    except AIExecutionError:
        raise

    except HTTPException as exc:
        raise AIExecutionError(
            error_code="AI_PIPELINE_FAILED",
            safe_message="The AI pipeline failed.",
            retryable=exc.status_code >= 500,
        ) from exc

    except MemoryError as exc:
        raise AIExecutionError(
            error_code="RESOURCE_EXHAUSTED",
            safe_message="The AI service does not have enough memory.",
            retryable=True,
        ) from exc

    except Exception as exc:
        lower_message = str(exc).lower()

        if "out of memory" in lower_message:
            raise AIExecutionError(
                error_code="GPU_OUT_OF_MEMORY",
                safe_message="The GPU does not have enough free memory.",
                retryable=True,
            ) from exc

        if "timeout" in lower_message:
            raise AIExecutionError(
                error_code="PROCESSING_TIMEOUT",
                safe_message="AI processing timed out.",
                retryable=True,
            ) from exc

        raise AIExecutionError(
            error_code="AI_PIPELINE_FAILED",
            safe_message="The AI pipeline failed unexpectedly.",
            retryable=True,
        ) from exc

    elapsed_seconds = time.perf_counter() - started_at

    report_progress("validating_output", 95)

    result = normalize_pipeline_result(
        raw_result=raw_result,
        cfg=processing_config,
        elapsed_seconds=elapsed_seconds,
    )

    report_progress("completed", 100)

    return result
