"""Request/response shapes for the controller's HTTP API.

The organising idea is a line down the middle: **configuration is storable,
credentials are not.**

`ProcessingConfig` describes a job completely -- pipeline, models, versions --
and carries no secret, so the backend can store it, hash it, and put the hash
in an audit trail. `ExecutionCredentials` carries the API keys, arrives with
each request, and is never persisted anywhere.

That split is why there are two STT slot types. `SttSlotConfig` is the part
worth keeping; `RuntimeSttSlotConfig` is it plus a key, built per request and
dropped when the request ends.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

import config


def reveal(secret: SecretStr | None) -> str | None:
    """Unwrap a SecretStr for an outbound call."""
    return secret.get_secret_value() if secret is not None else None


class Pipeline(str, Enum):
    """How a recording gets turned into a report.

    SEPARATE   -- STT slots transcribe; an LLM reconciles the transcripts.
    MULTIMODAL -- no STT; an audio-capable LLM hears the recording itself.
    HYBRID     -- both: the LLM hears the recording AND gets the transcripts
                  as cross-check material.
    """
    SEPARATE = "separate"
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"

    @property
    def uses_stt(self) -> bool:
        return self is not Pipeline.MULTIMODAL

    @property
    def uses_audio_llm(self) -> bool:
        return self is not Pipeline.SEPARATE


# --- Configuration: storable, hashable, no secrets -------------------------
class SttSlotConfig(BaseModel):
    """One STT engine, as it can safely be written down.

    `slot_id` names the slot so credentials can be addressed to it without the
    two having to travel together.
    """
    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    language: Optional[str] = None  # overrides the job's language for this slot
    base_url: Optional[str] = None


class ProcessingConfig(BaseModel):
    """Everything one job needs, except the secrets.

    Immutable by intent: the backend stores this, `execution.calculate_config_hash`
    hashes it, and the hash goes into the result. Two results with the same hash
    ran under identical settings.
    """
    model_config = ConfigDict(extra="forbid")

    pipeline: Pipeline
    language: Optional[str] = "fa"
    stt_slots: list[Optional[SttSlotConfig]] = Field(
        default_factory=list, max_length=config.MAX_STT_SLOTS)

    llm_model: str = Field(min_length=1, max_length=200)
    llm_base_url: Optional[str] = None

    # Which implementation produced a result. Refused if the controller does
    # not implement the named version -- see config.SUPPORTED_*_VERSIONS.
    preprocessing_version: str = Field(min_length=1, max_length=100)
    pipeline_version: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    dictionary_version: Optional[str] = None

    @field_validator("stt_slots")
    @classmethod
    def slot_ids_are_unique(cls, slots):
        """Credentials are addressed by slot_id, so a duplicate would silently
        send one slot's key to another."""
        ids = [slot.slot_id for slot in slots if slot is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("STT slot IDs must be unique")
        return slots

    @property
    def active_slots(self) -> list[SttSlotConfig]:
        return [slot for slot in self.stt_slots if slot is not None]


# --- Credentials: per request, never stored --------------------------------
class ProviderCredential(BaseModel):
    """One provider's key, supplied for the duration of a single job."""
    model_config = ConfigDict(extra="forbid")

    api_key: Optional[SecretStr] = Field(default=None, repr=False)
    base_url: Optional[str] = None


class ExecutionCredentials(BaseModel):
    """The secrets for one job, keyed to the config that needs them."""
    model_config = ConfigDict(extra="forbid")

    llm: Optional[ProviderCredential] = None
    stt: dict[str, ProviderCredential] = Field(default_factory=dict)  # by slot_id


class RuntimeSttSlotConfig(SttSlotConfig):
    """A slot with its credential merged in, built per request and discarded."""
    api_key: Optional[SecretStr] = Field(default=None, repr=False)


@dataclass(frozen=True)
class LlmTarget:
    """The LLM to call for one job, with credentials already resolved.

    Internal only -- built in execution.py so pipelines don't thread three
    arguments around.
    """
    model: str
    api_key: str | None = None
    base_url: str | None = None


# --- Results ---------------------------------------------------------------
class LLMReportOutput(BaseModel):
    """What the LLM is required to have returned.

    Checked rather than trusted: a model that answers with prose, or with JSON
    missing `final_text`, becomes one clear error instead of a malformed
    response travelling onward.
    """
    model_config = ConfigDict(extra="forbid")

    raw_transcript: str
    corrected_transcript: str
    final_text: str
    discrepancies_found: list[Any] = Field(default_factory=list)
    notes: str | list[str] | None = None


class ModelMetadata(BaseModel):
    """Which configuration produced a result, repeated in the result itself so
    it survives being copied out of its request."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    pipeline: Pipeline
    stt_models: list[str]
    llm_model: str
    preprocessing_version: str
    pipeline_version: str
    prompt_version: str
    dictionary_version: Optional[str] = None


class ProcessingMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_seconds: float = Field(ge=0)


class AIJobResult(BaseModel):
    """The controller's stable output contract."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    stt_transcripts: dict[str, str] = Field(default_factory=dict)
    raw_transcript: str
    corrected_transcript: str
    final_text: str
    discrepancies_found: list[Any] = Field(default_factory=list)
    notes: str | list[str] | None = None
    model_metadata: ModelMetadata
    processing_metrics: ProcessingMetrics


class JobExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["completed"]
    config_hash: str
    result: AIJobResult


class ErrorDetail(BaseModel):
    """The body of every 4xx/5xx this service returns.

    `retryable` is the field that matters to a caller: it separates "this will
    never work" from "the GPU was busy, come back".
    """
    error_code: str
    message: str
    retryable: bool


# --- Evaluation passthrough ------------------------------------------------
class EvaluationRequest(BaseModel):
    """What POST /evaluate forwards to the evaluation service.

    The required fields are declared so the API docs are useful and an obvious
    mistake fails here rather than a hop later. Extra fields are allowed
    through untouched, so the evaluation service can grow its contract without
    a matching change in the controller.
    """
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    asset_id: str
    model: str
    hypothesis: str          # the pipeline's final_text
    reference: str           # the radiologist-verified text
    pipeline: str | None = None
    model_version: str | None = None
