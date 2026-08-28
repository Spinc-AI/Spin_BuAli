# from typing import Optional

# from pydantic import BaseModel


# class SttSlotConfig(BaseModel):
#     """One STT engine's config: local model name, or "openai:<model>" for cloud."""
#     model: str
#     api_key: Optional[str] = None
#     base_url: Optional[str] = None
#     language: Optional[str] = None  # overrides the session/run language for this slot only


# class SessionRequest(BaseModel):
#     llm_model: str                      # local model name, or "openai:<model>"/"gemini:<model>"
#     language: Optional[str] = None      # e.g. "fa" or "en" — default STT language
#     stt_api_key: Optional[str] = None
#     stt_base_url: Optional[str] = None
#     llm_api_key: Optional[str] = None
#     llm_base_url: Optional[str] = None
#     stt_slots: Optional[list[Optional[SttSlotConfig]]] = None  # up to 3 independent STT engine configs
#     pipeline: str = "separate"          # "separate" | "multimodal" | "hybrid"


# class Session(BaseModel):
#     llm_model: str
#     language: Optional[str] = None
#     stt_api_key: Optional[str] = None
#     stt_base_url: Optional[str] = None
#     llm_api_key: Optional[str] = None
#     llm_base_url: Optional[str] = None
#     stt_slots: Optional[list[Optional[SttSlotConfig]]] = None
#     pipeline: str = "separate"
#     stt_ready: bool = False
#     llm_ready: bool = False




from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


PipelineType = Literal["separate", "multimodal", "hybrid"]


LanguageType = Literal["fa", "en"]


class SttSlotConfig(BaseModel):
    """Persistent, non-secret configuration of one STT engine."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    language: Optional[LanguageType] = None
    base_url: Optional[str] = None


class RuntimeSttSlotConfig(SttSlotConfig):
    """STT configuration used only while executing one Job."""

    api_key: Optional[str] = Field(default=None, repr=False)


class ProviderCredential(BaseModel):
    """Temporary credential sent by Backend during execution."""

    model_config = ConfigDict(extra="forbid")

    api_key: Optional[str] = Field(default=None, repr=False)
    base_url: Optional[str] = None


class ExecutionCredentials(BaseModel):
    """Temporary secrets that must not be stored in Job configuration."""

    model_config = ConfigDict(extra="forbid")

    llm: Optional[ProviderCredential] = None

    # Key is slot_id, for example: stt_1
    stt: dict[str, ProviderCredential] = Field(default_factory=dict)


class ProcessingConfig(BaseModel):
    """Immutable configuration snapshot for one Job."""

    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineType
    language: Optional[LanguageType] = "fa"

    stt_slots: list[Optional[SttSlotConfig]] = Field(
        default_factory=list,
        max_length=3,
    )

    llm_model: str = Field(min_length=1, max_length=200)
    llm_base_url: Optional[str] = None

    preprocessing_version: str = Field(min_length=1, max_length=100)
    pipeline_version: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    dictionary_version: Optional[str] = None

    @field_validator("stt_slots")
    @classmethod
    def validate_slot_ids(
        cls,
        slots: list[Optional[SttSlotConfig]],
    ) -> list[Optional[SttSlotConfig]]:
        ids = [slot.slot_id for slot in slots if slot is not None]

        if len(ids) != len(set(ids)):
            raise ValueError("STT slot IDs must be unique")

        return slots


class LLMReportOutput(BaseModel):
    """Validated structure expected from the LLM."""

    model_config = ConfigDict(extra="forbid")

    raw_transcript: str
    corrected_transcript: str
    final_text: str
    discrepancies_found: list[Any] = Field(default_factory=list)
    notes: str | list[str] | None = None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineType
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
    model_config = ConfigDict(extra="forbid")

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
