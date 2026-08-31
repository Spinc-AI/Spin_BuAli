"""Request/response shapes for the controller's HTTP API.

API keys are `SecretStr`, so they never appear in a response body or a log
line by accident — call `reveal()` to get the raw value at the point of use.
"""
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, SecretStr

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


class SttSlotConfig(BaseModel):
    """One STT engine: a local model name, or "openai:<model>" for the cloud."""
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    language: str | None = None  # overrides the session/run language for this slot


class SessionConfig(BaseModel):
    """What POST /session accepts, and what GET /status reports back."""
    pipeline: Pipeline = Pipeline.SEPARATE
    llm_model: str  # local model name, or "openai:<model>" / "gemini:<model>"
    language: str | None = None  # default STT language, e.g. "fa" or "en"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    stt_slots: list[SttSlotConfig | None] | None = Field(
        default=None, max_length=config.MAX_STT_SLOTS
    )

    @property
    def active_slots(self) -> list[SttSlotConfig]:
        """The configured slots, with the unused (null) ones dropped."""
        return [slot for slot in self.stt_slots or [] if slot is not None]


@dataclass(frozen=True)
class LlmTarget:
    """The LLM to call for one run, with credentials already resolved.

    Internal only -- built in main.py from the session plus any per-run
    overrides, so pipelines don't have to thread three arguments around.
    """
    model: str
    api_key: str | None = None
    base_url: str | None = None


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
