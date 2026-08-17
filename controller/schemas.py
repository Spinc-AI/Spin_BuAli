from typing import Optional

from pydantic import BaseModel


class SttSlotConfig(BaseModel):
    """One STT engine's config: local model name, or "openai:<model>" for cloud."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    language: Optional[str] = None  # overrides the session/run language for this slot only


class SessionRequest(BaseModel):
    llm_model: str                      # local model name, or "openai:<model>"/"gemini:<model>"
    language: Optional[str] = None      # e.g. "fa" or "en" — default STT language
    stt_api_key: Optional[str] = None
    stt_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    stt_slots: Optional[list[Optional[SttSlotConfig]]] = None  # up to 3 independent STT engine configs
    pipeline: str = "separate"          # "separate" | "multimodal" | "hybrid"


class Session(BaseModel):
    llm_model: str
    language: Optional[str] = None
    stt_api_key: Optional[str] = None
    stt_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    stt_slots: Optional[list[Optional[SttSlotConfig]]] = None
    pipeline: str = "separate"
    stt_ready: bool = False
    llm_ready: bool = False
