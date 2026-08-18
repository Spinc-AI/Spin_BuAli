"""Pydantic request/response models for the Core_LLM HTTP API."""
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """One message in the standard OpenAI chat format."""
    role: str       # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None        # falls back to config.DEFAULT_MODEL
    temperature: float = 0.3        # low default — medical use wants consistency
    response_format: dict | None = None  # e.g. {"type": "json_object"} for JSON mode


class ChatResponse(BaseModel):
    model: str
    reply: str


class HealthResponse(BaseModel):
    status: str
    model: str
