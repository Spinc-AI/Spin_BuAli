"""Request/response shapes for the Core_LLM HTTP API."""
from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    """One message in the standard OpenAI chat format."""
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None  # falls back to config.DEFAULT_MODEL
    temperature: float = 0.3  # low by default -- medical use wants consistency


class ChatResponse(BaseModel):
    model: str
    reply: str

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: str
    model: str

    model_config = ConfigDict(protected_namespaces=())
