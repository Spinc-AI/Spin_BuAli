'''Request and response shapes exposed by the API.'''

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


# ============================================================
# Responses
# ============================================================

class ModelInfo(BaseModel):
    '''Available model keys and the one currently held in memory.'''

    available: List[str]
    loaded: Optional[str]


class LanguagesResponse(BaseModel):
    '''Language codes accepted by POST /transcribe, mapped to a display name.'''

    available: Dict[str, str]
    default: str


class LoadResponse(BaseModel):
    '''Result of a model load request.'''

    status: str
    model: str

    model_config = ConfigDict(protected_namespaces=())


class TranscriptionResult(BaseModel):
    '''Transcribed text together with the model and language used.'''

    model: str
    language: str
    text: str

    model_config = ConfigDict(protected_namespaces=())
