'''HTTP layer: endpoints for listing, loading and running models.'''

import io
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.model import ModelManager
from app.schemas import LanguagesResponse, LoadResponse, ModelInfo, TranscriptionResult


# ============================================================
# Application setup
# ============================================================

manager = ModelManager()


@asynccontextmanager
async def lifespan(app):
    '''Optionally warm up a default model before the server accepts traffic.'''
    if config.PRELOAD_MODEL:
        await run_in_threadpool(manager.load, config.PRELOAD_MODEL)
    yield


app = FastAPI(title="STT Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helpers
# ============================================================

def _read_audio(raw):
    '''Decode raw audio bytes into a mono float32 array and its sample rate.'''
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, np.float32), int(sr)


# ============================================================
# Routes
# ============================================================

@app.get("/models", response_model=ModelInfo)
async def list_models():
    '''List available models and report which one is loaded.'''
    return ModelInfo(available=manager.available(), loaded=manager.loaded)


@app.post("/models/{key}/load", response_model=LoadResponse)
async def load_model(key: str):
    '''Load the requested model into memory, replacing any loaded model.'''
    if key not in manager.available():
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    await run_in_threadpool(manager.load, key)
    return LoadResponse(status="ready", model=key)


@app.post("/models/unload")
async def unload_model():
    """Unload the current model and free its memory."""
    await run_in_threadpool(manager.unload)
    return {"status": "unloaded", "loaded": manager.loaded}


@app.get("/languages", response_model=LanguagesResponse)
async def list_languages():
    '''List language codes accepted by POST /transcribe's `language` field.'''
    return LanguagesResponse(available=config.SUPPORTED_LANGUAGES, default=config.DEFAULT_LANGUAGE)


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = File(...), language: str = Form(default=None)):
    '''Transcribe an uploaded audio file with the currently loaded model.

    `language` is one of GET /languages' `available` codes (e.g. "fa", "en").
    Defaults to config.DEFAULT_LANGUAGE if omitted.
    '''
    if manager.loaded is None:
        raise HTTPException(
            status_code=409,
            detail="no model loaded — call POST /models/{key}/load first",
        )
    lang = language or config.DEFAULT_LANGUAGE
    if lang not in config.SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unknown language '{lang}'")
    raw = await file.read()
    try:
        audio, sr = _read_audio(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode audio file")
    text = await run_in_threadpool(manager.transcribe, audio, sr, language=lang)
    return TranscriptionResult(model=manager.loaded, language=lang, text=text)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)
