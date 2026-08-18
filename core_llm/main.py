"""HTTP layer for Core_LLM — a thin FastAPI wrapper around model.MANAGER.

The orchestrator (and other modules) call this service over HTTP instead of
importing Core_LLM directly. /chat and /chat_audio both route through the
SAME manager/registry (model.py) — served directly via `transformers`, not
Ollama (Ollama can't accept audio input at all, so there's no way to keep it
for the audio role; dropping it for the text role too means one unified
serving path instead of two, and a model loaded via one endpoint is already
warm for the other as long as the same registry key is requested).

Run:
    python main.py            # or: uvicorn main:app --host 0.0.0.0 --port 8001
Interactive docs at http://<host>:8001/docs
"""
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import config
from model import MANAGER
from schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(title="Core LLM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health():
    """Liveness check, and which model is currently loaded (if any)."""
    return HealthResponse(status="ok", model=MANAGER.loaded or config.DEFAULT_MODEL)


@app.get("/models")
def list_models():
    """List all registered local models, and which is loaded."""
    return {"available": MANAGER.available(), "loaded": MANAGER.loaded}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send chat messages (OpenAI format) and get the assistant's full reply."""
    key = req.model or config.DEFAULT_MODEL
    messages = [m.model_dump() for m in req.messages]
    try:
        reply = await run_in_threadpool(
            MANAGER.chat, key, messages,
            temperature=req.temperature, response_format=req.response_format,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # model load/generation error
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")
    return ChatResponse(model=key, reply=reply)


@app.post("/unload")
async def unload(model: str | None = None):
    """Unload the currently-loaded model, freeing its VRAM.

    `model` is accepted for API compatibility with callers that pass the
    model they were using, but is otherwise unused -- there's only ever one
    model loaded at a time now, so this always unloads whatever that is.
    """
    loaded = MANAGER.loaded
    await run_in_threadpool(MANAGER.unload)
    return {"status": "unloaded", "model": model or loaded}


@app.get("/chat_audio/models")
def chat_audio_models():
    """List the local AUDIO-CAPABLE models (a subset of GET /models), and which is loaded."""
    return {"available": MANAGER.available(audio_only=True), "loaded": MANAGER.loaded}


@app.post("/chat_audio")
async def chat_audio(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    text: str | None = Form(default=None),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.3),
):
    """Local multimodal chat: give an audio-capable model the audio directly,
    no STT step. `model` must be one of GET /chat_audio/models' available
    keys; defaults to config.DEFAULT_MODEL (only meaningful if that happens
    to be an audio-capable one -- otherwise pass `model` explicitly).
    """
    key = model or config.DEFAULT_MODEL
    audio_bytes = await file.read()
    audio_format = (file.filename or "").rsplit(".", 1)[-1].lower() or "wav"
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": text or ""}]
    try:
        reply = await run_in_threadpool(
            MANAGER.chat, key, messages, audio=audio_bytes, audio_format=audio_format,
            temperature=temperature,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Multimodal LLM error: {exc}")
    return {"model": key, "reply": reply}


@app.post("/chat_audio/unload")
async def chat_audio_unload():
    """Unload the currently-loaded model, freeing its VRAM (alias of POST /unload)."""
    loaded = MANAGER.loaded
    await run_in_threadpool(MANAGER.unload)
    return {"status": "unloaded", "model": loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
