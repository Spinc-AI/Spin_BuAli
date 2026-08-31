"""HTTP layer for Core_LLM -- a thin FastAPI wrapper around model.MANAGER.

Callers reach this service over HTTP rather than importing Core_LLM directly.
/chat and /chat_audio share one manager, so a model loaded through either is
already warm for the other.

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


async def _generate(**kwargs) -> str:
    """Run one generation off the event loop, mapping failures to HTTP codes."""
    try:
        return await run_in_threadpool(MANAGER.chat, **kwargs)
    except KeyError as exc:  # unknown registry key
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # model load or generation failure
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")


@app.get("/", response_model=HealthResponse)
async def health():
    """Liveness check, and which model is currently loaded."""
    return HealthResponse(status="ok", model=MANAGER.loaded or config.DEFAULT_MODEL)


@app.get("/models")
def list_models():
    """Every registered model, and which one is loaded."""
    return {"available": MANAGER.available(), "loaded": MANAGER.loaded}


@app.get("/chat_audio/models")
def list_audio_models():
    """Just the audio-capable models -- a subset of GET /models."""
    return {"available": MANAGER.available(audio_only=True), "loaded": MANAGER.loaded}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send chat messages (OpenAI format) and get the assistant's full reply."""
    key = request.model or config.DEFAULT_MODEL
    reply = await _generate(
        key=key,
        messages=[message.model_dump() for message in request.messages],
        temperature=request.temperature,
    )
    return ChatResponse(model=key, reply=reply)


@app.post("/chat_audio")
async def chat_audio(
    file: UploadFile = File(...),
    system_prompt: str = Form(...),
    text: str | None = Form(default=None),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.3),
):
    """Give an audio-capable model the recording directly, with no STT step.

    `model` must be one of GET /chat_audio/models' keys. It defaults to
    config.DEFAULT_MODEL, which is only useful if that happens to be
    audio-capable -- otherwise pass it explicitly.
    """
    key = model or config.DEFAULT_MODEL
    reply = await _generate(
        key=key,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": text or ""}],
        audio=await file.read(),
        audio_format=(file.filename or "").rsplit(".", 1)[-1].lower() or "wav",
        temperature=temperature,
    )
    return {"model": key, "reply": reply}


@app.post("/unload")
async def unload():
    """Unload the loaded model, freeing its VRAM."""
    loaded = MANAGER.loaded
    await run_in_threadpool(MANAGER.unload)
    return {"status": "unloaded", "model": loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
