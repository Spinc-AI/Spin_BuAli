# Core_LLM Service

Wraps a set of local LLMs — served directly via `transformers`, **not
Ollama** — behind a small HTTP API. `model.py` holds the model registry and
`LLMManager` (one model loaded at a time, swapped as needed); `main.py`
exposes it over HTTP.

`/chat` (text) and `/chat_audio` (audio-capable models only) share the same
manager, so a model loaded through either endpoint is already warm for the
other, as long as you keep requesting the same registry key. Ollama was
dropped because it can't accept audio input at all, which would have meant
running two serving paths side by side.

| File | Holds |
|---|---|
| `config.py` | ports, model IDs, device placement (all overridable via `.env`) |
| `model.py` | the `BaseLLM` subclasses, `MODEL_REGISTRY`, and `LLMManager` |
| `main.py` | the HTTP routes |
| `schemas.py` | request/response shapes |
| `chat.py` | terminal smoke test — loads a model with no HTTP server involved |

## Run
```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```
Serves on `0.0.0.0:8001` (docs at `/docs`). No model loads at startup — the
first request for a given key downloads it from Hugging Face and loads it
into VRAM (slow the first time, fast after). See `.env.example` to override
any model ID or the default registry key.

## Models

| `model` key | Model | Role |
|---|---|---|
| `aya-expanse-8b` (default) | `CohereLabs/aya-expanse-8b` | Text only |
| `aya-expanse-32b` | `CohereLabs/aya-expanse-32b` | Text only, bigger |
| `gemma-4-31b` | `google/gemma-4-31B-it` | Text only — Gemma 4's strongest model overall, but 26B-A4B/31B have **no audio input** |
| `gemma-4-e4b` | `google/gemma-4-E4B-it` | Text **and audio** — lighter, faster |
| `gemma-4-12b` | `google/gemma-4-12B-it` | Text **and audio** — largest audio-capable Gemma 4 |
| `qwen3-omni-30b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Text **and audio** — **best tested option for Persian audio**, confirmed via [PARSA-Bench](https://arxiv.org/html/2603.14456) (0.358 WER vs. 6-9 for Gemma-3n-class models). MoE, ~3B active params, but full weights are much larger — needs real VRAM headroom. |

`CohereLabs/aya-expanse-*` is **CC-BY-NC** (non-commercial) — revisit before
any commercial/clinical use. The rest are Apache 2.0.

## API
| Method & path | Purpose |
|---|---|
| `GET /` | health + which model is currently loaded |
| `GET /models` | all registered model keys + which is loaded |
| `GET /chat_audio/models` | just the **audio-capable** model keys + which is loaded |
| `POST /chat` | body `{messages, model?, temperature?}` -> `{model, reply}` |
| `POST /chat_audio` | multipart: `file` (audio) + `system_prompt` + `text?` + `model?` + `temperature?` -> `{model, reply}` |
| `POST /unload` | unload the currently-loaded model, freeing its VRAM |

`temperature` <= 0.01 forces greedy decoding. The default, 0.3, is
deliberately low — medical use wants consistency over creativity.

**There is no JSON mode.** Nothing here reproduces OpenAI's
`response_format`, so a caller that needs JSON asks for it in the system
prompt and parses the reply tolerantly — which is what the controller's
`llm_client.extract_json()` does.

## Examples
```bash
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'

curl http://localhost:8001/chat_audio/models

curl -X POST http://localhost:8001/chat_audio \
  -F "file=@report.wav" \
  -F "system_prompt=Transcribe this radiology report as JSON." \
  -F "text=Optional extra instructions" \
  -F "model=qwen3-omni-30b"
```

Passing a text-only model's key to `/chat_audio` (e.g. `model=aya-expanse-8b`)
returns a clean 502 explaining it can't accept audio, rather than a confusing
failure deeper in the pipeline.
