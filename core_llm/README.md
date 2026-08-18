# Core_LLM Service

Wraps a set of local LLMs — served directly via `transformers`, **not
Ollama** — behind a small HTTP API. `model.py` holds the model registry and
`LLMManager` (one model loaded at a time, swapped as needed); `main.py`
exposes it over HTTP. Ollama was dropped entirely: it can't accept audio
input at all, so keeping it around just for the text role would have meant
two separate serving paths side by side. Now `/chat` (text) and `/chat_audio`
(audio-capable models only) share the same manager — load a model once via
either endpoint, and it's already warm for the other, as long as you keep
requesting the same registry key.

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
| `POST /chat` | body `{messages, model?, temperature?, response_format?}` -> `{model, reply}` |
| `POST /unload?model=` | unload the currently-loaded model, freeing its VRAM |
| `GET /chat_audio/models` | just the **audio-capable** model keys + which is loaded |
| `POST /chat_audio` | multipart: `file` (audio) + `system_prompt` + `text?` + `model?` + `temperature?` -> `{model, reply}` |
| `POST /chat_audio/unload` | alias of `POST /unload` |

`response_format`/`temperature` are accepted for interface parity with the
old Ollama-backed API. `temperature` <= 0.01 forces greedy decoding (the
default, 0.3, is deliberately low — medical use wants consistency over
creativity). There's no local equivalent of Ollama/OpenAI's JSON mode for
`response_format`; it's accepted but not enforced — rely on the system
prompt asking for JSON and the caller's own tolerant parsing (Orchestrator's
`_extract_json()` already does this).

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
