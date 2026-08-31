# BuAli Controller

The entry point for the whole system, and the only service callers address
directly. It turns a spoken radiology report into a corrected report through
one of three pipelines, and forwards scoring requests to `evaluation/`.

The pipeline logic is written directly in Python (`pipelines.py`) rather than
driven by a generic instruction engine — same behaviour, no abstraction layer
to read through first.

## Files

| File | Responsibility |
|---|---|
| `main.py` | HTTP routes, validation, and the active session |
| `pipelines.py` | The three pipelines and the order they run in |
| `prompts.py` | System prompts and the JSON output template |
| `providers.py` | Model routing (local / OpenAI / Gemini), credentials, accepted audio formats |
| `stt_client.py` | HTTP client for the STT service |
| `llm_client.py` | HTTP client for the language model (all three providers) |
| `evaluation_client.py` | HTTP client for the evaluation service |
| `schemas.py` | Request and response shapes |

## Pipelines

| Pipeline | Behaviour |
|---|---|
| `separate` (default) | Up to 3 independent STT engines transcribe the audio; an LLM reconciles the transcripts. |
| `multimodal` | No STT at all — the audio goes straight to an audio-capable LLM. |
| `hybrid` | Both: the STT slots run **and** the LLM hears the audio itself, with the transcripts passed as reference material rather than ground truth. |

`multimodal` is `hybrid` with no STT slots configured, so the two share one code
path instead of having a function each.

## Choosing a model

The provider is carried in the model name (see `providers.py`):

| Prefix | Goes to |
|---|---|
| *(none)* | the local service in this repo |
| `openai:<model>` | any OpenAI-compatible API |
| `gemini:<model>` | Gemini's native generateContent API |

All three providers work in all three pipelines. For `multimodal` and `hybrid`
the model must be able to accept audio.

## Run

```bash
pip install -r requirements.txt
python main.py          # or: run.bat (Windows) / ./run.sh (Linux)
```

Binds to `0.0.0.0:9002`. Interactive docs at `/docs`.

## API

| Method and path | Purpose |
|---|---|
| `GET /` | Service health, plus whether STT / LLM / evaluation are reachable |
| `GET /models` | Proxy for the local STT model registry |
| `GET /llm/models` | Proxy for the local LLM registry, and its audio-capable subset |
| `GET /languages` | Proxy for the supported language codes |
| `GET /status` | The active session (API keys are never returned) |
| `POST /session` | Body: `{llm_model, pipeline?, language?, stt_slots?, llm_api_key?, llm_base_url?}` |
| `POST /run` | multipart: `file` (audio, required) plus optional overrides: `language`, `llm_api_key`, `llm_base_url`, `stt_slots_json` |
| `POST /session/unload` | Free the models and end the session |
| `POST /evaluate` | Score a transcript against a reference — passed through to `evaluation/` |

The controller is the only door into the system: `stt/`, `core_llm/` and
`evaluation/` are never called directly. `POST /evaluate` forwards the body
without interpreting it and returns the reply as received, so the metric
contract stays owned by the module that implements it.

STT credentials belong to each slot (`stt_slots[].api_key` / `.base_url`), not
to the session.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

The STT, LLM and evaluation services are stubbed — no model is loaded and no
network request is made.

## Examples

```bash
curl -X POST http://localhost:9002/session -H "Content-Type: application/json" \
  -d '{"pipeline": "separate", "llm_model": "aya-expanse-8b",
       "stt_slots": [{"model": "whisper"}, {"model": "openai:whisper-1"}]}'

curl -X POST http://localhost:9002/run -F "file=@report.wav"
```
