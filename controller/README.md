# BuAli Controller

The entry point for the whole system, and the only service callers address
directly. It turns a spoken radiology report into a corrected report through
one of three pipelines, and forwards scoring requests to `evaluation/`.

**Stateless.** Every request carries everything it needs — the recording, the
configuration and the credentials for that one call. Nothing is remembered
between requests, so two callers cannot interfere with each other and a restart
loses nothing.

The pipeline logic is written directly in Python (`pipelines.py`) rather than
driven by a generic instruction engine — same behaviour, no abstraction layer
to read through first.

## Files

| File | Responsibility |
|---|---|
| `main.py` | HTTP routes, authentication, upload limits |
| `execution.py` | Runs one job: validation, credentials, error classification |
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
| `POST /internal/jobs/{job_id}/execute` | Run one recording. multipart: `file`, `config_json`, `credentials_json?` |
| `POST /internal/admin/models/unload` | Free the local services' weights |
| `POST /evaluate` | Score a transcript against a reference — passed through to `evaluation/` |

`job_id` belongs to the caller. The controller neither creates nor stores it;
it is echoed back so a result can be matched to the request that asked for it.

The controller is the only door into the system: `stt/`, `core_llm/` and
`evaluation/` are never called directly. `POST /evaluate` forwards the body
without interpreting it and returns the reply as received, so the metric
contract stays owned by the module that implements it.

## Authentication

Every route except `GET /` requires `X-Internal-Token`, compared in constant
time against `INTERNAL_API_TOKEN`. An unset token fails **closed** with 503 —
no secret configured is a deployment mistake, and treating it as "no check
needed" would leave the service open at exactly the moment nobody noticed.

`GET /` is deliberately open: a load balancer has no token, and it reveals
nothing beyond which services are up.

## Configuration and credentials are separate

This is the idea the schemas are built around.

| `ProcessingConfig` — `config_json` | `ExecutionCredentials` — `credentials_json` |
|---|---|
| pipeline, models, language, versions | API keys and base URLs |
| **no secrets, ever** | secrets only |
| stored and hashed by the caller | used for one call, never written down |

So the backend can keep a job's configuration, hash it, and put that hash in an
audit trail without ever holding an API key. STT credentials are addressed to a
slot by `slot_id`, which is why slot ids must be unique.

`config_hash` is a SHA-256 over the canonical JSON of `ProcessingConfig` and is
returned with every result: two results with the same hash ran under identical
settings.

## Version pinning

Every job declares `preprocessing_version`, `pipeline_version` and
`prompt_version`. Anything this build does not implement is refused. Without
it, the same request quietly means something different after a deploy and a
stored result cannot be traced to the code that produced it.

Configure the accepted sets with `SUPPORTED_*_VERSIONS`; `GET /` lists them.

## Errors

Every failure body is the same three fields:

```json
{"error_code": "GPU_OUT_OF_MEMORY", "message": "The GPU has insufficient free memory.", "retryable": true}
```

`retryable` decides the status: **503** when trying again could work, **422**
when the request itself is the problem. The message is deliberately plain — a
caller gets a code and a sentence, never a stack trace or an upstream address.

| Code | |
|---|---|
| `UNAUTHORIZED_INTERNAL_CALL` / `INTERNAL_AUTH_NOT_CONFIGURED` | 401 / 503 |
| `INVALID_PROCESSING_CONFIG` / `INVALID_EXECUTION_CREDENTIALS` | 400, malformed JSON |
| `UNSUPPORTED_{PREPROCESSING,PIPELINE,PROMPT}_VERSION` | 422 |
| `INVALID_PIPELINE_CONFIG` | 422, e.g. `separate` with no STT slot |
| `INVALID_AUDIO` / `AUDIO_TOO_LARGE` | 422 / 413 |
| `INVALID_MODEL_OUTPUT` | 422, the LLM returned no usable report |
| `GPU_OUT_OF_MEMORY` / `RESOURCE_EXHAUSTED` / `PROCESSING_TIMEOUT` | 503 |
| `AI_PIPELINE_FAILED` | 503 |

Uploads are capped at `MAX_UPLOAD_BYTES` (50 MB by default), enforced **while
reading** — an oversized file is refused before it is held in memory, not
after.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

The STT, LLM and evaluation services are stubbed — no model is loaded and no
network request is made.

## Examples

```bash
curl -X POST http://localhost:9002/internal/jobs/job-001/execute   -H "X-Internal-Token: $INTERNAL_API_TOKEN"   -F "file=@report.mp3"   -F 'config_json={
        "pipeline": "separate",
        "language": "fa",
        "llm_model": "aya-expanse-8b",
        "stt_slots": [{"slot_id": "stt_1", "model": "whisper"},
                      {"slot_id": "stt_2", "model": "openai:whisper-1"}],
        "preprocessing_version": "legacy-v1",
        "pipeline_version": "buali-v1",
        "prompt_version": "radiology-v1"}'   -F 'credentials_json={"stt": {"stt_2": {"api_key": "sk-..."}}}'
```

The response carries the report, each slot's transcript, the configuration that
ran, and the hash of it:

```json
{"job_id": "job-001", "status": "completed", "config_hash": "bfad5fff...",
 "result": {"stt_transcripts": {"transcript_1": "...", "transcript_2": "..."},
            "final_text": "...",
            "model_metadata": {"pipeline": "separate", "prompt_version": "radiology-v1"},
            "processing_metrics": {"total_seconds": 12.4}}}
```
