# STT Service

A small HTTP backend that loads a Persian speech-to-text model into memory and
transcribes uploaded audio. The model is chosen and loaded **before** any audio
is sent, so transcription requests never pay the loading cost.

## Models

Ordered best -> worst by clinic-realistic-noise WER, per
[`../benchmark/benchmark_summary.pdf`](../benchmark/benchmark_summary.pdf). This
is also the order `GET /models` and the UI dropdowns present them in — don't
reorder `MODEL_REGISTRY` without re-checking the benchmark's ranking.

| # | key                      | model                                                                  | clinic WER | notes |
|---|--------------------------|-------------------------------------------------------------------------|-----------|-------|
| 1 | `seamless`               | `facebook/seamless-m4t-v2-large`                                       | 0.315     | best overall; ~11-15 GB VRAM |
| 2 | `whisper`                | `nezamisafa/whisper-persian-v4`                                        | 0.484     | lowest hallucination (0.4%); ~3.4 GB VRAM — value pick |
| 3 | `seamless-medium`        | `facebook/hf-seamless-m4t-medium`                                      | 0.495     | SeamlessM4T v1, not v2 |
| 4 | `whisper-halakoo`        | `MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch` | 0.601  | |
| 5 | `mms-all`                | `facebook/mms-1b-all`                                                  | 0.610     | CTC, never hallucinates, but drops speech under noise |
| 6 | `mms-fl102`              | `facebook/mms-1b-fl102`                                                | 0.623     | in-domain caveat per benchmark |
| 7 | `whisper-vhdm`           | `vhdm/whisper-large-fa-v1`                                             | 0.638     | ⚠ hallucinates under noise — avoid for clinical use |
| 8 | `wav2vec2-xlsr53`        | `jonatasgrosman/wav2vec2-large-xlsr-53-persian`                        | 0.676     | CTC, fails safely (drops speech, doesn't fabricate) |
| 9 | `whisper-large-v3`       | `openai/whisper-large-v3`                                              | 0.716     | hallucinates once mic/codec degradation hits |
| 10| `whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo`                                        | 1.088     | ⚠ worst — collapses by fabrication under noise, avoid for clinical use |

Only one model is held in memory at a time. Loading a model unloads the previous one.

## Run

```bash
pip install -r requirements.txt
./run.sh          # Linux;  run.bat on Windows
```

> Install the **CUDA build** of `torch` / `torchaudio` to use the GPU, otherwise
> they fall back to CPU. See https://pytorch.org for the right install command.

The server binds to `0.0.0.0:8000` — that means "listen on all interfaces," **not** an
address you connect to. Use the base URL for where you're calling from:

| Calling from…                         | Base URL                     |
|---------------------------------------|------------------------------|
| the same machine (local test)         | `http://localhost:8000`      |
| another machine → the deployed server | `http://193.93.169.134:8000` |

Add `/docs` to either for the interactive API docs.

## API contract

### `GET /models`
List available models and report which one is loaded.

```json
{ "available": ["seamless", "whisper", "seamless-medium", "whisper-halakoo", "mms-all", "mms-fl102", "whisper-vhdm", "wav2vec2-xlsr53", "whisper-large-v3", "whisper-large-v3-turbo"], "loaded": null }
```

### `POST /models/{key}/load`
Load a model into memory. Returns once the model is ready. Call this first.

```json
{ "status": "ready", "model": "whisper" }
```

- `404` if `key` is not a known model.

### `GET /languages`
List language codes accepted by `POST /transcribe`.

```json
{ "available": { "fa": "Persian", "en": "English" }, "default": "fa" }
```

### `POST /transcribe`
Transcribe an audio file with the currently loaded model.

- Request: `multipart/form-data`, field `file` = an audio file (e.g. wav),
  optional field `language` = a code from `GET /languages` (defaults to `fa`).
- Response:

```json
{ "model": "whisper", "language": "fa", "text": "..." }
```

- `409` if no model has been loaded yet — call `POST /models/{key}/load` first.
- `400` if the audio file cannot be decoded, or `language` is unknown.

> The Persian-fine-tuned `whisper` model can still transcribe English on a
> best-effort basis (the language token is forced, the weights are not
> English-optimized) — accuracy will be lower than a native English model.

### `POST /models/unload`
Unload the current model and free its memory.

```json
{ "status": "unloaded", "loaded": null }
```

## Typical sequence

Base URL: `http://193.93.169.134:8000`

```bash
# 1. see available models / which is loaded
curl http://193.93.169.134:8000/models

# 2. load a model (wait for {"status":"ready"})
curl -X POST http://193.93.169.134:8000/models/whisper/load

# 3. transcribe an audio file (language defaults to fa; add -F "language=en" for English)
curl -X POST http://193.93.169.134:8000/transcribe -F "file=@clip.wav" -F "language=en"
```

## Adding a model

1. Add a `BaseSTTModel` subclass in `app/model.py`.
2. Register it in `_MODEL_TYPES`.
3. Add an entry to `MODEL_REGISTRY` in `app/config.py`.

No changes to the API layer are required.
