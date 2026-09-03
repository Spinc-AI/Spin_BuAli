# Spin BuAli

Turns a spoken radiology report into a corrected written report. Five services
live in this repo and talk to each other over HTTP — a single `git clone` runs
the whole system, with no dependency on any other project.

## How the pieces fit together

```
                    ┌──────────────┐
   your website ───▶│  controller  │ :9002   the only entry point
   / demo_app       └──────┬───────┘
                           │ HTTP
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌──────────────┐
   │    stt     │   │  core_llm  │   │  evaluation  │
   │   :8000    │   │   :8001    │   │    :8002     │
   └────────────┘   └────────────┘   └──────────────┘
    speech → text   text/audio → text     scoring
```

Nothing behind the controller is addressed directly. Callers reach speech
recognition, the language model and scoring through it, so there is one door
into the system and one place where authentication, validation and credentials
live.

The controller is **stateless**: each request carries its own configuration and
credentials, and nothing is kept between them.

## Modules

| Folder | Role | Port |
|---|---|---|
| [`stt/`](stt/README.md) | Speech recognition (10 Persian models) | `8000` |
| [`core_llm/`](core_llm/README.md) | Language model, including the audio-capable path | `8001` |
| [`evaluation/`](evaluation/README.md) | Scores a transcript against a verified reference | `8002` |
| [`controller/`](controller/README.md) | BuAli's brain: the three pipelines, prompts, auth | `9002` |
| [`preprocessing/`](preprocessing/README.md) | Audio validation, standardization, VAD and chunking | — |
| [`demo_app/`](demo_app/README.md) | Tkinter desktop client for manual testing | — |
| [`benchmark/`](benchmark/README.md) | A self-contained Kaggle notebook ranking STT models | — |
| [`docs/`](docs/README.md) | Roadmap and reference documents | — |

Each service folder is independently deployable: its own `requirements.txt`,
its own `config.py`, and no direct imports from any other module. Moving one to
a different machine means changing a URL and nothing else.

`benchmark/` is not a service at all: it is a single Kaggle notebook, with no
port and never in the request path. It embeds `evaluation/`'s metric code
verbatim so a model comparison is scored by exactly what runs in production —
see [its README](benchmark/README.md#about-the-code-in-section-2).

## Quick start

One terminal per service. Set `INTERNAL_API_TOKEN` in `controller/.env` first —
the controller authenticates every call and refuses to serve without it:

```bash
# terminal 1 — STT
cd stt && pip install -r requirements.txt && python -m app.main

# terminal 2 — Core_LLM
cd core_llm && pip install -r requirements.txt && python main.py

# terminal 3 — evaluation
cd evaluation && pip install -r requirements.txt && python main.py

# terminal 4 — the BuAli controller
cd controller && pip install -r requirements.txt && python main.py

# terminal 5 — the demo client
cd demo_app && pip install -r requirements.txt && python app.py
```

Everything binds to `localhost` on the ports above. The `multimodal` and
`hybrid` pipelines need an audio-capable model (`gemma-4-e4b`, `gemma-4-12b`,
`qwen3-omni-30b`) in `core_llm/`, which wants serious VRAM — or use a cloud
model (`openai:` / `gemini:`) instead. The first run of any model downloads its
weights from Hugging Face; that is slow once, then cached.

> To try the controller with a cloud model only, `stt/` and `core_llm/` are not
> needed: the `multimodal` pipeline with a `gemini:` or `openai:` model calls no
> local service.

## The three pipelines

| Pipeline | Audio to the LLM | STT transcripts |
|---|---|---|
| `separate` | ❌ | ✅ up to 3 engines, then an LLM reconciles them |
| `multimodal` | ✅ | ❌ |
| `hybrid` | ✅ | ✅ passed as reference material, not ground truth |

Model routing and the local/cloud prefix convention are documented in
[controller/README.md](controller/README.md).

## Tests

```bash
cd controller && python -m pytest tests/     # auth, jobs, pipelines (no network, no models)
cd evaluation && python -m pytest tests/     # normalisation and the clinical metrics
cd demo_app   && python -m pytest tests/     # controller client and Word export
cd stt        && python -m pytest tests/     # requires torch
```
