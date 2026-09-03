# BuAli Demo App

A Tkinter desktop client for the BuAli controller, for testing by hand. One
scrollable window: pick a pipeline (Separate / Multimodal / Hybrid), configure
up to 3 independent STT slots, choose an LLM (local or cloud), load or record
audio, run it, and save the finished report as a Word document.

## Files

| File | Responsibility |
|---|---|
| `app.py` | The main window and the wiring between parts (entry point) |
| `api.py` | Every HTTP call to the controller |
| `widgets.py` | Reusable widgets — no HTTP, no threading |
| `audio.py` | Microphone capture |
| `export.py` | Word export and right-to-left detection |
| `config.py` | Constants |

## Run

```bash
pip install -r requirements.txt
python app.py          # or: run.bat (Windows) / ./run.sh (Linux)
```

The controller must already be running on `localhost:9002`, or wherever you
point the Host/Port fields.

The local model lists — both STT and LLM — are fetched from the controller
rather than hard-coded, so this app stays current as the service registries
change. If the controller is unreachable the lists stay empty; press `Refresh`
once the connection is up.

Microphone recording needs PortAudio. Without it the rest of the app still
works normally for file-based runs.

## Using it

1. Check the controller address (`Check connection`).
2. Pick a pipeline:
   - **Separate** — enable and configure up to 3 STT slots (each local or cloud).
   - **Multimodal** — choose an audio-capable LLM; the STT slots are hidden.
   - **Hybrid** — configure both the STT slot(s) and an audio-capable LLM.
3. Choose the LLM source (local or cloud). For cloud audio through Gemini,
   prefix the model with `gemini:`.
4. Press `Start job`.
5. Choose or record an audio file.
6. Press `Run`. The result — `raw_transcript`, `corrected_transcript`,
   `final_text`, `discrepancies_found`, `notes` — appears in the output box.
7. Optionally press `Save Transcript (.docx)` to write the final text to a Word
   file (Persian paragraphs are marked right-to-left automatically).

## Tests

```bash
pip install pytest
python -m pytest tests/
```
