# Server setup (run these ON the GPU server)

Core_LLM serves all its models directly via `transformers` — no Ollama, no
external API, nothing to install beyond Python packages and a CUDA-enabled
`torch`. Do this once per server.

## 1. Confirm the GPU is visible

```bash
nvidia-smi          # should list your GPU and its VRAM
```

## 2. Install dependencies

```bash
cd Core_LLM/deployment
pip install -r requirements.txt
```
If `pip install torch` grabs a CPU-only build, reinstall the CUDA one explicitly
(see [pytorch.org](https://pytorch.org) for the exact command for your CUDA version):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 3. Start the service

```bash
bash run.sh          # or: python main.py
```
Listens on `0.0.0.0:8001`. **No model is loaded at startup** — the first
request for a given model (via `/chat` or `/chat_audio`) downloads it from
Hugging Face and loads it into VRAM, which takes a while the first time;
after that it stays in memory until you switch models or call `/unload`.

Confirm it's up:
```bash
curl http://localhost:8001/
curl http://localhost:8001/models          # all registered models
curl http://localhost:8001/chat_audio/models  # just the audio-capable ones
```

---

# Connecting from your Windows machine

Don't expose port 8001 to the open internet. Open a secure **SSH tunnel** so
the server's port 8001 appears as `localhost:8001` on your machine:
```powershell
ssh -p <port> -L 8001:localhost:8001 user@your-server-address
```

Then, in a second window, either hit the HTTP API directly (`curl`, the demo
app's Core_LLM tab) or run the no-server terminal test:
```powershell
python chat.py                 # uses config.DEFAULT_MODEL
python chat.py qwen3-omni-30b  # or pick a specific registry key
```

When you're done, close the tunnel and shut the server down to stop billing.
