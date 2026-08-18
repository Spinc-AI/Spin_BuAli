#!/usr/bin/env bash
# Start the Core_LLM HTTP service on Linux. Models are served locally via
# transformers (see model.py) -- no Ollama required. First request for a
# given model downloads it from Hugging Face and loads it into VRAM, which
# takes a while; subsequent requests for the same model are fast.
set -e
pip install -r requirements.txt
python main.py
