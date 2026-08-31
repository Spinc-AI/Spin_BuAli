#!/usr/bin/env bash
# Start the Core_LLM service (default port 8001). The first request for a
# given model downloads it from Hugging Face and loads it into VRAM.
set -e
pip install -r requirements.txt
python main.py
