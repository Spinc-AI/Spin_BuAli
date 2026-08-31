#!/usr/bin/env bash
# Start the STT service (default port 8000).
set -e
pip install -r requirements.txt
python -m app.main
