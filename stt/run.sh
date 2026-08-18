#!/usr/bin/env bash
# Start the STT service on Linux.
set -e
pip install -r requirements.txt
python -m app.main
