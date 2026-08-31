#!/usr/bin/env bash
# Start the evaluation service (default port 8002). Independent of the
# transcription path -- it only needs a transcript and a reference.
set -e
pip install -r requirements.txt
python main.py
