#!/usr/bin/env bash
# Preprocess audio for the STT service on Linux.
set -e
pip install -r requirements.txt
python audio_preprocessing.py "$@"
