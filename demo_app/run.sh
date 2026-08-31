#!/usr/bin/env bash
# Start the BuAli demo client. The controller (default port 9002) must be running.
set -e
pip install -r requirements.txt
python app.py
