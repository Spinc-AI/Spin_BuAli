@echo off
REM Start the Core_LLM HTTP service. Open the SSH tunnel first (see SERVER_SETUP.md).
pip install -r requirements.txt
python main.py
