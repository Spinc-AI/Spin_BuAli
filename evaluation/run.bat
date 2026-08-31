@echo off
REM Start the evaluation service (default port 8002). Independent of the
REM transcription path -- it only needs a transcript and a reference.
pip install -r requirements.txt
python main.py
