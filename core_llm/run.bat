@echo off
REM Start the Core_LLM service (default port 8001). The first request for a
REM given model downloads it from Hugging Face and loads it into VRAM.
pip install -r requirements.txt
python main.py
