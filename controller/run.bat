@echo off
REM Start the BuAli controller (default port 9002). The stt/ and core_llm/
REM services must be reachable -- set STT_URL / LLM_URL in .env if they
REM aren't at the defaults.
pip install -r requirements.txt
python main.py
