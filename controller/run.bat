@echo off
REM Start the BuAli controller. STT and Core_LLM (from Spin_Medical_Assistant_Project)
REM must be running/reachable (set STT_URL / LLM_URL in .env if not at the defaults).
pip install -r requirements.txt
python main.py
