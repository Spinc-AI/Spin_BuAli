"""Minimal terminal chat loop — your first end-to-end test, no HTTP server needed.

Loads a model directly via model.py's MANAGER (same code the FastAPI layer
in main.py uses). First message is slow (model load); the rest are fast.

Run:
    python chat.py [model-key]   # model-key defaults to config.DEFAULT_MODEL

Type a question (Persian or English). Ctrl+C to quit.
"""
import sys

import config
from model import MANAGER

# A system prompt sets the model's role. This is a placeholder you'll later
# enrich with retrieved RAG references and outputs from the other modules.
SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping physicians. "
    "You answer in the language the doctor uses (Persian or English). "
    "Be precise, cite uncertainty, and never replace the physician's judgment."
)


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_MODEL
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Loading {model_key}... (first message will be slow)")
    print("Medical LLM ready. Ask something (Ctrl+C to quit).\n")

    try:
        while True:
            user = input("doctor> ").strip()
            if not user:
                continue
            history.append({"role": "user", "content": user})

            reply = MANAGER.chat(model_key, history)
            print(f"assistant> {reply}\n")
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
