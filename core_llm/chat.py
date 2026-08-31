"""Terminal chat loop -- a no-HTTP smoke test for model loading and generation.

Loads a model through the same model.MANAGER the HTTP layer uses, so if this
works the service will too. The first message is slow (model load); the rest
are fast.

Run:
    python chat.py [model-key]   # defaults to config.DEFAULT_MODEL

Ctrl+C to quit.
"""
import sys

import config
from model import MANAGER

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping physicians. "
    "You answer in the language the doctor uses (Persian or English). "
    "Be precise, cite uncertainty, and never replace the physician's judgment."
)


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_MODEL
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Loading {model_key}... (the first message will be slow)")

    try:
        while True:
            question = input("doctor> ").strip()
            if not question:
                continue
            history.append({"role": "user", "content": question})
            reply = MANAGER.chat(model_key, history)
            print(f"assistant> {reply}\n")
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
