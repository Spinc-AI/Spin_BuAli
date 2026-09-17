#!/usr/bin/env python3
"""Baseline vs fine-tuned WER -- plain Python, no notebook.

    python evaluate.py --labels-csv /path/to/labels.csv
    python evaluate.py --labels-csv /path/to/labels.csv --adapter-dir ./gemma-buali-lora

A one-line proxy to `gemma/evaluate.py`'s own `main()` -- see `run.py`'s
docstring for why this thin wrapper exists at all; the same reasoning
applies here unchanged.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gemma.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
