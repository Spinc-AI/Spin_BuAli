#!/usr/bin/env python3
"""Train Gemma 4 on this project's own dictations -- plain Python, no
notebook, no Kaggle account required.

    python run.py --labels-csv /path/to/labels.csv --dry-run
    python run.py --labels-csv /path/to/labels.csv

This is a one-line proxy to `gemma/train_lora.py`'s own `main()` -- not a
second implementation. `finetune/notebooks/kaggle_finetune.ipynb` calls that
exact same script too, as a subprocess (`python -m gemma.train_lora ...`),
for Kaggle-specific plumbing this file has no use for: GPU provisioning, the
live HF-token prompt, mounting the `spin-buali-dataset` Kaggle Dataset. None
of that is training logic, and none of it lives only in the notebook --
`gemma/train_lora.py` was always a normal `argparse` CLI script, tested and
runnable on its own (see `finetune/tests/test_cli.py`), on any machine with
a GPU and `pip install -r requirements.txt` run. This file exists only
because `python run.py ...` is a more obvious first thing to reach for than
`python -m gemma.train_lora ...` -- both do exactly the same thing.

For everything else (evaluating a checkpoint, the LoRA recipe, what
"--dry-run" proves and doesn't) see `finetune/README.md` and
`gemma/train_lora.py`'s own module docstring -- not repeated here, so
there is exactly one place each fact can go stale.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gemma.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
