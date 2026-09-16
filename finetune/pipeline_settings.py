"""Fine-tuning knobs, from the environment or `finetune/.env`.

Deliberately not called `settings.py`, despite `benchmark/settings.py`
existing for exactly the same reason (that name was tried first here, and
`test_cli.py` caught the exact collision within an hour of writing it):
`data.py` puts `benchmark/` on `sys.path` to reach `dataset.py`, and once
that happens, this folder's own `import settings` silently starts resolving
to `benchmark/settings.py` instead -- no error, just this file's own knobs
replaced by the wrong ones' values for the rest of the process. `config.py`
would collide the same way with `evaluation/`, `controller/`, `stt/app/` and
`core_llm/`'s own `config.py`, which is `benchmark/settings.py`'s reason.
`pipeline_settings.py` collides with nothing already on `sys.path` here.
"""
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK_DIR = pathlib.Path(os.getenv("BENCHMARK_DIR", str(REPO_ROOT / "benchmark")))

TARGET_SAMPLE_RATE = 16000

# Held out for eval, not trained on. A fraction of the WHOLE dataset -- with
# nine labelled clips today that rounds to roughly one, which is the point:
# see data.train_eval_split for why a fraction alone is not enough once the
# dataset is this small.
EVAL_FRACTION = float(os.getenv("FT_EVAL_FRACTION", "0.15"))
MIN_EVAL_ITEMS = int(os.getenv("FT_MIN_EVAL_ITEMS", "1"))
SPLIT_SEED = int(os.getenv("FT_SPLIT_SEED", "13"))
