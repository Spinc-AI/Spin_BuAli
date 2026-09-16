"""Turning `benchmark/dataset.py`'s `Item`s into a training/eval split.

Reuses the benchmark's own CSV/manifest loader rather than re-parsing
`labels.csv` here -- the same rule `benchmark/notebooks/build_notebook.py`
follows for its rosters, for the same reason: a second copy of "how to read
this dataset" drifts from the first one silently.

Only `dataset.py` itself is imported, not the whole `benchmark` package --
that module's own imports (`csv`, `json`, `pathlib`, `numpy`, `soundfile`) are
torch-free, so everything in this file runs without a GPU or `transformers`
installed. `to_hf_dataset` is the one function that needs `datasets`; the
split logic below it does not, and is what `finetune/tests/test_data.py`
actually exercises.
"""
import pathlib
import random
import sys

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent.parent / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import dataset as bm_dataset  # noqa: E402


def load_items(labels_path):
    """Every labelled `Item` at `labels_path` -- a `labels.csv` or a JSON
    manifest, whichever `bm_dataset` recognises by extension.

    Unlabelled recordings are dropped here, not upstream in `dataset.py`:
    the benchmark keeps them (a run over unlabelled audio still produces
    useful drafts), but there is nothing to compute a training loss against
    for one, so keeping it in a training split would be a silent no-op
    example diluting every batch it lands in.
    """
    labels_path = pathlib.Path(labels_path)
    loader = bm_dataset.from_json if labels_path.suffix == ".json" else bm_dataset.from_csv
    items = loader(labels_path)
    labelled = [item for item in items if item.labelled]
    missing = [item.asset_id for item in labelled if not item.audio.is_file()]
    if missing:
        raise FileNotFoundError(f"labelled but missing audio file: {missing}")
    return labelled


def train_eval_split(items, eval_fraction=0.15, min_eval=1, seed=13):
    """A deterministic (train, eval) split.

    `eval_fraction` alone breaks on a dataset this small: 0.15 of nine items
    is 1.35, which `round()` sends to 1 -- fine today, but 0.15 of *three*
    items rounds to 0, silently training with no eval set at all and no
    error to say so. `min_eval` is the floor that makes that impossible
    instead of quietly happening; raise if the dataset cannot afford it
    rather than training on an eval set that doesn't exist.

    The split is by `asset_id`, sorted before shuffling, so the same seed
    reproduces the same split regardless of what order the CSV rows happen
    to be in -- a labels.csv re-exported from a spreadsheet can and does
    reorder rows without changing their content.
    """
    if len(items) <= min_eval:
        raise ValueError(
            f"{len(items)} labelled item(s) is not enough to hold out "
            f"min_eval={min_eval} for evaluation and still have anything "
            "left to train on")
    ordered = sorted(items, key=lambda item: item.asset_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    eval_count = max(min_eval, round(len(ordered) * eval_fraction))
    eval_count = min(eval_count, len(ordered) - 1)  # always leave >=1 to train on
    eval_items = ordered[:eval_count]
    train_items = ordered[eval_count:]
    return train_items, eval_items


def to_hf_dataset(items):
    """A `datasets.Dataset` with `asset_id`, `audio_path` and `text` columns.

    `audio_path` is a plain string, not a decoded `datasets.Audio` column on
    purpose: `gemma/collator.py` hands it straight to
    `processor.apply_chat_template` as `{"type": "audio", "url": ...}`,
    exactly the way `core_llm/model.py`'s `GemmaAudioModel` does at
    inference -- Gemma's own processor loads and resamples the file itself.
    Decoding it again here would be redundant work with its own chance to
    disagree with the processor's own resampling.

    Import is local: `datasets` is only needed here, not by the split logic
    above, which `finetune/tests/test_data.py` runs without it installed.
    """
    from datasets import Dataset

    rows = [{"asset_id": item.asset_id, "audio_path": str(item.audio), "text": item.reference}
            for item in items]
    return Dataset.from_list(rows)
