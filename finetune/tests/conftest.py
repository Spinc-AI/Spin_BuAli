"""Shared fixtures. Nothing here loads a model, touches a GPU, or needs
`torch`/`transformers`/`peft` installed -- see each test module's own
docstring for which files that lets it cover directly instead of only
through source-text pinning.
"""
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

FINETUNE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))


@pytest.fixture
def tiny_dataset(tmp_path):
    """A `labels.csv` plus real (silent, one-second) audio files -- enough
    for `benchmark/dataset.py`'s own `from_csv` and `load_audio` to run
    against for real, without a network call or a GPU.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for i in range(9):
        asset_id = f"CLIP{i:03d}"
        path = audio_dir / f"{asset_id}.wav"
        sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
        rows.append((asset_id, f"audio/{asset_id}.wav", f"finding number {i}"))

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "asset_id,audio,report\n" +
        "\n".join(f"{a},{p},{r}" for a, p, r in rows) + "\n",
        encoding="utf-8")
    return csv_path
