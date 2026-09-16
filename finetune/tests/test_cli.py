"""Argument parsing for `gemma/train_lora.py` and `gemma/evaluate.py` -- the
part of each that runs without `torch`/`transformers`/`peft` installed, so
it is real coverage, not a source-text pin.
"""
import pathlib

import pytest

from gemma import evaluate as gemma_evaluate
from gemma import train_lora as gemma_train


class TestTrainArgs:
    def test_the_documented_lora_recipe_is_the_default(self):
        """r=8, alpha=16 -- DebuggerCafe's reported Gemma 4 transcription
        recipe (see finetune/README.md's Research section)."""
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.lora_r == 8
        assert args.lora_alpha == 16
        assert args.lora_dropout == 0.0

    def test_target_modules_defaults_to_the_verified_peft_wildcard(self):
        """Not the article's specific audio-layer names -- see
        train_lora.py's module docstring for why guessing unverified
        submodule names is worse than PEFT's own documented "all-linear"."""
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.target_modules == ["all-linear"]

    def test_per_device_batch_size_defaults_to_one(self):
        """The only batch size GemmaChatCollator's audio-feature
        concatenation has been reasoned through against a real precedent --
        see its own docstring."""
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.per_device_batch_size == 1
        assert args.gradient_accumulation_steps > 1  # the larger effective batch instead

    def test_the_documented_learning_rate_is_the_default(self):
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.learning_rate == 5e-5

    def test_default_checkpoint_matches_the_registered_multimodal_roster_key(self):
        """The same checkpoint id `core_llm/config.py`'s GEMMA_E4B_MODEL_ID
        defaults to."""
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.model_checkpoint == "google/gemma-4-E4B-it"

    def test_structure_guide_defaults_off(self):
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.structure_guide is False

    def test_load_in_8bit_defaults_off(self):
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.load_in_8bit is False

    def test_labels_csv_is_required(self):
        with pytest.raises(SystemExit):
            gemma_train.parse_args([])

    def test_labels_csv_becomes_a_path(self):
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert isinstance(args.labels_csv, pathlib.Path)

    def test_dry_run_defaults_off(self):
        args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        assert args.dry_run is False


class TestEvaluateArgs:
    def test_default_checkpoint_matches_training(self):
        args = gemma_evaluate.parse_args(["--labels-csv", "x.csv"])
        assert args.model_checkpoint == "google/gemma-4-E4B-it"

    def test_adapter_dir_defaults_to_none_meaning_base_checkpoint_alone(self):
        args = gemma_evaluate.parse_args(["--labels-csv", "x.csv"])
        assert args.adapter_dir is None

    def test_split_defaults_match_training(self):
        """A different split here than train_lora.py used would score
        against items the model may have trained on."""
        train_args = gemma_train.parse_args(["--labels-csv", "x.csv"])
        eval_args = gemma_evaluate.parse_args(["--labels-csv", "x.csv"])
        assert train_args.eval_fraction == eval_args.eval_fraction
        assert train_args.min_eval_items == eval_args.min_eval_items
        assert train_args.split_seed == eval_args.split_seed

    def test_labels_csv_is_required(self):
        with pytest.raises(SystemExit):
            gemma_evaluate.parse_args([])
