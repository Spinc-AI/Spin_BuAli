"""Argument parsing for both training scripts -- the part of each that runs
without `torch`/`transformers`/`peft` installed, so it is real coverage, not
a source-text pin. Everything past `parse_args` needs those packages and is
covered by `test_collators_are_wired_correctly.py` instead (text-parsed, the
same reasoning `Spin_BuAli/benchmark/tests/test_llm.py` uses for
`core_llm/model.py` -- see that file's own docstring).
"""
import pathlib

import pytest

from voxtral import train_lora as voxtral_train
from whisper import train_lora as whisper_train


class TestVoxtralArgs:
    def test_the_documented_lora_recipe_is_the_default(self):
        args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        assert args.lora_r == 32
        assert args.lora_alpha == 32
        assert args.rslora is True
        assert "multi_modal_projector.linear_1" in args.target_modules
        assert "multi_modal_projector.linear_2" in args.target_modules
        for module in ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"):
            assert module in args.target_modules

    def test_the_documented_learning_rate_is_the_default(self):
        args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        assert args.learning_rate == 5e-5

    def test_no_rslora_flag_turns_it_off(self):
        args = voxtral_train.parse_args(["--labels-csv", "x.csv", "--no-rslora"])
        assert args.rslora is False

    def test_labels_csv_is_required(self):
        with pytest.raises(SystemExit):
            voxtral_train.parse_args([])

    def test_labels_csv_becomes_a_path(self):
        args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        assert isinstance(args.labels_csv, pathlib.Path)

    def test_dry_run_defaults_off(self):
        args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        assert args.dry_run is False

    def test_default_checkpoint_matches_the_registered_multimodal_roster_key(self):
        """This is the same checkpoint id `core_llm/config.py`'s
        VOXTRAL_MINI_MODEL_ID defaults to -- a fine-tune of anything else
        would produce an adapter the benchmark's own VoxtralModel couldn't
        load against by default."""
        args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        assert args.model_checkpoint == "mistralai/Voxtral-Mini-3B-2507"


class TestWhisperArgs:
    def test_the_documented_lora_recipe_is_the_default(self):
        args = whisper_train.parse_args(["--labels-csv", "x.csv"])
        assert args.lora_r == 32
        assert args.lora_alpha == 64
        assert args.target_modules == ["q_proj", "v_proj"]

    def test_the_documented_learning_rate_is_the_default(self):
        args = whisper_train.parse_args(["--labels-csv", "x.csv"])
        assert args.learning_rate == 1e-3

    def test_default_checkpoint_matches_the_production_stt_registry_key(self):
        """`stt/app/config.py`'s MODEL_REGISTRY["whisper"] -- domain-adapting
        the checkpoint already in production, not a generic Whisper that has
        never seen Persian."""
        args = whisper_train.parse_args(["--labels-csv", "x.csv"])
        assert args.model_checkpoint == "nezamisafa/whisper-persian-v4"

    def test_language_defaults_to_english_dominant_not_persian_dominant(self):
        """The project's own established framing: this audio is mostly
        English with Persian words mixed in, not the other way around."""
        args = whisper_train.parse_args(["--labels-csv", "x.csv"])
        assert args.language == "english"

    def test_labels_csv_is_required(self):
        with pytest.raises(SystemExit):
            whisper_train.parse_args([])


class TestBothScriptsShareTheSameSplitDefaults:
    """A train/eval split done differently by the two scripts would make a
    Voxtral run and a Whisper run silently incomparable -- same items, but
    scored against different eval sets."""

    def test_eval_fraction_matches(self):
        voxtral_args = voxtral_train.parse_args(["--labels-csv", "x.csv"])
        whisper_args = whisper_train.parse_args(["--labels-csv", "x.csv"])
        assert voxtral_args.eval_fraction == whisper_args.eval_fraction
        assert voxtral_args.split_seed == whisper_args.split_seed
        assert voxtral_args.min_eval_items == whisper_args.min_eval_items
