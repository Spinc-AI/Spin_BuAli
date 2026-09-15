"""`CoreLLMAdapter` honouring the placement the tier system chose for it.

The bug these pin is not hypothetical, and it did not look like a bug in the
results. `tiers.plan_placements` put gemma-4-12b at `int8` because ~24 GB of
fp16 weights do not fit a 14.56 GB T4. `CoreLLMAdapter` stored that `precision`
and never used it -- `core_llm/` had no quantization at all -- so the model
loaded at full precision, OOMed on one clip, span into a repetition loop on
three more, and wrote a CSV whose every row says `precision=int8`.

Four failures under a precision the run never used is worse than four failures:
it is a row that will be read, ranked and believed. So there are two guarantees
here -- the placement is applied, and a placement that *cannot* be applied is a
loud failure rather than a quiet fp16 load.

Nothing here touches a GPU or downloads weights: a stub stands in for
`core_llm/model.py`'s class and a throwaway module for its `config`.
"""
import types

import pytest

import llm as llm_module


class StubCoreModel:
    """What `bridge.build_llm_model` returns, minus the weights.

    Records what `config` looked like at the moment `load()` was called --
    which is the only moment that matters, since `CoreLLMAdapter` is supposed
    to put every knob back afterwards.
    """

    supports_audio = True

    def __init__(self, core_config):
        self.model_id = "stub/model"
        self._core_llm_config = core_config
        self.seen_at_load = None
        self.loaded = False

    def load(self):
        self.loaded = True
        self.seen_at_load = {
            "DEVICE_MAP": self._core_llm_config.DEVICE_MAP,
            "QUANTIZATION": getattr(self._core_llm_config, "QUANTIZATION", "<absent>"),
        }

    def chat(self, messages, audio_path=None, temperature=0.3):
        return "{}"

    def unload(self):
        self.loaded = False


def core_config(quantization_knob=True):
    """A stand-in for `core_llm/config.py`.

    `quantization_knob=False` is an older core_llm without the knob at all --
    exactly the state this repo was in when run 11 was recorded.
    """
    module = types.ModuleType("stub_core_config")
    module.DEVICE_MAP = "cuda"
    module.MAX_NEW_TOKENS = 2048
    if quantization_knob:
        module.QUANTIZATION = None
    return module


def adapter(precision, cards=1, quantization_knob=True):
    config = core_config(quantization_knob)
    stub = StubCoreModel(config)
    return llm_module.CoreLLMAdapter("stub-model", stub, precision=precision,
                                     cards=cards), stub, config


class TestThePlacementIsApplied:
    @pytest.mark.parametrize("precision", ["int8", "nf4"])
    def test_a_quantized_placement_reaches_core_llm(self, precision):
        built, stub, _ = adapter(precision)
        built.load()
        assert stub.seen_at_load["QUANTIZATION"] == precision

    def test_fp16_asks_for_no_quantization(self):
        built, stub, _ = adapter("fp16")
        built.load()
        assert stub.seen_at_load["QUANTIZATION"] is None

    def test_two_cards_shard_rather_than_pinning_one(self):
        """core_llm defaults DEVICE_MAP to the string "cuda" -- a single GPU.
        A placement of (int8, 2 cards) that loaded onto one card would OOM
        for a reason the plan already ruled out."""
        built, stub, _ = adapter("int8", cards=2)
        built.load()
        assert stub.seen_at_load["DEVICE_MAP"] == "auto"

    @pytest.mark.parametrize("precision", ["fp16", "int8"])
    def test_every_knob_is_put_back_afterwards(self, precision):
        """core_llm's config is a module, so these writes are process-wide.
        Leaving either one set would silently change the *next* model loaded
        in the same kernel -- and in a notebook that is the next cell."""
        built, _, config = adapter(precision, cards=2)
        built.load()
        assert config.DEVICE_MAP == "cuda"
        assert config.QUANTIZATION is None

    def test_the_knobs_are_restored_even_when_the_load_fails(self):
        built, stub, config = adapter("nf4", cards=2)
        stub.load = lambda: (_ for _ in ()).throw(RuntimeError("no weights here"))
        with pytest.raises(RuntimeError):
            built.load()
        assert config.DEVICE_MAP == "cuda"
        assert config.QUANTIZATION is None


class TestAnUnapplicablePlacementFailsLoudly:
    @pytest.mark.parametrize("precision", ["int8", "nf4"])
    def test_it_refuses_rather_than_loading_at_full_precision(self, precision):
        """The exact shape of the gemma-4-12b run: asked for int8, got fp16,
        reported int8. Better to have no row than a wrong one."""
        built, stub, _ = adapter(precision, quantization_knob=False)
        with pytest.raises(llm_module.LoadFailed) as error:
            built.load()
        assert precision in str(error.value)
        assert not stub.loaded

    def test_fp16_still_loads_without_the_knob(self):
        """Only a *quantized* placement needs it. An older core_llm can still
        serve every tier-A model, and should."""
        built, stub, _ = adapter("fp16", quantization_knob=False)
        built.load()
        assert stub.loaded


class TestRouting:
    def test_audio_capable_and_nonstandard_keys_avoid_LocalLLM(self):
        """The routing bug that cost three identical debugging rounds:
        `build()` checked `plan.AUDIO_CAPABLE`, so medgemma-1.5-4b -- text
        only, but needing AutoModelForImageTextToText -- fell through to
        LocalLLM, "loaded" without error and returned nothing but empty
        replies. The predicate is TEXT_ONLY_STANDARD, and these keys are the
        ones it must not claim."""
        import plan

        for key in plan.AUDIO_CAPABLE | {"medgemma-1.5-4b"}:
            assert key not in plan.TEXT_ONLY_STANDARD, (
                f"{key} would route to LocalLLM, which loads a plain causal LM")

    def test_every_multimodal_roster_key_is_audio_capable_in_the_plan(self):
        import plan
        import runner

        assert set(runner.MULTIMODAL_LLM) <= plan.AUDIO_CAPABLE

    def test_every_rostered_key_has_a_parameter_count_to_place_it_by(self):
        """Without one, `tiers.plan_placements` silently skips the model and
        the notebook's `placement_for()` raises StopIteration -- a
        confusing failure for what is really a missing dict entry."""
        import plan
        import runner

        for key in runner.MULTIMODAL_LLM + runner.TOP3_LLM:
            assert key in plan.LLM_PARAMS
