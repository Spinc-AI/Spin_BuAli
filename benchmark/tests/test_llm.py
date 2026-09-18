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
            "MAX_MEMORY": getattr(self._core_llm_config, "MAX_MEMORY", "<absent>"),
        }

    def chat(self, messages, audio_path=None, temperature=0.3):
        return "{}"

    def unload(self):
        self.loaded = False


def core_config(quantization_knob=True, max_memory_knob=True):
    """A stand-in for `core_llm/config.py`.

    `quantization_knob=False` is an older core_llm without that knob at all
    -- exactly the state this repo was in when run 11 was recorded.
    `max_memory_knob=False` is the same idea for `MAX_MEMORY`.
    """
    module = types.ModuleType("stub_core_config")
    module.DEVICE_MAP = "cuda"
    module.MAX_NEW_TOKENS = 2048
    if quantization_knob:
        module.QUANTIZATION = None
    if max_memory_knob:
        module.MAX_MEMORY = None
    return module


def adapter(precision, cards=1, quantization_knob=True, max_memory_knob=True):
    config = core_config(quantization_knob, max_memory_knob)
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


class TestMaxMemoryForSharding:
    """The qwen3-omni-30b fix: a bare `device_map="auto"` with no explicit
    `max_memory` hit a known transformers/accelerate bug (#47211) where a
    single large leaf module collapses the whole device_map onto CPU/disk
    even when combined GPU budget is several times the model's size, and
    bitsandbytes' 4-bit quantizer then refuses that outright. An explicit
    `max_memory` (this adapter's own `_max_memory_for_sharding()`) routes
    around it.
    """

    def test_two_cards_sets_an_explicit_max_memory(self, monkeypatch):
        monkeypatch.setattr(llm_module, "_max_memory_for_sharding",
                            lambda: {0: "13.5GiB", 1: "13.5GiB", "cpu": "0GiB"})
        built, stub, _ = adapter("nf4", cards=2)
        built.load()
        assert stub.seen_at_load["MAX_MEMORY"] == {0: "13.5GiB", 1: "13.5GiB", "cpu": "0GiB"}

    def test_a_single_card_load_leaves_max_memory_alone(self):
        """emptiest_cuda_device() already picks a specific device for a
        single-card load -- an explicit max_memory there would be a second,
        redundant way of saying the same thing, not a fix for anything."""
        built, stub, _ = adapter("nf4", cards=1)
        built.load()
        assert stub.seen_at_load["MAX_MEMORY"] is None

    def test_max_memory_is_restored_after_a_sharded_load(self, monkeypatch):
        monkeypatch.setattr(llm_module, "_max_memory_for_sharding",
                            lambda: {0: "13.5GiB", "cpu": "0GiB"})
        built, _, config = adapter("nf4", cards=2)
        built.load()
        assert config.MAX_MEMORY is None

    def test_max_memory_is_restored_even_when_the_load_fails(self, monkeypatch):
        monkeypatch.setattr(llm_module, "_max_memory_for_sharding",
                            lambda: {0: "13.5GiB", "cpu": "0GiB"})
        built, stub, config = adapter("nf4", cards=2)
        stub.load = lambda: (_ for _ in ()).throw(RuntimeError("no weights here"))
        with pytest.raises(RuntimeError):
            built.load()
        assert config.MAX_MEMORY is None

    def test_an_older_core_llm_without_the_knob_still_loads(self):
        """Same graceful-degradation shape as the QUANTIZATION knob: a
        core_llm build that predates MAX_MEMORY should still serve a
        sharded load, just without the #47211 workaround."""
        built, stub, _ = adapter("nf4", cards=2, max_memory_knob=False)
        built.load()
        assert stub.loaded


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


class TestQuantizedLoadsForceFP16:
    """core_llm/model.py itself, not the adapter -- pinned as text, the same
    way test_runner.py checks core_llm without importing torch (not
    installed in this test environment; see that file's own comment).

    The bug: a quantized `_load_kwargs()` used to `kwargs.pop("dtype", None)`
    -- meaning a quantized GemmaAudioModel load specified no dtype at all,
    so its non-quantized layers (norms, embeddings) kept the checkpoint's own
    bf16. bitsandbytes' 8-bit matmul has one fixed compute dtype (fp16) and
    casts every bf16 activation that reaches it -- once per matmul, logged
    every time, through bitsandbytes' own logger rather than `warnings.warn`,
    which is why the notebook's blanket `warnings.filterwarnings("ignore")`
    never touched it. Confirmed live: gemma-4-12b at int8 produced enough
    duplicate lines across a nine-clip generation to hang the Kaggle notebook
    frontend solid, recoverable only with a hard reload.
    """

    @staticmethod
    def _source():
        import pathlib

        import settings

        return (settings.REPO_ROOT / "core_llm" / "model.py").read_text(encoding="utf-8")

    def test_a_quantized_load_kwargs_call_sets_fp16_not_pops_dtype(self):
        source = self._source()
        start = source.index("def _load_kwargs(")
        end = source.index("\n\n\n", start)
        body = source[start:end]
        assert 'kwargs.pop("dtype"' not in body, (
            "popping dtype under quantization leaves the checkpoint's own "
            "dtype (bf16 for every Gemma 4 variant) on the unquantized "
            "layers -- see this test's class docstring")
        assert 'kwargs["dtype"] = torch.float16' in body

    def test_bitsandbytes_logging_is_capped_as_defense_in_depth(self):
        source = self._source()
        assert 'logging.getLogger("bitsandbytes")' in source


class TestVoxtralAndQwen2AudioLiveFixes:
    """Two live-run failures, both 9/9 on the first real run of each model,
    pinned as text against core_llm/model.py -- same reasoning as
    TestQuantizedLoadsForceFP16 above: torch is not installed in this test
    environment, so the class bodies can only be inspected as source.
    """

    @staticmethod
    def _class_body(name):
        import re

        import settings

        source = (settings.REPO_ROOT / "core_llm" / "model.py").read_text(encoding="utf-8")
        match = re.search(rf"^class {name}\(BaseLLM\):\n(.*?)(?=^class |\Z)",
                          source, re.MULTILINE | re.DOTALL)
        assert match, f"{name} not found in core_llm/model.py"
        return match.group(1)

    def test_voxtral_never_sends_a_bare_system_role(self):
        """mistral-common's own request validation refuses a SystemMessage
        alongside an AudioChunk: `ValueError: Found system messages at
        indexes [...] and audio chunks in messages at indexes [...]. This is
        not allowed prior to the tokenizer version 13.` -- hit on 9/9 clips,
        every one identical. The fix folds any system content into the next
        user turn instead of emitting a separate system role at all."""
        body = self._class_body("VoxtralModel")
        assert '{"role": "system"' not in body
        assert "pending_system" in body

    def test_qwen2_audio_never_sends_an_empty_user_turn_with_audio(self):
        """Every official Qwen2-Audio example pairs its audio with a real
        question; a live run that sent audio alone (system-only
        instructions, empty user turn -- this benchmark's usual multimodal
        shape) returned raw SRT-style hallucinated captions on 9/9 clips,
        never the requested JSON."""
        body = self._class_body("Qwen2AudioModel")
        assert "not text" in body


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

    def test_qwen3_omni_is_back_in_the_default_roster(self):
        """Was excluded for a real, now-fixed reason: nf4 across two cards
        with a bare device_map="auto" and no explicit max_memory hit
        `ValueError: Some modules are dispatched on the CPU or the disk...`
        -- a known transformers/accelerate bug (#47211) where a single large
        leaf module collapses the whole device_map onto CPU/disk even when
        combined GPU budget is ample. CoreLLMAdapter.load() now passes an
        explicit max_memory whenever cards > 1 (see
        TestMaxMemoryForSharding below and llm._max_memory_for_sharding),
        which routes around that inference path. Re-included on the strength
        of that fix; not yet re-verified against a live card."""
        import runner

        assert "qwen3-omni-30b" in runner.MULTIMODAL_LLM
