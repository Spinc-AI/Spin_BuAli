"""The report stage: transcripts in, a scored radiology report out.

The claim being protected is that this measures the shipped system. It uses
the controller's prompts, not its own; it scores the generated report, not the
transcript; and a model that fails on one recording costs that recording only.
"""
import json
import pathlib

import pytest

import bridge
import llm as llm_module
import pipeline
import scoring
from dataset import Item

REFERENCE = "There is a 6 mm stone in the distal right ureter. No hydronephrosis."


class FakeLLM:
    """Answers with whatever it is told to, and records what it was asked."""

    model_key, precision, cards, load_seconds = "fake", "fp16", 1, 0.0

    def __init__(self, text=REFERENCE, raw_reply=None, fail=False):
        self.text = text
        self.raw_reply = raw_reply
        self.fail = fail
        self.calls = []

    def load(self):
        return self

    def unload(self):
        pass

    def generate(self, system_prompt, user_text, audio_path=None):
        self.calls.append({"system": system_prompt, "user": user_text, "audio_path": audio_path})
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        if self.raw_reply is not None:
            return self.raw_reply
        return json.dumps({"raw_transcript": "raw", "corrected_transcript": "c",
                           "final_text": self.text})


@pytest.fixture
def items():
    return [Item("A1", pathlib.Path("A1.wav"), REFERENCE)]


class TestItUsesTheProductionPrompt:
    def test_separate_gets_the_reconcile_prompt(self):
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "a stone"}, model, "separate")
        assert model.calls[0]["system"].startswith(bridge.RECONCILE)

    def test_multimodal_gets_the_audio_prompt(self):
        model = FakeLLM()
        pipeline.build_report("A1", {}, model, "multimodal", audio_path=pathlib.Path("A1.wav"))
        assert model.calls[0]["system"].startswith(bridge.TRANSCRIBE_FROM_AUDIO)

    def test_multimodal_without_audio_path_is_refused(self):
        """Silently generating from no transcript and no audio would produce a
        report about nothing -- the exact failure mode this whole thread
        started from (multimodal cells that never actually heard anything)."""
        model = FakeLLM()
        report = pipeline.build_report("A1", {}, model, "multimodal")
        assert "needs the recording" in report.error
        assert model.calls == []

    def test_the_audio_path_reaches_generate(self):
        model = FakeLLM()
        pipeline.build_report("A1", {}, model, "multimodal", audio_path=pathlib.Path("A1.wav"))
        assert model.calls[0]["audio_path"] == pathlib.Path("A1.wav")

    def test_separate_refuses_an_audio_path(self):
        """separate reconciles transcripts; an audio_path here is a caller
        bug (asking the wrong pipeline to use audio), not something to
        silently ignore."""
        model = FakeLLM()
        report = pipeline.build_report("A1", {"transcript_1": "x"}, model, "separate",
                                       audio_path=pathlib.Path("A1.wav"))
        assert "not audio" in report.error
        assert model.calls == []

    def test_the_json_template_is_always_appended(self):
        """Without it the model has no idea what shape to answer in."""
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "x"}, model, "separate")
        assert "JSON template to fill" in model.calls[0]["system"]

    def test_the_prompt_is_the_controller_s_not_a_copy(self):
        """A benchmark with its own wording ranks a system nobody ships."""
        import sys

        sys.path.insert(0, str(bridge.settings.CONTROLLER_DIR))
        import prompts

        assert bridge.RECONCILE is prompts.RECONCILE

    def test_transcripts_are_labelled_by_slot(self):
        """The prompt asks the model to weigh them against each other, which it
        cannot do if it cannot tell where one ends."""
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "first", "transcript_2": "second"},
                              model, "separate")
        user = model.calls[0]["user"]
        assert "STT engine 1 transcript" in user and "STT engine 2 transcript" in user

    def test_multimodal_calls_its_transcripts_fallible(self):
        """Reference material, not ground truth -- it changes what the model
        does with a disagreement."""
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "x"}, model, "multimodal",
                              audio_path=pathlib.Path("A1.wav"))
        assert "may contain errors" in model.calls[0]["user"]

    def test_slot_order_follows_the_number_not_the_string(self):
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_10": "tenth", "transcript_2": "second"},
                              model, "separate")
        user = model.calls[0]["user"]
        assert user.index("second") < user.index("tenth")

    def test_context_reaches_the_system_prompt(self):
        """Modality/region context sits between the base prompt and the
        structure guide -- see reconcile_prompt's docstring for why that
        order."""
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "x"}, model, "separate",
                              context="Modality: Ultrasound.")
        assert "Modality: Ultrasound." in model.calls[0]["system"]

    def test_no_context_means_no_context_line(self):
        model = FakeLLM()
        pipeline.build_report("A1", {"transcript_1": "x"}, model, "separate")
        assert "Modality" not in model.calls[0]["system"]


class TestParsingTheReply:
    def test_a_fenced_reply_is_still_read(self):
        """Every one of these models wraps its answer in a code fence."""
        reply = "Here you go:\n```json\n" + json.dumps({
            "raw_transcript": "r", "corrected_transcript": "c",
            "final_text": REFERENCE}) + "\n```"
        report = pipeline.build_report("A1", {"transcript_1": "x"},
                                       FakeLLM(raw_reply=reply), "separate")
        assert report.final_text == REFERENCE and report.error is None

    def test_a_reply_that_is_not_json_is_recorded_not_raised(self):
        report = pipeline.build_report("A1", {"transcript_1": "x"},
                                       FakeLLM(raw_reply="I am unable to help."), "separate")
        assert report.error and "no JSON object" in report.error

    def test_a_reply_missing_final_text_is_named(self):
        reply = json.dumps({"raw_transcript": "r", "corrected_transcript": "c"})
        report = pipeline.build_report("A1", {}, FakeLLM(raw_reply=reply), "multimodal",
                                       audio_path=pathlib.Path("A1.wav"))
        assert report.error.startswith("the model returned no final_text")

    def test_the_error_carries_a_snippet_of_the_reply(self):
        """A failure without the actual reply text is a dead end -- the CSV's
        `transcription_error` column is the only place a Kaggle run's failures
        are visible after the fact."""
        reply = json.dumps({"raw_transcript": "r", "corrected_transcript": "c"})
        report = pipeline.build_report("A1", {}, FakeLLM(raw_reply=reply), "multimodal",
                                       audio_path=pathlib.Path("A1.wav"))
        assert "raw_transcript" in report.error

    def test_a_long_reply_is_truncated_to_head_and_tail(self):
        reply = json.dumps({"raw_transcript": "x" * 5000})
        report = pipeline.build_report("A1", {}, FakeLLM(raw_reply=reply), "multimodal",
                                       audio_path=pathlib.Path("A1.wav"))
        assert "chars omitted" in report.error
        assert len(report.error) < 1000

    def test_separate_with_no_transcript_is_refused(self):
        report = pipeline.build_report("A1", {}, FakeLLM(), "separate")
        assert "at least one transcript" in report.error

    def test_one_failed_recording_does_not_end_the_batch(self):
        """It costs that recording, not the other eight."""
        reports = pipeline.build_reports(
            {"A1": {"transcript_1": "x"}, "A2": {"transcript_1": "y"}},
            FakeLLM(fail=True), "separate")
        assert len(reports) == 2 and all(r.error for r in reports)

    def test_build_reports_derives_context_from_items(self):
        """Unlike audio_path, context reaches `separate` too -- items carries
        it because that's where modality/regions live (dataset.Item), not
        because separate needs the audio."""
        model = FakeLLM()
        items = [Item("A1", pathlib.Path("A1.wav"), "ref", modality="Ultrasound",
                      regions=("Abdomen", "Pelvis"))]
        pipeline.build_reports({"A1": {"transcript_1": "x"}}, model, "separate", items=items)
        assert "Modality: Ultrasound." in model.calls[0]["system"]
        assert "Region(s) examined: Abdomen, Pelvis." in model.calls[0]["system"]

    def test_build_reports_without_items_has_no_context(self):
        model = FakeLLM()
        pipeline.build_reports({"A1": {"transcript_1": "x"}}, model, "separate")
        assert "Modality" not in model.calls[0]["system"]


class TestScoringTheReport:
    def test_a_perfect_report_scores_zero(self, items):
        reports = pipeline.build_reports({"A1": {"transcript_1": "x"}},
                                         FakeLLM(REFERENCE), "separate")
        results, _ = scoring.score_reports(reports, items, bridge.ClinicalTerms())
        assert results[0]["general"]["wer"] == 0.0
        assert results[0]["clinical_metrics"]["medical_term_f1"] == 1.0

    def test_a_flipped_side_barely_moves_wer_but_trips_laterality(self, items):
        """The reason the leaderboard ranks on clinical metrics: this is the
        error that sends a surgeon to the wrong kidney, and WER shrugs."""
        wrong = REFERENCE.replace("right", "left")
        reports = pipeline.build_reports({"A1": {"transcript_1": "x"}},
                                         FakeLLM(wrong), "separate")
        results, _ = scoring.score_reports(reports, items, bridge.ClinicalTerms())
        assert results[0]["general"]["wer"] < 0.1
        assert results[0]["clinical_counts"]["laterality_errors"] > 0

    def test_a_failed_report_scores_as_empty_not_absent(self, items):
        """Dropping it would flatter the configuration that broke."""
        reports = pipeline.build_reports({"A1": {"transcript_1": "x"}},
                                         FakeLLM(fail=True), "separate")
        results, _ = scoring.score_reports(reports, items, bridge.ClinicalTerms())
        assert len(results) == 1
        assert results[0]["general"]["hypothesis_words"] == 0
        assert results[0]["transcription_error"]

    def test_the_report_is_scored_not_the_transcript(self, items):
        """The labels are signed reports, so the report is what is graded."""
        reports = pipeline.build_reports({"A1": {"transcript_1": "totally different words"}},
                                         FakeLLM(REFERENCE), "separate")
        results, _ = scoring.score_reports(reports, items, bridge.ClinicalTerms())
        assert results[0]["hypothesis"] == REFERENCE

    def test_unlabelled_recordings_are_not_scored(self):
        items = [Item("A1", pathlib.Path("A1.wav"), None)]
        reports = pipeline.build_reports({"A1": {}}, FakeLLM(), "multimodal", items=items)
        results, _ = scoring.score_reports(reports, items, bridge.ClinicalTerms())
        assert results == []

    def test_the_summary_carries_the_corpus_figures(self, items):
        reports = pipeline.build_reports({"A1": {"transcript_1": "x"}},
                                         FakeLLM(REFERENCE), "separate")
        _, summary = scoring.score_reports(reports, items, bridge.ClinicalTerms(),
                                           {"llm_model": "m", "pipeline": "separate"})
        assert summary["models"][0]["corpus"]["corpus_wer"] == 0.0


class TestModelSelection:
    def test_a_prefixed_model_goes_to_the_cloud_path(self):
        assert isinstance(llm_module.build("gemini:gemini-2.5-pro"), llm_module.CloudLLM)

    def test_a_registry_key_resolves_to_a_checkpoint(self):
        """Read from core_llm/config.py, so adding a model there is enough."""
        model = llm_module.build("aya-expanse-8b", precision="fp16", cards=2)
        assert isinstance(model, llm_module.LocalLLM)
        assert "aya" in model.model_id.lower()

    def test_an_unknown_key_says_where_to_add_it(self):
        with pytest.raises(llm_module.LoadFailed, match="MODEL_REGISTRY"):
            llm_module.build("no-such-model")

    def test_a_non_audio_special_model_still_routes_through_core_llm(self, monkeypatch):
        """The real bug behind medgemma-1.5-4b's silent empty replies: it is
        not audio-capable, but it still needs core_llm/model.py's
        MedGemmaTextModel (AutoModelForImageTextToText), not LocalLLM's
        plain AutoModelForCausalLM. Routing on plan.AUDIO_CAPABLE alone let
        it fall through to LocalLLM, which "loaded" without error and then
        produced nothing but empty replies -- three times, unchanged, because
        every fix aimed at MedGemmaTextModel was never actually reached.
        Routing on plan.TEXT_ONLY_STANDARD instead closes that gap for any
        future non-audio, non-plain-causal-LM key too, not just this one."""
        import plan as plan_module

        class Stub:
            model_id = "google/medgemma-1.5-4b-it"

        assert "medgemma-1.5-4b" not in plan_module.AUDIO_CAPABLE
        assert "medgemma-1.5-4b" not in plan_module.TEXT_ONLY_STANDARD
        stub = Stub()
        monkeypatch.setattr(bridge, "build_llm_model", lambda key: stub)
        model = llm_module.build("medgemma-1.5-4b", precision="fp16", cards=1)
        assert isinstance(model, llm_module.CoreLLMAdapter)

    def test_the_audio_path_reaches_the_model_as_a_string(self):
        """dataset.Item.audio is a pathlib.Path, and the processors reject
        one -- "Incorrect format used for `audio`. Should be a numpy array or
        a `str`". That failed every recording of every multimodal run."""
        class FakeCoreModel:
            model_id = "fake/core"
            supports_audio = True

            def __init__(self):
                self.seen = None

            def chat(self, messages, audio_path=None, temperature=0.3):
                self.seen = audio_path
                return "{}"

        core = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", core).generate(
            "sys", "user", audio_path=pathlib.Path("A1.wav"))
        assert isinstance(core.seen, str), f"got {type(core.seen).__name__}"

    def test_core_llm_classes_stringify_the_audio_path(self):
        """The same guarantee inside core_llm/model.py, where the payload is
        actually built. Read as text -- it imports torch at module level."""
        import re

        import settings

        source = (settings.REPO_ROOT / "core_llm" / "model.py").read_text(encoding="utf-8")
        for match in re.finditer(r'"(?:url|path)":\s*([^,}\n]+)', source):
            value = match.group(1).strip()
            assert value.startswith("str("), (
                f'audio payload passes {value} unconverted; a Path is rejected')

    def test_the_stt_models_use_the_singular_audio_kwarg(self):
        """`audios=` was deprecated and is a hard ValueError from transformers
        v5 on. SeamlessV1Model still used it, so seamless-medium failed every
        recording while SeamlessV2Model (already singular) was fine.

        Read as text, not imported -- stt/app/model.py imports torch at module
        level, and this suite deliberately runs without it."""
        import re

        import settings

        source = (settings.REPO_ROOT / "stt" / "app" / "model.py").read_text(encoding="utf-8")
        assert not re.search(r"\baudios\s*=", source), (
            "a processor call still passes audios=; transformers v5 rejects it")

    def test_a_single_card_placement_takes_the_emptiest_card(self, monkeypatch):
        """Not always cuda:0: whatever ran before does not always give every
        byte back, and a 1-card model has no reason to insist on a busy card
        when another is free."""
        class FakeCoreConfig:
            DEVICE_MAP = "cuda"

        class FakeCoreModel:
            model_id = "fake/core"
            supports_audio = False

            def __init__(self):
                self.device_map_at_load = None
                self._core_llm_config = FakeCoreConfig

            def load(self):
                self.device_map_at_load = FakeCoreConfig.DEVICE_MAP

        monkeypatch.setattr(llm_module, "emptiest_cuda_device", lambda: "cuda:1")
        core = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", core, cards=1).load()
        assert core.device_map_at_load == "cuda:1"
        assert FakeCoreConfig.DEVICE_MAP == "cuda", "restored afterwards"

    def test_core_llm_adapter_shards_across_cards(self):
        """The bug this pins: CoreLLMAdapter stored `cards` and never used
        it, while core_llm's classes hardcode device_map="cuda" (one GPU).
        A (fp16, 2 cards) placement for gemma-4-e4b -- 7.85B, ~15.7 GB, which
        cannot fit one 14.56 GB T4 -- still loaded onto a single card and
        died with a CUDA OOM partway through the weights."""
        class FakeCoreConfig:
            DEVICE_MAP = "cuda"

        class FakeCoreModel:
            model_id = "fake/core"
            supports_audio = False

            def __init__(self):
                self.device_map_at_load = None
                self._core_llm_config = FakeCoreConfig

            def load(self):
                self.device_map_at_load = FakeCoreConfig.DEVICE_MAP

        two = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", two, cards=2).load()
        assert two.device_map_at_load == "auto", "a 2-card placement must shard"
        assert FakeCoreConfig.DEVICE_MAP == "cuda", "and restore afterwards"

        one = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", one, cards=1).load()
        assert one.device_map_at_load == "cuda", "a 1-card placement stays pinned"

    def test_core_llm_adapter_honours_max_new_tokens(self):
        """The bug this pins: CoreLLMAdapter accepted max_new_tokens and
        silently discarded it, so runner.run_one(max_new_tokens=...) -- and
        the notebook's MAX_NEW_TOKENS knob, which exists specifically to
        recover from truncated-JSON replies -- did nothing at all for every
        model routed through this adapter (most of the roster)."""
        class FakeCoreConfig:
            MAX_NEW_TOKENS = 1536

        class FakeCoreModel:
            model_id = "fake/core"
            supports_audio = False

            def __init__(self):
                self.seen = []
                self._core_llm_config = FakeCoreConfig

            def chat(self, messages, audio_path=None, temperature=0.3):
                self.seen.append(FakeCoreConfig.MAX_NEW_TOKENS)
                return "{}"

        core = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", core, max_new_tokens=3072).generate("sys", "user")
        assert core.seen == [3072], "the override must be visible during the call"
        assert FakeCoreConfig.MAX_NEW_TOKENS == 1536, "and restored after it"

    def test_no_override_leaves_core_llm_config_untouched(self):
        class FakeCoreConfig:
            MAX_NEW_TOKENS = 1536

        class FakeCoreModel:
            model_id = "fake/core"
            supports_audio = False

            def __init__(self):
                self.seen = []
                self._core_llm_config = FakeCoreConfig

            def chat(self, messages, audio_path=None, temperature=0.3):
                self.seen.append(FakeCoreConfig.MAX_NEW_TOKENS)
                return "{}"

        core = FakeCoreModel()
        llm_module.CoreLLMAdapter("k", core).generate("sys", "user")
        assert core.seen == [1536]

    def test_text_only_standard_keys_still_use_localllm(self):
        """Unchanged behaviour: aya-expanse and gemma-4-31b are plain
        TextOnlyModel in core_llm/model.py, functionally identical to
        LocalLLM's own loader -- no reason to route them elsewhere, and
        LocalLLM is what keeps quantized-tier support for them."""
        import plan as plan_module

        for key in plan_module.TEXT_ONLY_STANDARD:
            assert llm_module.build(key, precision="fp16", cards=1).__class__ is llm_module.LocalLLM

    def test_the_precision_from_the_plan_is_carried_through(self):
        model = llm_module.build("aya-expanse-32b", precision="nf4", cards=2)
        assert model.precision == "nf4" and model.cards == 2

    def test_the_dry_run_model_answers_without_weights(self):
        model = llm_module.dry_run_factory({}).load()
        reply = json.loads(model.generate("system", "the transcript"))
        assert reply["final_text"] == "the transcript"

    def test_the_dry_run_factory_accepts_either_calling_convention(self):
        """session.py calls a factory as factory(run_dict); runner.run_one
        calls one as factory(llm_key, precision=..., cards=...), matching
        build()'s own signature. Both must work without a TypeError."""
        llm_module.dry_run_factory({"llm_model": "x"})            # session.py's shape
        llm_module.dry_run_factory("aya-expanse-8b",               # runner.py's shape
                                   precision="fp16", cards=2)
