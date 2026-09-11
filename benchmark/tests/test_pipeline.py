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
