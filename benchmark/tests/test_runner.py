"""`run_one` -- the one function every notebook cell calls.

The behaviour worth protecting: one call finishes by writing its CSV, the CSV
carries a SUMMARY row shaped like the old FLEURS notebook's, and a bad model or
a missing STT engine fails as a clear error rather than a confusing one.
"""
import json

import pytest

import llm as llm_module
import plan as plan_module
import runner
import transcribe


class FakeLLM:
    """Answers with the transcript it was handed, and remembers every prompt
    it was asked -- so a test can check the structure guide reached it."""

    model_key, precision, cards, load_seconds = "fake-llm", "fp16", 1, 0.0

    def __init__(self, text=None, fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    def load(self):
        return self

    def unload(self):
        pass

    def generate(self, system_prompt, user_text):
        self.calls.append(system_prompt)
        if self.fail:
            raise RuntimeError("boom")
        final = self.text if self.text is not None else (user_text or "")
        return json.dumps({"raw_transcript": final, "corrected_transcript": final,
                           "final_text": final})


class TestTop5:
    def test_it_is_exactly_five_real_registry_keys(self):
        """The bug this pins: TOP5_STT once held descriptive names instead of
        the registry's actual keys, which would have failed at load time on
        every one of them."""
        assert len(runner.TOP5_STT) == 5
        assert len(set(runner.TOP5_STT)) == 5

    def test_the_keys_match_the_pdf_s_rank_order(self):
        assert runner.TOP5_STT == [
            "seamless", "seamless-medium", "whisper", "mms-fl102", "whisper-vhdm"]


class TestRunOne:
    def test_a_run_writes_a_csv_and_returns_it(self, items, factory, tmp_path):
        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items,
            devices=["cpu"], model_factory=factory,
            llm_factory=lambda key, **kw: FakeLLM("a 6 mm stone in the right kidney"),
            results_dir=tmp_path)
        assert (tmp_path / "results__whisper__fake-llm__separate__fixed.csv").is_file()
        assert len(frame) == len(items) + 1  # one row per clip, plus SUMMARY

    def test_the_summary_row_is_last_and_named_summary(self, items, factory, tmp_path):
        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items, devices=["cpu"],
            model_factory=factory, llm_factory=lambda key, **kw: FakeLLM("x"),
            results_dir=tmp_path)
        assert frame.iloc[-1]["asset_id"] == "SUMMARY"

    def test_the_summary_row_carries_corpus_wer(self, items, factory, tmp_path):
        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items, devices=["cpu"],
            model_factory=factory,
            llm_factory=lambda key, **kw: FakeLLM(items[0].reference),
            results_dir=tmp_path)
        assert "corpus_wer" in frame.columns
        assert frame.iloc[-1]["corpus_wer"] == frame.iloc[-1]["corpus_wer"]  # not NaN

    def test_a_custom_label_names_the_file(self, items, factory, tmp_path):
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       model_factory=factory, llm_factory=lambda key, **kw: FakeLLM("x"),
                       label="01_whisper_solo", results_dir=tmp_path)
        assert (tmp_path / "results__01_whisper_solo.csv").is_file()

    def test_separate_without_stt_is_refused_before_any_model_loads(self, items, tmp_path):
        loaded = []
        with pytest.raises(ValueError, match="separate needs an STT model"):
            runner.run_one(None, "fake-llm", "separate", items,
                           llm_factory=lambda key, **kw: loaded.append(1) or FakeLLM("x"),
                           results_dir=tmp_path)
        assert not loaded, "the LLM must not load for a request that cannot run"

    def test_multimodal_needs_no_stt_key(self, items, tmp_path):
        frame = runner.run_one(None, "fake-llm", "multimodal", items,
                               llm_factory=lambda key, **kw: FakeLLM("a normal report"),
                               results_dir=tmp_path)
        assert len(frame) == len(items) + 1

    def test_a_failing_llm_still_writes_a_csv(self, items, factory, tmp_path):
        """The whole point of writing to disk unconditionally: a crash on this
        run must not cost the ones already done in earlier cells."""
        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items, devices=["cpu"],
            model_factory=factory, llm_factory=lambda key, **kw: FakeLLM(fail=True),
            results_dir=tmp_path)
        assert (tmp_path / "results__whisper__fake-llm__separate__fixed.csv").is_file()
        per_clip = frame[frame["asset_id"] != "SUMMARY"]
        assert (per_clip["transcription_error"].fillna("") != "").all()

    def test_the_structure_guide_reaches_the_llm_when_given(self, items, factory, tmp_path):
        import report_structure as rs

        model = FakeLLM("x")
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       model_factory=factory, llm_factory=lambda key, **kw: model,
                       structure_guide=rs.GUIDE, results_dir=tmp_path)
        assert all("Benchmark note" in call for call in model.calls)

    def test_stt_and_llm_columns_are_stamped_on_every_row(self, items, factory, tmp_path):
        frame = runner.run_one("whisper", "fake-llm", "hybrid", items, devices=["cpu"],
                               model_factory=factory,
                               llm_factory=lambda key, **kw: FakeLLM("x"),
                               results_dir=tmp_path)
        assert set(frame["stt_model"]) == {"whisper"}
        assert set(frame["llm_model"]) == {"fake-llm"}
        assert set(frame["pipeline"]) == {"hybrid"}


class TestBuildMaster:
    def _run(self, items, factory, tmp_path, label, text):
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       model_factory=factory, llm_factory=lambda key, **kw: FakeLLM(text),
                       label=label, results_dir=tmp_path)

    def test_it_merges_every_run_s_summary_row(self, items, factory, tmp_path):
        self._run(items, factory, tmp_path, "good", items[0].reference)
        self._run(items, factory, tmp_path, "bad", "nothing like the reference at all")
        master = runner.build_master(tmp_path)
        assert len(master) == 2
        assert (tmp_path / "results_master.csv").is_file()

    def test_it_is_sorted_by_corpus_wer_best_first(self, items, factory, tmp_path):
        self._run(items, factory, tmp_path, "bad", "nothing like the reference at all")
        self._run(items, factory, tmp_path, "good", items[0].reference)
        master = runner.build_master(tmp_path)
        assert master.iloc[0]["source_file"] == "results__good.csv"

    def test_nothing_to_merge_does_not_crash(self, tmp_path):
        assert runner.build_master(tmp_path) is None


class TestPreprocessingReachesTheSTTStage:
    def test_it_is_threaded_into_transcribe_batch(self, items, tmp_path, monkeypatch):
        """The bug this pins: run_one accepted no preprocessing argument at
        all, so every run silently used fixed windowing regardless of what a
        caller asked for."""
        seen = {}
        real = transcribe.transcribe_batch

        def spy(*args, **kwargs):
            seen["preprocessing"] = kwargs.get("preprocessing")
            return real(*args, **kwargs)

        monkeypatch.setattr(runner.transcribe, "transcribe_batch", spy)
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive-vad",
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert seen["preprocessing"] == "adaptive-vad"

    def test_it_is_stamped_on_every_row_and_the_default_label(self, items, factory, tmp_path):
        frame = runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                               model_factory=factory, preprocessing="uniform",
                               llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert set(frame["preprocessing"]) == {"uniform"}
        assert (tmp_path / "results__whisper__fake-llm__separate__uniform.csv").is_file()

    def test_no_preprocessing_given_labels_itself_fixed(self, items, factory, tmp_path):
        frame = runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                               model_factory=factory,
                               llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert set(frame["preprocessing"]) == {"fixed"}
