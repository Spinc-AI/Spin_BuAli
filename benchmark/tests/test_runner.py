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
from conftest import FakeModel


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

    def generate(self, system_prompt, user_text, audio_path=None):
        self.calls.append(system_prompt)
        if self.fail:
            raise RuntimeError("boom")
        final = self.text if self.text is not None else (user_text or "")
        return json.dumps({"raw_transcript": final, "corrected_transcript": final,
                           "final_text": final})


class TestTop3:
    def test_it_is_exactly_three_real_registry_keys(self):
        """The bug this pins: the list once held descriptive names instead of
        the registry's actual keys, which would have failed at load time on
        every one of them."""
        assert len(runner.TOP3_STT) == 3
        assert len(set(runner.TOP3_STT)) == 3

    def test_the_keys_match_the_pdf_s_rank_order(self):
        assert runner.TOP3_STT == ["seamless", "seamless-medium", "whisper"]

    def test_top3_llm_and_multimodal_llm_are_exactly_three_real_registry_keys(self):
        """Same bug class as TOP3_STT, on the LLM side: a key that isn't
        actually in core_llm/model.py's MODEL_REGISTRY resolves to a
        LoadFailed at load time, not at notebook-build time. Resolving each
        key for real (against the live core_llm/config.py + model.py source)
        is what would have caught it."""
        for keys in (runner.TOP3_LLM, runner.MULTIMODAL_LLM):
            assert len(keys) == 3
            assert len(set(keys)) == 3
            for key in keys:
                assert llm_module._hugging_face_id(key)  # raises LoadFailed if unknown

    def test_multimodal_llm_are_all_audio_capable(self):
        """The whole reason MULTIMODAL_LLM exists as a separate list from
        TOP3_LLM: the multimodal pipeline sends audio directly to the model,
        so a text-only key here would fail every run, not just look odd.

        Parsed as text, not imported -- same reasoning as
        llm._hugging_face_id: importing core_llm pulls in torch and a
        `config` module that collides with the other siblings on sys.path.
        """
        import re

        import settings

        registry = (settings.REPO_ROOT / "core_llm" / "model.py").read_text(encoding="utf-8")
        # Audio-capable classes only, per core_llm/model.py's own module
        # docstring -- TextOnlyModel and MedGemmaTextModel are deliberately
        # excluded.
        audio_capable = {"GemmaAudioModel", "QwenOmniModel", "Phi4MultimodalModel"}
        for key in runner.MULTIMODAL_LLM:
            match = re.search(rf'"{re.escape(key)}":\s*\((\w+),', registry)
            assert match, f"{key} not found in core_llm/model.py's MODEL_REGISTRY"
            assert match.group(1) in audio_capable, (
                f"{key} is registered as {match.group(1)}, which cannot take audio input")


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
        frame = runner.run_one("whisper", "fake-llm", "multimodal", items, devices=["cpu"],
                               model_factory=factory,
                               llm_factory=lambda key, **kw: FakeLLM("x"),
                               results_dir=tmp_path)
        assert set(frame["stt_model"]) == {"whisper"}
        assert set(frame["llm_model"]) == {"fake-llm"}
        assert set(frame["pipeline"]) == {"multimodal"}


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


class TestTranscriptCaching:
    """Speech recognition is the expensive stage and does not change when the
    LLM or the pipeline does, so a second run against the same (preprocessing,
    stt_key) pair should not repeat it. What matters here: the cache key
    excludes the LLM and the pipeline, a cache hit never touches transcribe(),
    a failed transcription is never cached, and a caller can always force a
    fresh run."""

    def _asset_ids(self, items):
        return {item.asset_id for item in items}

    def test_the_cache_round_trips_through_disk(self, tmp_path):
        path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        runner._save_transcripts(path, {"A1": {"transcript_1": "hello"},
                                        "A2": {"transcript_1": "world"}})
        loaded = runner._load_cached_transcripts(path, {"A1", "A2"})
        assert loaded == {"A1": "hello", "A2": "world"}

    def test_a_cache_missing_a_needed_asset_is_treated_as_a_miss(self, tmp_path):
        """Widening the dataset after the cache was written must not silently
        score the new recordings against nothing."""
        path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        runner._save_transcripts(path, {"A1": {"transcript_1": "hello"}})
        assert runner._load_cached_transcripts(path, {"A1", "A2"}) is None

    def test_no_cache_file_is_a_miss_not_an_error(self, tmp_path):
        path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        assert runner._load_cached_transcripts(path, {"A1"}) is None

    def test_a_corrupt_cache_file_is_a_miss_not_a_crash(self, tmp_path):
        path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert runner._load_cached_transcripts(path, {"A1"}) is None

    def test_a_pre_seeded_cache_is_used_without_calling_transcribe_batch(
            self, items, tmp_path, monkeypatch):
        """The cache hit has to short-circuit *before* transcribe_batch, not
        just skip using its result -- transcribe_batch is what would try to
        load real STT weights."""
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        runner._save_transcripts(cache_path, {
            item.asset_id: {"transcript_1": f"cached text for {item.asset_id}"}
            for item in items})

        def explode(*args, **kwargs):
            raise AssertionError("transcribe_batch must not be called on a cache hit")

        monkeypatch.setattr(runner.transcribe, "transcribe_batch", explode)
        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items, preprocessing="adaptive",
            llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert set(frame["stt_cached"]) == {True}

    def test_a_fresh_run_writes_the_cache(self, items, factory, tmp_path):
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=factory,
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        assert cache_path.is_file()
        cached = runner._load_cached_transcripts(cache_path, self._asset_ids(items))
        assert cached is not None

    def test_a_model_factory_always_bypasses_reading_the_cache(self, items, factory, tmp_path):
        """An override is there to be exercised, not silently skipped by
        whatever a previous real run happened to produce."""
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        runner._save_transcripts(cache_path, {
            item.asset_id: {"transcript_1": "stale cached text"} for item in items})

        frame = runner.run_one(
            "whisper", "fake-llm", "separate", items, devices=["cpu"],
            preprocessing="adaptive", model_factory=factory,
            llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert set(frame["stt_cached"]) == {False}

    def test_use_cache_false_also_bypasses_a_present_cache(self, items, factory, tmp_path,
                                                            monkeypatch):
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        runner._save_transcripts(cache_path, {
            item.asset_id: {"transcript_1": "stale"} for item in items})

        calls = []
        monkeypatch.setattr(runner, "_load_cached_transcripts",
                            lambda *a, **k: calls.append(1))
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=factory, use_cache=False,
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert not calls, "use_cache=False must not even check the cache"

    def test_a_failed_transcription_is_not_cached(self, items, tmp_path):
        """Caching a failure would make every later run against this pair
        fail the same way for a reason nobody could see."""
        def failing_factory(key, device, **kwargs):
            return FakeModel(model_id=f"fake/{key}", device=device, fail_on=1)

        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=failing_factory,
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        assert not cache_path.is_file()

    def test_the_cache_key_ignores_the_llm_and_the_pipeline(self, items, factory, tmp_path):
        """The whole point: five LLMs against the same STT engine should cost
        one transcription, not five."""
        runner.run_one("whisper", "fake-llm-one", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=factory,
                       llm_factory=lambda key, **kw: FakeLLM("first"), results_dir=tmp_path)
        cache_path = runner._transcript_cache_path(tmp_path, "adaptive", "whisper")
        assert cache_path.is_file()

        frame = runner.run_one(
            "whisper", "fake-llm-two", "multimodal", items, preprocessing="adaptive",
            llm_factory=lambda key, **kw: FakeLLM("second"), results_dir=tmp_path)
        assert set(frame["stt_cached"]) == {True}

    def test_a_different_preprocessing_is_a_different_cache_entry(self, items, factory, tmp_path):
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=factory,
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert not runner._transcript_cache_path(tmp_path, "uniform", "whisper").is_file()

    def test_a_different_stt_engine_is_a_different_cache_entry(self, items, factory, tmp_path):
        runner.run_one("whisper", "fake-llm", "separate", items, devices=["cpu"],
                       preprocessing="adaptive", model_factory=factory,
                       llm_factory=lambda key, **kw: FakeLLM("x"), results_dir=tmp_path)
        assert not runner._transcript_cache_path(tmp_path, "adaptive", "seamless").is_file()
