"""Scoring and ranking.

The point of these is that the benchmark reuses `evaluation/` rather than
recomputing anything, and that the parts a benchmark can get wrong on its own
-- dropping failures, scoring unlabelled recordings, ranking backwards -- do not
happen.
"""
import bridge
import leaderboard
import scoring
import transcribe
from dataset import Item


import pytest


def _run(model, transcripts):
    run = transcribe.ModelRun(model=model, model_id=f"fake/{model}", devices=["cpu"])
    for entry in transcripts:
        asset_id, text, error = (*entry, None)[:3] if len(entry) == 2 else entry
        run.transcripts.append(transcribe.Transcript(
            asset_id=asset_id, model=model, text=text, error=error,
            audio_seconds=10.0, elapsed_seconds=5.0))
    return run


class TestScoreRun:
    def test_scores_come_from_the_evaluation_module(self, items):
        """Not a reimplementation: the same call the live service makes."""
        run = _run("m", [("A1", items[0].reference, None)])
        result = scoring.score_run(run, items, bridge.ClinicalTerms())[0]
        assert result["general"]["wer"] == 0.0
        assert result["evaluation"]["metrics_version"] == bridge.METRICS_VERSION

    def test_unlabelled_recordings_are_not_scored(self, tone):
        """They are still transcribed and timed -- they just have nothing to be
        compared against."""
        items = [Item("A1", tone("A1.wav"), None)]
        run = _run("m", [("A1", "anything at all", None)])
        assert scoring.score_run(run, items, bridge.ClinicalTerms()) == []

    def test_a_failed_transcription_scores_as_empty_not_as_absent(self, items):
        """Dropping the row would flatter the model that crashed."""
        run = _run("m", [("A1", "", "CUDA out of memory"), ("A2", items[1].reference, None)])
        results = scoring.score_run(run, items, bridge.ClinicalTerms())
        assert len(results) == 2
        failed = next(r for r in results if r["asset_id"] == "A1")
        assert failed["general"]["hypothesis_words"] == 0
        assert failed["transcription_error"] == "CUDA out of memory"

    def test_speed_travels_with_the_score(self, items):
        run = _run("m", [("A1", items[0].reference, None)])
        assert scoring.score_run(run, items, bridge.ClinicalTerms())[0]["real_time_factor"] == 0.5


class TestScoreAll:
    def test_every_model_gets_a_row(self, items):
        runs = [_run("good", [("A1", items[0].reference, None), ("A2", items[1].reference, None)]),
                _run("bad", [("A1", "completely different words", None),
                             ("A2", "nothing like it either", None)])]
        _, summary = scoring.score_all(runs, items, bridge.ClinicalTerms())
        assert {bucket["model"] for bucket in summary["models"]} == {"good", "bad"}

    def test_speed_is_folded_into_each_model_row(self, items):
        runs = [_run("m", [("A1", items[0].reference, None)])]
        _, summary = scoring.score_all(runs, items, bridge.ClinicalTerms())
        assert summary["models"][0]["speed"]["real_time_factor"] == 0.5

    def test_a_model_with_no_labelled_audio_is_reported_not_dropped(self, tone):
        items = [Item("A1", tone("A1.wav"), None)]
        runs = [_run("m", [("A1", "some text", None)])]
        _, summary = scoring.score_all(runs, items, bridge.ClinicalTerms())
        assert summary["models"] == []
        assert summary["unscored_models"][0]["model"] == "m"

    def test_the_label_census_is_carried_through(self, items, tone):
        items = items + [Item("A3", tone("A3.wav"), None)]
        runs = [_run("m", [("A1", items[0].reference, None)])]
        _, summary = scoring.score_all(runs, items, bridge.ClinicalTerms())
        assert (summary["labelled"], summary["unlabelled"]) == (2, 1)


class TestLeaderboard:
    def _summary(self, items):
        runs = [_run("good", [("A1", items[0].reference, None), ("A2", items[1].reference, None)]),
                _run("bad", [("A1", "completely different words here", None),
                             ("A2", "nothing at all like it either", None)])]
        return scoring.score_all(runs, items, bridge.ClinicalTerms())

    def test_the_better_model_ranks_first(self, items):
        _, summary = self._summary(items)
        assert leaderboard.rows(summary)[0]["model"] == "good"

    def test_accuracy_and_cost_are_on_the_same_row(self, items):
        """A model that wins on WER at four times real time has not won."""
        _, summary = self._summary(items)
        row = leaderboard.rows(summary)[0]
        assert "wer_p50" in row and "real_time_factor" in row and "peak_vram_gb" in row

    def test_the_markdown_table_lists_every_model(self, items):
        _, summary = self._summary(items)
        table = leaderboard.to_markdown(summary)
        assert "good" in table and "bad" in table

    def test_an_empty_summary_renders_without_crashing(self):
        assert leaderboard.to_markdown({"models": []}) == "_no scored models_"

    def test_worst_lists_the_reports_to_look_at(self, items):
        results, _ = self._summary(items)
        worst = leaderboard.worst(results, count=1)
        assert worst[0]["model"] == "bad"

    def test_an_unwritable_output_path_is_caught_before_the_run(self, tmp_path):
        """On Kaggle the repo usually sits on the read-only /kaggle/input mount.
        Discovering that when the results are written means discovering it after
        the GPU time is gone."""
        import pytest

        blocker = tmp_path / "file-not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(OSError, match="/kaggle/working"):
            leaderboard.check_writable(blocker / "results")

    def test_a_writable_path_passes_the_check(self, tmp_path):
        assert leaderboard.check_writable(tmp_path / "results").is_dir()

    def test_a_run_writes_transcripts_alongside_the_scores(self, items, tmp_path):
        """Transcripts are the input to the next labelling round, so they are
        written even where there was nothing to score them against."""
        results, summary = self._summary(items)
        runs = [_run("good", [("A1", "text", None)])]
        leaderboard.write(tmp_path, results, summary, runs)
        for name in ("summary.json", "results.json", "leaderboard.md", "transcripts.json"):
            assert (tmp_path / name).is_file()


class TestBatchViews:
    """The per-model views a comparison needs that a single report cannot give."""

    def _summary(self, items):
        runs = [_run("m", [("A1", items[0].reference), ("A2", items[1].reference)])]
        return scoring.score_all(runs, items, bridge.ClinicalTerms())

    def test_corpus_wer_is_the_headline_and_the_sort_key(self, items):
        _, summary = self._summary(items)
        assert "corpus_wer" in summary["models"][0]["corpus"]
        assert leaderboard.SORT_BY == "corpus_wer"

    def test_wer_is_broken_out_by_recording_length(self, items, tone):
        """Long audio is where windowing and stitching can go wrong; one
        overall WER hides that."""
        run = transcribe.ModelRun(model="m", devices=["cpu"])
        for asset_id, seconds in (("A1", 10.0), ("A2", 300.0)):
            reference = next(i.reference for i in items if i.asset_id == asset_id)
            run.transcripts.append(transcribe.Transcript(
                asset_id=asset_id, model="m", text=reference,
                audio_seconds=seconds, elapsed_seconds=1.0))
        _, summary = scoring.score_all([run], items, bridge.ClinicalTerms())
        buckets = summary["models"][0]["by_duration"]
        assert set(buckets) == {"lt_30s", "gt_2m"}
        assert buckets["lt_30s"]["reports"] == 1

    def test_throughput_is_the_inverse_of_the_real_time_factor(self, items):
        _, summary = self._summary(items)
        row = leaderboard.rows(summary)[0]
        assert row["throughput"] == pytest.approx(1 / row["real_time_factor"], rel=0.01)

    def test_a_run_writes_both_csvs(self, items, tmp_path):
        """CSV because a person opens and sorts a few hundred rows there, not
        in nested JSON."""
        results, summary = self._summary(items)
        leaderboard.write(tmp_path, results, summary)
        assert (tmp_path / "leaderboard.csv").is_file()
        assert (tmp_path / "per_report.csv").is_file()

    def test_the_per_report_csv_carries_the_text_to_look_at(self, items, tmp_path):
        results, _ = self._summary(items)
        row = leaderboard.per_report_rows(results)[0]
        assert row["reference"] and row["hypothesis"]
        assert "wer" in row and "medical_term_f1" in row

    def test_the_csv_is_written_for_excel_to_read_persian(self, items, tmp_path):
        results, summary = self._summary(items)
        leaderboard.write(tmp_path, results, summary)
        assert (tmp_path / "per_report.csv").read_bytes().startswith(b"\xef\xbb\xbf")


class TestSemanticIsBatched:
    def test_the_pairs_go_through_in_one_call(self, items, monkeypatch):
        """Report-by-report scoring means a batch of one against a model that
        takes longer to load than to run -- the difference between a minute
        and an afternoon."""
        calls = []

        def fake_batch(pairs):
            calls.append(len(pairs))
            return [{"bertscore_f1": 0.9, "semantic_similarity": 0.8}] * len(pairs)

        monkeypatch.setattr(bridge, "semantic_batch", fake_batch)
        runs = [_run("m", [("A1", items[0].reference), ("A2", items[1].reference)])]
        results, summary = scoring.score_all(runs, items, bridge.ClinicalTerms(),
                                             include_semantic=True)
        assert calls == [2], "one call for the whole batch"
        assert results[0]["semantic"]["bertscore_f1"] == 0.9
        assert summary["models"][0]["semantic"]["bertscore_f1_mean"] == 0.9

    def test_it_is_off_unless_asked_for(self, items):
        runs = [_run("m", [("A1", items[0].reference)])]
        results, _ = scoring.score_all(runs, items, bridge.ClinicalTerms())
        assert "semantic" not in results[0]
