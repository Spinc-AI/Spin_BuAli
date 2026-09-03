"""Windowing, stitching, and the per-device scheduling.

These are the parts of the harness that can quietly corrupt a comparison: a
window boundary that drops audio, a stitch that duplicates it, or a failure that
takes the whole run down with it.
"""
import pytest

import transcribe
from conftest import FakeModel


class TestPlanWindows:
    def test_short_audio_is_one_window(self):
        assert transcribe.plan_windows(10.0, window_sec=28, overlap_sec=3) == [(0.0, 10.0)]

    def test_long_audio_is_covered_end_to_end(self):
        """The failure this guards against is silent: a gap between windows is
        audio no model ever sees, scored as a deletion against every one of
        them."""
        windows = transcribe.plan_windows(120.0, window_sec=28, overlap_sec=3)
        assert windows[0][0] == 0.0
        assert windows[-1][1] == 120.0
        for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
            assert next_start < previous_end, "consecutive windows must overlap"

    def test_windows_overlap_by_the_requested_amount(self):
        windows = transcribe.plan_windows(120.0, window_sec=28, overlap_sec=3)
        assert windows[1][0] == pytest.approx(25.0)

    def test_overlap_must_be_shorter_than_the_window(self):
        with pytest.raises(ValueError):
            transcribe.plan_windows(120.0, window_sec=10, overlap_sec=10)

    def test_exactly_one_window_long(self):
        assert transcribe.plan_windows(28.0, window_sec=28, overlap_sec=3) == [(0.0, 28.0)]


class TestStitch:
    def test_repeated_words_at_the_seam_are_dropped_once(self):
        assert transcribe.stitch(
            ["there is a small stone in the", "stone in the right kidney"]
        ) == "there is a small stone in the right kidney"

    def test_disagreeing_windows_keep_both_halves(self):
        """No shared run means the two windows heard the overlap differently.
        Keeping both is the visible error; silently dropping one is not."""
        assert transcribe.stitch(["the liver is normal", "spleen is enlarged"]) == \
            "the liver is normal spleen is enlarged"

    def test_empty_windows_are_skipped(self):
        assert transcribe.stitch(["", "the liver", "", ""]) == "the liver"

    def test_a_single_window_is_returned_unchanged(self):
        assert transcribe.stitch(["a 6 mm stone"]) == "a 6 mm stone"

    def test_the_lookback_is_bounded(self):
        """A long identical run beyond the cap is left alone rather than
        collapsing two genuinely repeated sentences into one."""
        words = " ".join(str(n) for n in range(50))
        assert transcribe.stitch([words, words], max_overlap_words=5).count("0 1 2") == 2


class TestTranscribeBatch:
    def test_every_item_gets_a_transcript(self, items, factory):
        run = transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=factory)
        assert [t.asset_id for t in run.transcripts] == ["A1", "A2"]
        assert all(t.error is None for t in run.transcripts)

    def test_results_keep_the_input_order_across_devices(self, items, factory):
        """Two devices finish interleaved; the report must not."""
        run = transcribe.transcribe_batch(
            "fake", items, devices=["cpu", "cpu"], model_factory=factory)
        assert [t.asset_id for t in run.transcripts] == ["A1", "A2"]

    def test_the_model_is_loaded_once_per_device_not_once_per_file(self, items, factory):
        transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=factory)
        assert [model.load_count for model in factory.made] == [1]

    def test_each_device_gets_its_own_replica(self, items, factory):
        run = transcribe.transcribe_batch(
            "fake", items, devices=["cpu", "cpu"], model_factory=factory)
        assert len(factory.made) == 2
        assert {t.device for t in run.transcripts} == {"cpu"}
        assert run.devices == ["cpu", "cpu"]

    def test_the_replica_is_unloaded_afterwards(self, items, factory):
        """Two large models resident at once is how a 16 GB T4 runs out."""
        transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=factory)
        assert all(not model.loaded for model in factory.made)

    def test_speed_is_measured(self, items, factory):
        run = transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=factory)
        assert run.summary()["audio_seconds"] == pytest.approx(2.0, abs=0.1)
        assert all(t.audio_seconds > 0 for t in run.transcripts)


class TestFailuresAreRecorded:
    def test_one_bad_recording_does_not_end_the_run(self, items, tone):
        def build(key, device):
            return FakeModel(device=device, fail_on=1)

        run = transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=build)
        failed = [t for t in run.transcripts if t.error]
        assert len(failed) == 1 and len(run.transcripts) == 2
        assert "boom" in failed[0].error

    def test_a_model_that_cannot_load_fails_every_item_with_a_reason(self, items):
        def build(key, device):
            raise RuntimeError("no weights")

        run = transcribe.transcribe_batch("fake", items, devices=["cpu"], model_factory=build)
        assert len(run.transcripts) == 2
        assert all("load failed" in t.error for t in run.transcripts)

    def test_a_missing_audio_file_is_a_recorded_error(self, factory, tmp_path):
        import dataset

        missing = [dataset.Item("gone", tmp_path / "nope.wav", "text")]
        run = transcribe.transcribe_batch("fake", missing, devices=["cpu"], model_factory=factory)
        assert run.transcripts[0].error is not None


class TestDevices:
    def test_an_explicit_spec_is_honoured(self):
        assert transcribe.resolve_devices("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]

    def test_auto_always_returns_at_least_one_device(self):
        assert transcribe.resolve_devices("auto")

    def test_devices_describe_themselves(self):
        described = transcribe.describe_devices(["cpu"])
        assert described[0]["device"] == "cpu"
