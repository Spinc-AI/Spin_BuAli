"""Tiering, planning, and surviving a killed session.

The behaviour worth protecting here is what happens when Kaggle stops the
session: finished work stays finished, unfinished work is retried, and nothing
half-written is ever mistaken for a result.
"""
import json

import pytest

import ledger as ledger_module
import plan as plan_module
import session
import tiers
import transcribe
from dataset import Item

CARD = 14.8   # GB usable on one T4 after the CUDA context


class TestPlacement:
    def test_a_model_that_fits_stays_unquantized(self):
        """Tier A is the only tier whose numbers mean exactly what they say."""
        placed = tiers.place("small", 7.0, CARD, cards=2)
        assert placed.tier == "A" and placed.precision == "fp16"

    def test_one_card_is_preferred_over_two_at_equal_precision(self):
        """Sharding pays a PCIe cost per forward pass and occupies both cards."""
        assert tiers.place("tiny", 3.0, CARD, cards=2).cards == 1

    def test_a_model_is_quantized_only_as_far_as_it_must_be(self):
        """int8 before nf4: precision is given up reluctantly, not by default."""
        # 20B: fp16 needs 46 GB (no), int8 needs 23 GB (fits on two cards).
        placed = tiers.place("mid", 20.0, CARD, cards=2)
        assert placed.tier == "B" and placed.precision == "int8"

    def test_a_model_too_large_even_at_4_bit_is_deferred(self):
        placed = tiers.place("huge", 200.0, CARD, cards=2)
        assert placed.tier == "C" and placed.precision is None
        assert not placed.runnable
        assert "unquantized" in placed.describe()

    def test_more_cards_change_the_verdict(self):
        assert tiers.place("mid", 12.0, CARD, cards=1).tier == "B"
        assert tiers.place("mid", 12.0, CARD, cards=2).tier == "A"

    def test_every_quantized_model_also_gets_a_deferred_twin(self):
        """A quantized 32B and a native 8B differ two ways at once; the native
        run of the same model is the control that separates them."""
        placements = tiers.plan_placements({"small": 7.0, "big": 32.0}, CARD, cards=2)
        by_tier = {(p.model, p.tier) for p in placements}
        assert ("big", "B") in by_tier and ("big", "C") in by_tier
        assert ("small", "C") not in by_tier, "nothing to defer for a model that fit"


class TestPlan:
    def _plan(self, **kwargs):
        return plan_module.build(
            ["whisper", "seamless"], ["aya-expanse-8b", "aya-expanse-32b"],
            usable_gb=CARD, cards=2, **kwargs)

    def test_run_ids_are_stable_across_builds(self):
        """Resuming depends on it: re-planning must not renumber the work."""
        assert [r["run_id"] for r in self._plan()] == [r["run_id"] for r in self._plan()]

    def test_run_ids_are_unique(self):
        runs = self._plan(max_slots=2)
        assert len({r["run_id"] for r in runs}) == len(runs)

    def test_configuration_decides_the_id_not_position(self):
        one = self._plan(pipelines=("separate",))
        two = self._plan(pipelines=("separate", "multimodal"))
        shared = {r["run_id"] for r in one} & {r["run_id"] for r in two}
        assert shared, "the same configuration keeps its id when the plan grows"

    def test_tier_a_is_planned_before_b_before_c(self):
        """A short session should spend itself on the trustworthy runs first."""
        tiers_in_order = [run["tier"] for run in self._plan()]
        assert tiers_in_order == sorted(tiers_in_order)

    def test_a_text_only_llm_is_kept_out_of_multimodal(self):
        runs = self._plan(pipelines=("multimodal",))
        assert all(r["llm_model"] in plan_module.AUDIO_CAPABLE for r in runs)

    def test_multimodal_runs_carry_no_stt_engine(self):
        runs = plan_module.build(["whisper"], ["gemma-4-e4b"], CARD, 2,
                                 pipelines=("multimodal",))
        assert runs and all(r["stt_models"] == [] for r in runs)

    def test_slot_combinations_are_order_independent(self):
        """whisper+seamless is one experiment, not two."""
        combos = plan_module.stt_combinations(["a", "b", "c"], 2)
        assert len(combos) == 6 and ("b", "a") not in combos


class TestLedgerSurvivesAKilledSession:
    def test_a_run_counts_as_done_only_once_its_csv_exists(self, tmp_path):
        store = ledger_module.Ledger(tmp_path)
        assert not store.is_done("abc")
        store.record({"run_id": "abc"}, [{"asset_id": "A1", "wer": 0.1}], {"models": []})
        assert store.is_done("abc")

    def test_pending_skips_what_is_already_recorded(self, tmp_path):
        store = ledger_module.Ledger(tmp_path)
        runs = [{"run_id": "a", "tier": "A"}, {"run_id": "b", "tier": "A"}]
        store.record(runs[0], [{"x": 1}], {})
        assert [r["run_id"] for r in store.pending(runs)] == ["b"]

    def test_deferred_runs_are_never_pending(self, tmp_path):
        """Tier C is planned for other hardware; it must not block this one."""
        store = ledger_module.Ledger(tmp_path)
        runs = [{"run_id": "a", "tier": "C"}, {"run_id": "b", "tier": "A"}]
        assert [r["run_id"] for r in store.pending(runs)] == ["b"]

    def test_a_partial_write_never_looks_complete(self, tmp_path):
        """The failure this prevents: a killed session leaving a truncated CSV
        that resume treats as a finished run."""
        path = tmp_path / "runs" / "x.csv"
        path.parent.mkdir(parents=True)

        def explode(handle):
            handle.write("asset_id,wer\n")
            raise KeyboardInterrupt("session killed mid-write")

        with pytest.raises(KeyboardInterrupt):
            ledger_module.write_atomic(path, explode)
        assert not path.exists(), "only the .tmp should exist"
        assert list(path.parent.glob("*.tmp"))

    def test_csv_carries_the_bom_and_json_does_not(self, tmp_path):
        """Excel needs the BOM to read Persian; a BOM in front of JSON breaks
        every parser that reads it back."""
        store = ledger_module.Ledger(tmp_path)
        store.record({"run_id": "r"}, [{"a": "کلیه"}], {"models": []})
        assert store.result_path("r").read_bytes().startswith(b"\xef\xbb\xbf")
        assert not store.summary_path("r").read_bytes().startswith(b"\xef\xbb\xbf")
        json.loads(store.summary_path("r").read_text(encoding="utf-8"))

    def test_results_are_rebuilt_from_disk_not_memory(self, tmp_path):
        store = ledger_module.Ledger(tmp_path)
        store.record({"run_id": "a"}, [{"asset_id": "A1"}], {})
        store.record({"run_id": "b"}, [{"asset_id": "A2"}], {})
        assert len(ledger_module.Ledger(tmp_path).all_rows()) == 2


class TestBudget:
    def test_it_refuses_a_run_that_will_not_finish_in_time(self):
        """Better to leave time unused than be killed with nothing written."""
        budget = ledger_module.Budget(minutes=10)
        assert budget.allows_another(expected_minutes=5)[0] is True
        allowed, reason = budget.allows_another(expected_minutes=40)
        assert allowed is False and "budget" in reason

    def test_a_run_limit_is_honoured(self):
        budget = ledger_module.Budget(max_runs=2)
        for _ in range(2):
            assert budget.allows_another()[0]
            budget.record_run()
        assert budget.allows_another()[0] is False

    def test_no_budget_never_stops(self):
        assert ledger_module.Budget().allows_another(expected_minutes=10_000)[0] is True


class TestWorkingThrough:
    def _items(self, tone):
        return [Item("A1", tone("A1.wav", 1.0), "a 6 mm stone in the right kidney")]

    def _plan(self):
        return plan_module.build(["whisper", "seamless"], ["aya-expanse-8b"],
                                 CARD, 2, pipelines=("separate",),
                                 preprocessing=("adaptive",))

    def test_a_session_writes_a_csv_per_run(self, tone, tmp_path, factory):
        runs = self._plan()
        session.work_through(runs, self._items(tone), tmp_path, model_factory=factory,
                             devices=["cpu"])
        written = {p.stem for p in (tmp_path / "runs").glob("*.csv")}
        assert written == {run["run_id"] for run in runs if run["tier"] != "C"}

    def test_a_second_session_does_not_repeat_the_first(self, tone, tmp_path, factory):
        """The whole point: stop the session, come back, continue."""
        runs, items = self._plan(), self._items(tone)
        first = session.work_through(runs, items, tmp_path, devices=["cpu"],
                                     budget=ledger_module.Budget(max_runs=1),
                                     model_factory=factory)
        second = session.work_through(runs, items, tmp_path, devices=["cpu"],
                                      model_factory=factory)
        assert len(first["performed"]) == 1
        done = {outcome["run_id"] for outcome in first["performed"] + second["performed"]}
        assert len(done) == len(first["performed"]) + len(second["performed"])
        assert second["remaining"] == 0

    def test_a_session_can_be_limited_to_one_tier(self, tone, tmp_path, factory):
        runs = plan_module.build(["whisper"], ["aya-expanse-8b", "aya-expanse-32b"],
                                 CARD, 2, pipelines=("separate",),
                                 preprocessing=("adaptive",))
        report = session.work_through(runs, self._items(tone), tmp_path, tier="A",
                                      devices=["cpu"], model_factory=factory)
        assert report["status"]["by_tier"]["A"]["pending"] == 0
        assert report["status"]["by_tier"]["B"]["pending"] > 0, "tier B left alone"

    def test_the_run_configuration_is_stamped_on_every_row(self, tone, tmp_path, factory):
        """A row has to say which run produced it once the CSVs are combined."""
        runs = self._plan()
        session.work_through(runs, self._items(tone), tmp_path, model_factory=factory,
                             devices=["cpu"])
        rows = ledger_module.Ledger(tmp_path).all_rows()
        assert all(row["run_id"] and row["tier"] and row["preprocessing"] for row in rows)

    def test_combining_rebuilds_the_leaderboard_from_the_files(self, tone, tmp_path, factory):
        session.work_through(self._plan(), self._items(tone), tmp_path,
                             model_factory=factory, devices=["cpu"])
        result = session.combine(tmp_path)
        assert result["runs"] > 0
        assert (tmp_path / "leaderboard.csv").is_file()
        assert (tmp_path / "all_reports.csv").is_file()
