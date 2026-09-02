"""The metrics that only exist across many reports.

A single report cannot have a 90th-percentile WER or a corpus error rate, so
these live in `evaluate_results.py` rather than in `evaluate()`. The rule that
matters most here is the one about denominators: a rate over a batch is the
summed errors over the summed opportunities, never the mean of the per-report
rates.
"""
import pytest

from evaluate_results import summarize
from medical_metrics import evaluate


def scored(model, pairs, terms):
    """Score real report pairs -- the aggregation is only meaningful over
    output the metrics actually produced."""
    return [{"asset_id": f"A{index}", "model": model, "pipeline": None,
             "model_version": None, **evaluate(hypothesis, reference, terms)}
            for index, (hypothesis, reference) in enumerate(pairs)]


class TestDenominatorsAreReported:
    """Without these, no clinical rate can be rebuilt over a batch."""

    @pytest.mark.parametrize("field", [
        "negation_scored", "laterality_scored",
        "critical_omission_scored", "unsupported_addition_scored",
        "hypothesis_measurements",
    ])
    def test_the_denominator_is_in_the_report(self, terms, field):
        counts = evaluate("a 6 mm stone in the right kidney",
                          "a 6 mm stone in the right kidney", terms)["clinical_counts"]
        assert field in counts

    def test_a_rate_equals_its_errors_over_its_denominator(self, terms):
        result = evaluate("There is a mass in the right kidney",
                          "There is no mass in the right kidney", terms)
        counts, metrics = result["clinical_counts"], result["clinical_metrics"]
        assert counts["negation_scored"] > 0
        assert metrics["negation_error_rate"] == pytest.approx(
            counts["negation_errors"] / counts["negation_scored"], abs=1e-4)


class TestRatesComeFromSummedCounts:
    def test_a_short_report_does_not_dominate(self, terms):
        """The failure this prevents: one negation error in a two-word report
        is a 100% rate. Averaged against a long clean report it reads as 50%
        wrong; over summed counts it is one error in many opportunities."""
        results = scored("m", [
            ("no stone", "a stone"),
            ("There is no mass, no stone and no hydronephrosis in the right kidney "
             "and the urinary bladder is normal",
             "There is no mass, no stone and no hydronephrosis in the right kidney "
             "and the urinary bladder is normal"),
        ], terms)
        rate = summarize(results, terms)["models"][0]["rates"]["negation_error_rate"]
        mean_of_rates = sum(r["clinical_metrics"]["negation_error_rate"]
                            for r in results) / len(results)
        assert rate < mean_of_rates

    @pytest.mark.parametrize("rate", [
        "negation_error_rate", "laterality_error_rate", "number_error_rate",
        "unit_error_rate", "critical_omission_rate", "unsupported_addition_rate",
        "medical_term_precision", "medical_term_recall", "medical_term_f1",
        "review_rate",
    ])
    def test_every_clinical_rate_aggregates(self, terms, rate):
        results = scored("m", [("a 6 mm stone", "a 7 mm stone")], terms)
        assert rate in summarize(results, terms)["models"][0]["rates"]

    def test_a_perfect_batch_has_zero_error_rates(self, terms):
        text = "There is a 6 mm stone in the right kidney and no mass"
        rates = summarize(scored("m", [(text, text)], terms), terms)["models"][0]["rates"]
        assert rates["negation_error_rate"] == 0.0
        assert rates["number_error_rate"] == 0.0
        assert rates["medical_term_f1"] == 1.0


class TestTextDistribution:
    def test_typical_quality_metrics_report_a_mean(self, terms):
        text = summarize(scored("m", [("a 6 mm stone", "a 7 mm stone")], terms),
                         terms)["models"][0]["text"]
        for field in ("cer_mean", "chrf_mean", "punctuation_f1_mean"):
            assert field in text

    def test_failure_detectors_report_the_tail_as_well(self, terms):
        """Their mean is near zero even when a model loops on one report in
        twenty, so the mean alone would hide exactly what they exist to find."""
        text = summarize(scored("m", [("a 6 mm stone", "a 7 mm stone")], terms),
                         terms)["models"][0]["text"]
        for field in ("repetition_score_p95", "repetition_score_max",
                      "hallucination_ratio_p95", "hallucination_ratio_max"):
            assert field in text

    def test_one_looping_report_among_clean_ones_shows_in_the_tail(self, terms):
        clean = "There is a 6 mm stone in the right kidney"
        results = scored("m", [(clean, clean)] * 9 + [(clean + " " + clean * 40, clean)], terms)
        text = summarize(results, terms)["models"][0]["text"]
        assert text["repetition_score_max"] > 0.9
        assert text["repetition_score_mean"] < text["repetition_score_max"]


class TestReliability:
    def test_the_distribution_metrics_are_present(self, terms):
        results = scored("m", [("a 6 mm stone", "a 7 mm stone"), ("x", "a 6 mm stone")], terms)
        reliability = summarize(results, terms)["models"][0]["reliability"]
        for field in ("ser", "wer_p50", "wer_p90", "wer_p95",
                      "pct_perfect", "pct_catastrophic", "empty_output_rate"):
            assert field in reliability

    def test_an_empty_transcript_counts_towards_the_empty_rate(self, terms):
        results = scored("m", [("", "a 6 mm stone")], terms)
        assert summarize(results, terms)["models"][0]["reliability"]["empty_output_rate"] == 1.0

    def test_a_perfect_batch_is_all_perfect_and_no_errors(self, terms):
        text = "a 6 mm stone"
        reliability = summarize(scored("m", [(text, text)], terms),
                                terms)["models"][0]["reliability"]
        assert reliability["pct_perfect"] == 1.0
        assert reliability["ser"] == 0.0


class TestModelsAreSeparated:
    def test_each_model_gets_its_own_bucket(self, terms):
        results = (scored("good", [("a 6 mm stone", "a 6 mm stone")], terms)
                   + scored("bad", [("nothing similar", "a 6 mm stone")], terms))
        buckets = {b["model"]: b for b in summarize(results, terms)["models"]}
        assert buckets["good"]["reliability"]["wer_p50"] < buckets["bad"]["reliability"]["wer_p50"]

    def test_the_ruler_is_recorded(self, terms):
        """Results only compare across reports scored with the same versions."""
        summary = summarize(scored("m", [("a", "a")], terms), terms)
        assert summary["evaluation"]["terms_sha"] == terms.sha
        assert summary["evaluation"]["metrics_version"]


class TestCorpusRates:
    """The headline number, and the one most easily got wrong."""

    def test_corpus_wer_is_not_the_mean_of_report_wers(self, terms):
        """A one-line report and a long one carry equal weight in a mean, so a
        model that fails on the short one looks far worse than it is. The
        corpus figure weights by how much was actually said."""
        long_reference = ("There is a 6 mm stone in the distal right ureter and no "
                          "hydronephrosis and the urinary bladder is normal and the "
                          "prostate measures 23 cc")
        results = scored("m", [("wrong", "normal"), (long_reference, long_reference)], terms)
        corpus = summarize(results, terms)["models"][0]["corpus"]["corpus_wer"]
        mean_of_reports = sum(r["general"]["wer"] for r in results) / len(results)
        assert corpus < mean_of_reports

    def test_corpus_wer_is_total_edits_over_total_words(self, terms):
        results = scored("m", [("a 6 mm stone", "a 7 mm stone"),
                               ("the liver", "the spleen")], terms)
        corpus = summarize(results, terms)["models"][0]["corpus"]
        edits = sum(r["general"]["substitutions"] + r["general"]["insertions"]
                    + r["general"]["deletions"] for r in results)
        words = sum(r["general"]["reference_words"] for r in results)
        assert corpus["corpus_wer"] == pytest.approx(edits / words, abs=1e-4)

    def test_corpus_cer_uses_character_counts(self, terms):
        results = scored("m", [("a 6 mm stone", "a 7 mm stone")], terms)
        corpus = summarize(results, terms)["models"][0]["corpus"]
        assert corpus["reference_chars"] > corpus["reference_words"]
        assert corpus["corpus_cer"] == pytest.approx(
            corpus["character_errors"] / corpus["reference_chars"], abs=1e-4)

    def test_the_edit_breakdown_sums_to_the_word_error_rate(self, terms):
        results = scored("m", [("a stone", "a 6 mm stone"), ("x y z", "x")], terms)
        corpus = summarize(results, terms)["models"][0]["corpus"]
        assert (corpus["substitution_rate"] + corpus["insertion_rate"]
                + corpus["deletion_rate"]) == pytest.approx(corpus["corpus_wer"], abs=1e-3)

    def test_a_perfect_batch_has_zero_corpus_error(self, terms):
        text = "There is a 6 mm stone in the right kidney"
        corpus = summarize(scored("m", [(text, text)], terms), terms)["models"][0]["corpus"]
        assert corpus["corpus_wer"] == 0.0 and corpus["corpus_cer"] == 0.0

    def test_insertions_are_reported_separately(self, terms):
        """In a medical transcript an insertion is the fabrication signal, not
        just noise, so it gets its own rate rather than hiding inside WER."""
        results = scored("m", [("a 6 mm stone and a large mass and free fluid",
                                "a 6 mm stone")], terms)
        corpus = summarize(results, terms)["models"][0]["corpus"]
        assert corpus["insertion_rate"] > 0
