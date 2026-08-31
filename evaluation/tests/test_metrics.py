"""End-to-end scoring: term matching, general metrics, and the review flag."""
from general_metrics import character_error_rate, word_error_rate
from medical_metrics import METRICS_VERSION, evaluate


class TestGeneralMetrics:
    def test_identical_text_scores_zero(self):
        assert word_error_rate("a 6 mm stone", "a 6 mm stone").rate == 0.0
        assert character_error_rate("a 6 mm stone", "a 6 mm stone").rate == 0.0

    def test_operations_are_broken_out(self):
        counts = word_error_rate("the kidney is normal", "the kidney is enlarged")
        assert counts.substitutions == 1
        assert counts.insertions == 0 and counts.deletions == 0

    def test_an_insertion_is_distinguished_from_a_substitution(self):
        counts = word_error_rate("the kidney", "the right kidney")
        assert counts.insertions == 1 and counts.substitutions == 0

    def test_empty_reference_does_not_divide_by_zero(self):
        assert word_error_rate("", "anything").rate == 0.0


class TestTermMatching:
    def test_synonyms_count_as_the_same_concept(self, terms):
        """"stone" and "calculus" are one concept, so neither is a miss."""
        result = evaluate("a 6 mm stone", "a 6 mm calculus", terms)
        metrics = result["clinical_metrics"]
        assert metrics["medical_term_precision"] == 1.0
        assert metrics["medical_term_recall"] == 1.0

    def test_persian_and_english_terms_match(self, terms):
        result = evaluate("کلیه راست ۱۰۷ در ۴۴ و سنگ شش میلی‌متر",
                          "right kidney 107 x 44 with 6 mm stone", terms)
        counts = result["clinical_counts"]
        assert counts["false_positive_terms"] == 0
        assert counts["false_negative_terms"] == 0
        assert counts["number_errors"] == 0

    def test_a_missing_finding_lowers_recall(self, terms):
        result = evaluate("the bladder is normal",
                          "the bladder is normal and there is a cyst", terms)
        assert result["clinical_metrics"]["medical_term_recall"] < 1.0

    def test_an_invented_finding_lowers_precision(self, terms):
        result = evaluate("the bladder is normal and there is a cyst",
                          "the bladder is normal", terms)
        assert result["clinical_metrics"]["medical_term_precision"] < 1.0

    def test_the_longest_phrase_wins(self, terms):
        """"fatty liver" must not also be counted as bare "liver"."""
        result = evaluate("fatty liver", "fatty liver", terms)
        assert result["clinical_counts"]["reference_entities"] == 1


class TestFabrication:
    def test_invented_history_is_an_unsupported_addition(self, terms):
        """The observed failure: a model inventing a history never dictated."""
        result = evaluate(
            "The patient has hepatomegaly and a cyst. Bladder normal.",
            "Bladder normal.", terms)
        assert result["clinical_counts"]["unsupported_additions"] >= 2
        assert result["requires_medical_review"] is True


class TestReviewFlag:
    def test_a_clean_report_needs_no_review(self, terms):
        text = "There is a 6 mm stone in the distal right ureter."
        result = evaluate(text, text, terms)
        assert result["requires_medical_review"] is False
        assert result["review_reasons"] == []

    def test_reasons_name_what_triggered_it(self, terms):
        result = evaluate("a stone in the left kidney",
                          "no stone in the right kidney", terms)
        assert "negation_errors" in result["review_reasons"]
        assert "laterality_errors" in result["review_reasons"]


class TestVersionStamp:
    def test_every_result_records_the_ruler_that_measured_it(self, terms):
        evaluation = evaluate("a", "a", terms)["evaluation"]
        assert evaluation["metrics_version"] == METRICS_VERSION
        assert evaluation["terms_version"] == terms.version
        assert len(evaluation["terms_sha"]) == 8

    def test_the_terms_hash_tracks_the_file(self, terms, tmp_path):
        """Editing the vocabulary changes the hash, so two results measured
        with different term lists can never look identical."""
        import json
        from extractors import ClinicalTerms

        path = tmp_path / "terms.json"
        path.write_text(json.dumps({
            "version": "test", "concepts": [
                {"id": "kidney", "category": "anatomy", "variants": ["kidney"]}]
        }), encoding="utf-8")
        assert ClinicalTerms(path).sha != terms.sha
