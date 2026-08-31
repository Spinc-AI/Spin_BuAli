"""Laterality detection and the laterality error rate."""
import pytest

from extractors import laterality_at
from medical_metrics import evaluate
from text_normalizer import tokenize


def side_of(text, word):
    tokens = tokenize(text)
    return laterality_at(tokens, tokens.index(word))


class TestDetection:
    @pytest.mark.parametrize("text, word, expected", [
        ("right kidney", "kidney", "right"),          # English, pre-posed
        ("کلیه راست", "کلیه", "right"),                # Persian, post-posed
        ("left lower pole", "pole", "left"),
        ("کلیه چپ", "کلیه", "left"),
        ("bilateral kidneys", "kidneys", "bilateral"),
        ("both kidneys", "kidneys", "bilateral"),
    ])
    def test_marker_is_found_in_either_language_and_order(self, text, word, expected):
        assert side_of(text, word) == expected

    def test_abbreviations_are_recognised(self):
        assert side_of("rt kidney", "kidney") == "right"
        assert side_of("lt kidney", "kidney") == "left"

    def test_absent_marker_reads_as_none(self):
        assert side_of("the spleen is enlarged", "spleen") is None

    def test_nearest_marker_wins(self):
        tokens = tokenize("right ureter and the left kidney")
        assert laterality_at(tokens, tokens.index("kidney")) == "left"


class TestLateralityErrorRate:
    def test_a_flip_is_counted(self, terms):
        result = evaluate("hydronephrosis in the left kidney",
                          "hydronephrosis in the right kidney", terms)
        assert result["clinical_counts"]["laterality_errors"] > 0
        assert result["clinical_metrics"]["laterality_error_rate"] == 1.0
        assert result["requires_medical_review"] is True

    def test_agreement_scores_zero(self, terms):
        text = "hydronephrosis in the right kidney"
        assert evaluate(text, text, terms)["clinical_counts"]["laterality_errors"] == 0

    def test_flip_is_reported_as_a_critical_error(self, terms):
        result = evaluate("stone in the left kidney", "stone in the right kidney", terms)
        flip = next(e for e in result["critical_errors"] if e["type"] == "laterality_flip")
        assert flip["reference"] == "right"
        assert flip["prediction"] == "left"

    def test_a_dropped_side_counts_as_an_error(self, terms):
        result = evaluate("stone in the kidney", "stone in the right kidney", terms)
        assert result["clinical_counts"]["laterality_errors"] > 0

    def test_negation_and_laterality_are_scored_independently(self, terms):
        """The acceptance criterion: the two must not be conflated."""
        result = evaluate("no stone in the left kidney",
                          "a stone in the right kidney", terms)
        counts = result["clinical_counts"]
        assert counts["negation_errors"] == 1
        assert counts["laterality_errors"] > 0
