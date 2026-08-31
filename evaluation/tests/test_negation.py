"""Negation detection and the negation error rate."""
import pytest

from extractors import is_negated
from medical_metrics import evaluate
from text_normalizer import tokenize


def negated_words(text):
    tokens = tokenize(text)
    return {token for index, token in enumerate(tokens) if is_negated(tokens, index)}


class TestEnglishNegation:
    @pytest.mark.parametrize("text, word", [
        ("no stone or mass", "stone"),
        ("without sign of acute cholecystitis", "cholecystitis"),
        ("there is no evidence of hydronephrosis", "hydronephrosis"),
        ("the study is negative for calculus", "calculus"),
    ])
    def test_cue_negates_what_follows(self, text, word):
        assert word in negated_words(text)

    def test_affirmed_finding_is_not_negated(self):
        assert "stone" not in negated_words("there is a 6 mm stone")


class TestPersianNegation:
    @pytest.mark.parametrize("text, word", [
        ("سنگ وجود ندارد", "سنگ"),        # verb-final negation
        ("توده مشاهده نشد", "توده"),
        ("بدون سنگ", "سنگ"),              # pre-posed cue
    ])
    def test_cue_negates_what_precedes_it(self, text, word):
        assert word in negated_words(text)

    def test_romanised_persian_cue_is_recognised(self):
        """Transcripts contain romanised Persian, e.g. "adenopathy nadarad"."""
        assert "adenopathy" in negated_words("adenopathy retroperitoneal nadarad")


class TestScope:
    def test_scope_stops_at_a_contrast_word(self):
        assert negated_words("no stone but mass is seen") == {"no", "stone"}

    def test_distant_words_are_outside_the_window(self):
        text = "no stone " + " ".join(["filler"] * 10) + " mass"
        assert "mass" not in negated_words(text)


class TestNegationErrorRate:
    def test_a_flip_is_counted(self, terms):
        result = evaluate("there is a stone in the bladder",
                          "no stone in the bladder", terms)
        assert result["clinical_counts"]["negation_errors"] == 1
        assert result["clinical_metrics"]["negation_error_rate"] == 1.0
        assert result["requires_medical_review"] is True

    def test_agreement_scores_zero(self, terms):
        result = evaluate("no stone in the bladder", "no stone in the bladder", terms)
        assert result["clinical_counts"]["negation_errors"] == 0

    def test_flip_is_reported_as_a_critical_error(self, terms):
        result = evaluate("there is a stone", "no stone", terms)
        flip = next(e for e in result["critical_errors"] if e["type"] == "negation_flip")
        assert flip["reference"] == "negative"
        assert flip["prediction"] == "positive"

    def test_anatomy_is_not_scored_for_negation(self, terms):
        """A report says "no stone", never "no kidney" -- scoring anatomy would
        just measure how far the cue window reached."""
        result = evaluate("no stone in the urinary bladder",
                          "a stone in the urinary bladder", terms)
        # The stone flipped; the bladder must not also be counted.
        assert result["clinical_counts"]["negation_errors"] == 1
