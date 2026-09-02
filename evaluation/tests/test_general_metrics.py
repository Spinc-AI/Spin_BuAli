"""Text-level metrics: chrF, and the three failure-shape signals."""
import pytest

from general_metrics import (
    chrf,
    hallucination_ratio,
    punctuation_f1,
    repetition_score,
    script_contamination,
)
from medical_metrics import evaluate


class TestChrf:
    def test_identical_text_scores_one(self):
        assert chrf("a 6 mm stone", "a 6 mm stone") == 1.0

    def test_unrelated_text_scores_low(self):
        assert chrf("the kidney is normal", "xyz qrs") < 0.2

    def test_partial_credit_where_wer_gives_none(self):
        """A morphological variant shares most of its characters. WER counts it
        as fully wrong; chrF should not."""
        score = chrf("کلیه‌ها طبیعی هستند", "کلیه طبیعی است")
        assert 0.2 < score < 1.0

    def test_empty_inputs(self):
        assert chrf("", "") == 1.0
        assert chrf("something", "") == 0.0


class TestRepetitionScore:
    def test_a_loop_scores_near_one(self):
        """The observed failure: a model emitting one sentence over and over.
        Every clinical metric stays quiet because the repeated content is
        individually plausible -- this is the only signal that catches it."""
        looped = "There is a small right kidney stone. " * 60
        assert repetition_score(looped) > 0.95

    def test_normal_text_scores_near_zero(self):
        text = "There is a 6 mm stone in the distal right ureter and the prostate is normal"
        assert repetition_score(text) == 0.0

    def test_text_shorter_than_the_window_is_not_repetitive(self):
        assert repetition_score("two words") == 0.0


class TestHallucinationRatio:
    def test_equal_lengths_score_one(self):
        assert hallucination_ratio("a b c d", "w x y z") == 1.0

    def test_padded_output_scores_above_one(self):
        """Catches fabrication built from words outside the clinical
        vocabulary, which the concept-based metrics cannot see."""
        ratio = hallucination_ratio(
            "Bladder normal.",
            "The patient is a 68 year old male with chronic obstructive pulmonary disease.")
        assert ratio > 3

    def test_empty_output_scores_zero(self):
        assert hallucination_ratio("the kidney is normal", "") == 0.0

    def test_empty_reference_does_not_divide_by_zero(self):
        assert hallucination_ratio("", "anything") == 0.0


class TestPunctuationF1:
    def test_matching_punctuation_scores_one(self):
        assert punctuation_f1("no stone, or mass.", "no stone, or mass.") == 1.0

    def test_bare_text_against_punctuated_scores_zero(self):
        """Models that emit bare text (Wav2Vec2, MMS) are otherwise not
        comparable with punctuating ones (Whisper)."""
        assert punctuation_f1("no stone, or mass.", "no stone or mass") == 0.0

    def test_neither_punctuating_counts_as_agreement(self):
        assert punctuation_f1("no stone or mass", "no stone or mass") == 1.0

    def test_persian_punctuation_is_recognised(self):
        assert punctuation_f1("سنگ، توده؟", "سنگ، توده؟") == 1.0


class TestInTheReport:
    @pytest.mark.parametrize("field", [
        "chrf", "hallucination_ratio", "repetition_score",
        "punctuation_f1", "hypothesis_words",
        "script_contamination", "reference_script_contamination",
        "character_errors", "reference_chars",
    ])
    def test_new_fields_are_reported(self, terms, field):
        assert field in evaluate("a 6 mm stone", "a 6 mm stone", terms)["general"]

    def test_semantic_block_is_absent_unless_requested(self, terms):
        assert "semantic" not in evaluate("a", "a", terms)


class TestDegenerateOutput:
    """A broken output must be flagged even when the clinical counters agree."""

    def test_a_looping_output_requires_review(self, terms):
        result = evaluate("There is a small right kidney stone. " * 40,
                          "There is a small right kidney stone.", terms)
        assert result["clinical_counts"]["negation_errors"] == 0, "no clinical error fires"
        assert result["requires_medical_review"] is True
        assert "repetition" in result["review_reasons"]

    def test_a_padded_output_requires_review(self, terms):
        result = evaluate(
            "The patient is a 68 year old male with chronic obstructive pulmonary disease.",
            "Bladder normal.", terms)
        assert "length_ratio" in result["review_reasons"]

    def test_a_truncated_output_requires_review(self, terms):
        result = evaluate("normal", "There is a 6 mm stone in the distal right ureter "
                                    "and the prostate measures 23 cc", terms)
        assert "length_ratio" in result["review_reasons"]

    def test_a_normal_report_is_not_flagged(self, terms):
        text = "There is a 6 mm stone in the distal right ureter."
        result = evaluate(text, text, terms)
        assert result["requires_medical_review"] is False


class TestScriptContamination:
    """Share of letters that are not Arabic-script.

    Kept deliberately simple, and reported beside the reference's own figure,
    because on this corpus the absolute number is not interpretable on its own.
    """

    def test_pure_persian_scores_zero(self):
        assert script_contamination("کلیه راست طبیعی است") == 0.0

    def test_pure_latin_scores_one(self):
        assert script_contamination("the right kidney is normal") == 1.0

    def test_mixed_text_scores_the_share(self):
        assert script_contamination("کلیه normal") == pytest.approx(6 / 10, abs=0.01)

    def test_digits_and_punctuation_are_not_letters(self):
        """Only alphabetic characters count, so "۶ mm" is judged on "mm"."""
        assert script_contamination("۶ mm، ۷") == 1.0
        assert script_contamination("۶ ۷ ۸") == 0.0

    def test_empty_text_scores_zero(self):
        assert script_contamination("") == 0.0

    def test_a_correct_code_switched_transcript_scores_high(self):
        """The documented limitation, pinned as a test so it is never mistaken
        for a bug: this radiologist dictates English terms on purpose, so a
        perfect transcript is 'contaminated'."""
        assert script_contamination("در sonography کلیه راست stone دیده شد") > 0.4

    def test_the_reference_figure_is_reported_alongside(self, terms):
        """Which is what makes the hypothesis figure readable -- a model
        drifting into Latin is a gap between the two, not a high number."""
        text = "در sonography کلیه راست stone دیده شد"
        general = evaluate(text, text, terms)["general"]
        assert general["script_contamination"] == general["reference_script_contamination"]

    def test_a_model_answering_in_english_shows_as_a_gap(self):
        reference = "کلیه راست طبیعی است"
        assert (script_contamination("the right kidney is normal")
                - script_contamination(reference)) == 1.0
