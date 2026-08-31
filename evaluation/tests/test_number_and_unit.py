"""Measurement extraction, and the number / unit error rates.

These two are scored separately but computed together, because the only way to
tell a misheard number from a misheard unit is to compare physical quantities
first and then look at what actually differs.
"""
import pytest

from extractors import extract_measurements
from medical_metrics import align_measurements, classify_measurement_error, evaluate


def one(text):
    measurements = extract_measurements(text)
    assert len(measurements) == 1, f"expected one measurement, got {measurements}"
    return measurements[0]


class TestExtraction:
    @pytest.mark.parametrize("text, value, unit", [
        ("a 6 mm stone", 6.0, "mm"),
        ("a 1 cm stone", 1.0, "cm"),
        ("prostate is 23 cc", 23.0, "cc"),
        ("diffuse thick 4.2 mm wall", 4.2, "mm"),
        ("سنگ شش میلی‌متر", 6.0, "میلیمتر"),
    ])
    def test_value_and_unit_are_read(self, text, value, unit):
        measurement = one(text)
        assert (measurement.value, measurement.unit) == (value, unit)

    def test_a_bare_number_is_not_a_measurement(self):
        """In Persian the word for "one" is also the indefinite article, so
        counting every digit would invent measurements."""
        assert extract_measurements("یک کیست ساده") == []
        assert extract_measurements("grade 2 fatty liver") == []

    def test_dimension_groups_are_kept_without_a_unit(self):
        """Radiologists dictate "107 در 44" with the unit implied."""
        assert [m.value for m in extract_measurements("right kidney 107 x 44")] == [107.0, 44.0]
        assert [m.value for m in extract_measurements("کلیه راست ۱۰۷ در ۴۴")] == [107.0, 44.0]

    def test_a_trailing_unit_covers_the_whole_group(self):
        measurements = extract_measurements("50 x 56 x 66 mm")
        assert [m.unit for m in measurements] == ["mm"] * 3
        assert [m.canonical for m in measurements] == [50.0, 56.0, 66.0]


class TestPhysicalQuantities:
    def test_the_same_size_in_different_units_is_equal(self):
        assert one("10 mm stone").same_quantity(one("1 cm stone"))

    def test_different_sizes_are_not_equal(self):
        assert not one("6 mm stone").same_quantity(one("7 mm stone"))

    def test_length_never_matches_volume(self):
        assert not one("23 mm").same_quantity(one("23 cc"))


class TestErrorClassification:
    def test_same_quantity_is_no_error(self):
        assert classify_measurement_error(one("10 mm"), one("1 cm")) == set()

    def test_same_literal_different_unit_is_a_unit_error(self):
        assert classify_measurement_error(one("6 mm"), one("6 cm")) == {"unit"}

    def test_same_unit_different_literal_is_a_number_error(self):
        assert classify_measurement_error(one("6 mm"), one("7 mm")) == {"number"}

    def test_both_wrong_counts_as_both(self):
        assert classify_measurement_error(one("6 mm"), one("7 cm")) == {"number", "unit"}


class TestAlignment:
    def test_correct_measurements_out_of_order_still_match(self):
        reference = extract_measurements("6 mm stone and 23 cc prostate")
        hypothesis = extract_measurements("23 cc prostate and 6 mm stone")
        pairs, missing, extra = align_measurements(reference, hypothesis)
        assert len(pairs) == 2 and not missing and not extra
        assert all(classify_measurement_error(r, h) == set() for r, h in pairs)

    def test_a_wrong_value_pairs_up_instead_of_becoming_add_plus_omit(self):
        """The double-counting trap: one wrong number must not read as one
        omission plus one unsupported addition."""
        pairs, missing, extra = align_measurements(
            extract_measurements("a 7 mm calculus"),
            extract_measurements("a 6 mm stone"))
        assert len(pairs) == 1 and not missing and not extra


class TestRates:
    def test_wrong_number_scores_one_number_error(self, terms):
        result = evaluate("There is a 6 mm stone in the distal right ureter.",
                          "There is a 7 mm calculus in the distal right ureter.", terms)
        counts = result["clinical_counts"]
        assert counts["number_errors"] == 1
        assert counts["unit_errors"] == 0
        assert counts["critical_omissions"] == 0
        assert counts["unsupported_additions"] == 0
        assert result["clinical_metrics"]["number_error_rate"] == 1.0

    def test_equivalent_units_score_clean(self, terms):
        result = evaluate("a 10 mm stone", "a 1 cm stone", terms)
        counts = result["clinical_counts"]
        assert (counts["number_errors"], counts["unit_errors"]) == (0, 0)
        assert result["requires_medical_review"] is False

    def test_a_unit_error_requires_review(self, terms):
        """6 cm for 6 mm is a tenfold error and must not pass quietly."""
        result = evaluate("a 6 cm stone", "a 6 mm stone", terms)
        assert result["clinical_counts"]["unit_errors"] == 1
        assert result["requires_medical_review"] is True

    def test_a_dropped_measurement_is_a_critical_omission(self, terms):
        result = evaluate("The kidney is normal.",
                          "There is a 6 mm stone in the right kidney.", terms)
        assert result["clinical_counts"]["critical_omissions"] >= 1

    def test_no_measurements_gives_a_zero_rate_not_an_error(self, terms):
        result = evaluate("the kidney is normal", "the kidney is normal", terms)
        assert result["clinical_metrics"]["number_error_rate"] == 0.0
