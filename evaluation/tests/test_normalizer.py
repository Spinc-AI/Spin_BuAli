"""Normalisation: script, spacing, and spelled-out numbers."""
import pytest

from text_normalizer import normalize, tokenize, words_to_numbers


class TestDigits:
    @pytest.mark.parametrize("text, expected", [
        ("۱۰۷", "107"),          # Persian digits
        ("٤٤", "44"),            # Arabic-Indic digits
        ("107", "107"),          # already ASCII
    ])
    def test_every_digit_script_becomes_ascii(self, text, expected):
        assert normalize(text) == expected

    def test_persian_and_ascii_digits_compare_equal(self):
        assert normalize("کلیه ۱۰۷") == normalize("کلیه 107")

    @pytest.mark.parametrize("text, expected", [
        ("4.2", "4.2"),          # decimal point survives
        ("۴٫۲", "4.2"),          # Persian decimal separator
    ])
    def test_decimals_survive_punctuation_stripping(self, text, expected):
        assert normalize(text) == expected


class TestScript:
    def test_arabic_letter_forms_become_persian(self):
        assert normalize("كيست") == normalize("کیست")

    def test_zwnj_joins_rather_than_splits(self):
        assert normalize("می‌رود") == "میرود"

    def test_diacritics_are_dropped(self):
        assert normalize("کَلیه") == "کلیه"

    def test_case_is_folded(self):
        assert normalize("Hydronephrosis") == "hydronephrosis"


class TestSpelledOutNumbers:
    @pytest.mark.parametrize("text, expected", [
        ("شش", "6"),
        ("بیست و دو", "22"),
        ("صد و هفت", "107"),
        ("six", "6"),
        ("twenty two", "22"),
        ("three hundred", "300"),
    ])
    def test_number_words_become_digits(self, text, expected):
        assert normalize(text) == expected

    def test_persian_and_english_numbers_are_comparable(self):
        """The acceptance criterion: ۶ / 6 / شش / six all agree."""
        forms = ["۶", "6", "شش", "six"]
        assert len({normalize(form) for form in forms}) == 1

    def test_bare_conjunction_is_not_a_number(self):
        """"و" joins number words but is also an ordinary "and"."""
        assert normalize("من و تو") == "من و تو"

    def test_disabling_the_conversion_keeps_the_words(self):
        assert normalize("شش", spell_out_numbers=False) == "شش"


class TestTokenize:
    def test_punctuation_does_not_create_tokens(self):
        assert tokenize("no stone, or mass.") == ["no", "stone", "or", "mass"]

    def test_empty_input_is_empty_output(self):
        assert tokenize("") == []
        assert normalize(None) == ""
