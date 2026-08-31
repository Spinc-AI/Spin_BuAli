"""Normalisation — the step every metric depends on.

Both sides of a comparison go through `normalize()` before anything is counted,
so ۱۰۷ and 107, "می‌رود" and "میرود", and "شش" and "6" are not scored as
differences. Nothing downstream should have to think about script or spacing.

The rules are deliberately conservative: they unify representations of the
*same* token and never rewrite clinical content.
"""
import re
import unicodedata

# --- Character tables ------------------------------------------------------
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"      # U+06F0..U+06F9
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"       # U+0660..U+0669
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(PERSIAN_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(ARABIC_DIGITS)})

# Arabic letter forms that Persian keyboards and STT engines emit
# interchangeably with their Persian counterparts.
_LETTER_MAP = {
    "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
    "ؤ": "و", "ئ": "ی",
}

# Harakat/tanwin, superscript alef, and kashida carry no lexical weight here.
_DIACRITICS = re.compile("[ً-ْٰـ]")

# Zero-width and bidi controls. ZWNJ (U+200C) is handled separately below.
_INVISIBLE = re.compile("[‍‎‏‪-‮⁦-⁩﻿]")
ZWNJ = "‌"

_PUNCTUATION = re.compile(r"[^\w\s.]", re.UNICODE)
# A dot only survives when it sits between two digits (a decimal point).
_STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_WHITESPACE = re.compile(r"\s+")

# --- Spelled-out numbers ---------------------------------------------------
_PERSIAN_UNITS = {
    "صفر": 0, "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "شیش": 6,
    "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12,
    "سیزده": 13, "چهارده": 14, "پانزده": 15, "پونزده": 15, "شانزده": 16,
    "شونزده": 16, "هفده": 17, "هیفده": 17, "هجده": 18, "هیجده": 18, "نوزده": 19,
}
_PERSIAN_TENS = {
    "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60,
    "هفتاد": 70, "هشتاد": 80, "نود": 90,
}
_PERSIAN_HUNDREDS = {
    "صد": 100, "یکصد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400,
    "پانصد": 500, "پونصد": 500, "ششصد": 600, "هفتصد": 700,
    "هشتصد": 800, "نهصد": 900,
}
_PERSIAN_SCALES = {"هزار": 1000, "میلیون": 1000000}

_ENGLISH_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_ENGLISH_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ENGLISH_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}

# "و" joins number words in Persian ("بیست و دو"), but is also an ordinary
# conjunction, so it only counts as a joiner between two number words.
_PERSIAN_JOINER = "و"
_NUMBER_WORDS = (
    set(_PERSIAN_UNITS) | set(_PERSIAN_TENS) | set(_PERSIAN_HUNDREDS)
    | set(_PERSIAN_SCALES) | set(_ENGLISH_UNITS) | set(_ENGLISH_TENS)
    | set(_ENGLISH_SCALES)
)


def _word_value(word):
    for table in (_PERSIAN_UNITS, _PERSIAN_TENS, _PERSIAN_HUNDREDS,
                  _ENGLISH_UNITS, _ENGLISH_TENS):
        if word in table:
            return table[word]
    return None


def _scale_value(word):
    return _PERSIAN_SCALES.get(word) or _ENGLISH_SCALES.get(word)


def _collapse_number_run(words):
    """Fold a run of number words into one integer, or None if it isn't one.

    Handles additive forms ("بیست و دو", "twenty two") and multiplicative
    scales ("three hundred", "دو هزار").
    """
    total = 0
    current = 0
    seen = False
    for word in words:
        if word == _PERSIAN_JOINER:
            continue
        scale = _scale_value(word)
        if scale is not None:
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            seen = True
            continue
        value = _word_value(word)
        if value is None:
            return None
        current += value
        seen = True
    return (total + current) if seen else None


def words_to_numbers(text):
    """Replace runs of spelled-out numbers with digits, so that
    "شش میلی‌متر" and "6 mm" agree once both sides are normalised."""
    tokens = text.split()
    out = []
    index = 0
    while index < len(tokens):
        if tokens[index] not in _NUMBER_WORDS:
            out.append(tokens[index])
            index += 1
            continue
        # Extend while the run still looks numeric. A trailing joiner is left
        # alone rather than swallowing the following clause.
        end = index
        while end < len(tokens) and (
            tokens[end] in _NUMBER_WORDS
            or (tokens[end] == _PERSIAN_JOINER
                and end + 1 < len(tokens) and tokens[end + 1] in _NUMBER_WORDS)
        ):
            end += 1
        run = tokens[index:end]
        value = _collapse_number_run(run)
        out.extend(run) if value is None else out.append(str(value))
        index = end
    return " ".join(out)


# --- Main entry point ------------------------------------------------------
def normalize(text, spell_out_numbers=True, strip_punctuation=True):
    """Canonical form of `text` for comparison.

    `spell_out_numbers` turns number words into digits; disable it for a
    verbatim view. `strip_punctuation` keeps decimal points inside numbers and
    drops everything else.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_DIGIT_MAP)
    for source, target in _LETTER_MAP.items():
        text = text.replace(source, target)
    text = _DIACRITICS.sub("", text)

    # ZWNJ joins: "می‌رود" -> "میرود". The spaced variant stays two tokens,
    # which is the honest reading of what was actually written.
    text = text.replace(ZWNJ, "")

    text = text.replace("٫", ".").replace("٬", "").replace("،", " ")
    text = text.replace("×", " x ").replace("−", "-")
    text = text.lower()

    if strip_punctuation:
        # Strip punctuation, then drop only the dots outside a number.
        text = _STRAY_DOT.sub(" ", _PUNCTUATION.sub(" ", text))

    text = _WHITESPACE.sub(" ", text).strip()

    if spell_out_numbers:
        text = words_to_numbers(text)
    return text


def tokenize(text, **kwargs):
    """Normalised whitespace tokens — the unit WER is measured in."""
    normalized = normalize(text, **kwargs)
    return normalized.split() if normalized else []
