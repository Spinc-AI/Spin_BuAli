"""Pulling structured facts out of normalised report text.

Everything here runs on `text_normalizer.normalize()` output, so patterns are
written in normalised form: ASCII digits, no ZWNJ, lowercase, no punctuation
except decimal points.

Four things get extracted, and each one backs a metric:
  measurements  -> number / unit error rates
  concepts      -> medical term precision / recall
  negation      -> negation error rate
  laterality    -> laterality error rate
"""
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass

from text_normalizer import normalize

# --- Units -----------------------------------------------------------------
# Canonical unit per dimension: millimetres for length, millilitres for volume.
# Keys are post-normalisation spellings (ZWNJ already removed).
LENGTH_UNITS = {
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "millimetre": 1.0,
    "میلیمتر": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0, "centimetre": 10.0,
    "سانتیمتر": 10.0,
    "m": 1000.0, "meter": 1000.0, "متر": 1000.0,
}
VOLUME_UNITS = {
    "cc": 1.0, "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "سیسی": 1.0, "میلیلیتر": 1.0,
}
_UNIT_DIMENSION = {unit: "length" for unit in LENGTH_UNITS}
_UNIT_DIMENSION.update({unit: "volume" for unit in VOLUME_UNITS})
_UNIT_FACTOR = dict(LENGTH_UNITS)
_UNIT_FACTOR.update(VOLUME_UNITS)

# Words joining the parts of a dimension group, e.g. 107 x 44 / 107 در 44.
_DIMENSION_JOINERS = {"x", "در", "by"}

_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class Measurement:
    """One numeric value with its unit, in canonical form."""
    value: float
    unit: str | None
    canonical: float
    dimension: str | None
    index: int

    def same_quantity(self, other):
        """True when both describe the same physical quantity."""
        if self.dimension != other.dimension:
            return False
        return abs(self.canonical - other.canonical) < 1e-9


def extract_measurements(text):
    """Numbers that are actually measurements.

    A bare number is ignored: in Persian the word for one doubles as the
    indefinite article, so counting every digit would invent measurements.
    A number qualifies when it carries a unit, or belongs to a dimension
    group such as 107 x 44, where a unit stated once applies to the group.
    """
    tokens = normalize(text).split()
    groups = []
    current = []

    for position, token in enumerate(tokens):
        if _NUMBER.match(token):
            current.append(position)
            continue
        if token in _DIMENSION_JOINERS and current:
            continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    measurements = []
    for group in groups:
        after = group[-1] + 1
        while after < len(tokens) and tokens[after] in _DIMENSION_JOINERS:
            after += 1
        unit = None
        if after < len(tokens) and tokens[after] in _UNIT_FACTOR:
            unit = tokens[after]

        if unit is None and len(group) < 2:
            continue

        for position in group:
            value = float(tokens[position])
            dimension = _UNIT_DIMENSION.get(unit) if unit else None
            factor = _UNIT_FACTOR.get(unit, 1.0) if unit else 1.0
            measurements.append(Measurement(
                value=value, unit=unit, canonical=value * factor,
                dimension=dimension, index=position,
            ))
    return measurements


# --- Negation --------------------------------------------------------------
# English negation precedes the finding (no stone); Persian negation usually
# follows it. Each cue therefore declares the direction it governs, and the
# window is searched that way.
NEGATION_CUES_BEFORE = {
    "no", "not", "without", "absent", "negative", "free", "denies",
    "unremarkable", "neither", "nor",
}
NEGATION_CUES_AFTER = {
    "ندارد", "نداره", "نداشت", "نیست", "نبود", "نشد", "نمیشود", "نمیشد",
    "ندارند", "nadarad", "nist", "nashod",
}
NEGATION_CUES_EITHER = {"بدون", "عدم", "فاقد", "منفی"}

# Tokens that close the scope, so "no stone but mass is seen" leaves the mass
# affirmed rather than negated.
_SCOPE_BREAKERS = {"but", "however", "اما", "ولی", "although", "though"}

NEGATION_WINDOW = 6


def negation_spans(tokens):
    """Token ranges (start, end) governed by a negation cue."""
    spans = []
    for position, token in enumerate(tokens):
        forward = token in NEGATION_CUES_BEFORE or token in NEGATION_CUES_EITHER
        backward = token in NEGATION_CUES_AFTER or token in NEGATION_CUES_EITHER
        if not (forward or backward):
            continue
        start = end = position
        if forward:
            while end + 1 < len(tokens) and end - position < NEGATION_WINDOW:
                if tokens[end + 1] in _SCOPE_BREAKERS:
                    break
                end += 1
        if backward:
            while start - 1 >= 0 and position - start < NEGATION_WINDOW:
                if tokens[start - 1] in _SCOPE_BREAKERS:
                    break
                start -= 1
        spans.append((start, end))
    return spans


def is_negated(tokens, index, spans=None):
    """Whether the token at `index` sits inside any negation scope."""
    spans = negation_spans(tokens) if spans is None else spans
    return any(start <= index <= end for start, end in spans)


# --- Laterality ------------------------------------------------------------
LATERALITY_TERMS = {
    "right": "right", "rt": "right", "راست": "right",
    "left": "left", "lt": "left", "چپ": "left",
    "bilateral": "bilateral", "both": "bilateral", "دوطرفه": "bilateral",
    "طرفین": "bilateral",
}
LATERALITY_WINDOW = 5


def laterality_at(tokens, index):
    """The laterality governing the token at `index`, if any.

    Uses the nearest marker within the window on either side, because word
    order differs between the two languages: right kidney / کلیه راست.
    """
    best = None
    best_distance = LATERALITY_WINDOW + 1
    low = max(0, index - LATERALITY_WINDOW)
    high = min(len(tokens), index + LATERALITY_WINDOW + 1)
    for position in range(low, high):
        side = LATERALITY_TERMS.get(tokens[position])
        if side is None:
            continue
        distance = abs(position - index)
        if distance < best_distance:
            best, best_distance = side, distance
    return best


# --- Clinical concepts -----------------------------------------------------
@dataclass(frozen=True)
class ConceptMention:
    concept_id: str
    category: str
    index: int
    negated: bool
    laterality: str | None


class ClinicalTerms:
    """The concept vocabulary, loaded from clinical_terms.json."""

    def __init__(self, path=None):
        path = pathlib.Path(path or pathlib.Path(__file__).parent / "clinical_terms.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.version = data.get("version", "unknown")
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        self._by_phrase = {}
        for concept in data["concepts"]:
            for variant in concept["variants"]:
                phrase = tuple(normalize(variant).split())
                if phrase:
                    self._by_phrase[phrase] = (concept["id"], concept["category"])
        self.max_phrase_len = max((len(phrase) for phrase in self._by_phrase), default=1)

    def find(self, tokens):
        """Every concept mention in `tokens`.

        Longest phrase wins, so "fatty liver" is not also counted as "liver".
        """
        spans = negation_spans(tokens)
        mentions = []
        position = 0
        while position < len(tokens):
            for length in range(min(self.max_phrase_len, len(tokens) - position), 0, -1):
                hit = self._by_phrase.get(tuple(tokens[position:position + length]))
                if hit is None:
                    continue
                concept_id, category = hit
                mentions.append(ConceptMention(
                    concept_id=concept_id, category=category, index=position,
                    negated=is_negated(tokens, position, spans),
                    laterality=laterality_at(tokens, position),
                ))
                position += length
                break
            else:
                position += 1
        return mentions
