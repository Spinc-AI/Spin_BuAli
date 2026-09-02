"""Text-level metrics: how much the two texts differ, regardless of meaning.

These say nothing clinical. They are here because they catch failure shapes the
concept-based metrics cannot see -- most importantly a model that collapses into
repeating itself, or one that returns far more text than was ever dictated.
"""
from collections import Counter
from dataclasses import dataclass

from text_normalizer import normalize, tokenize

# A hypothesis this much longer than its reference is padded, not transcribed.
REPETITION_ORDER = 4
CHRF_MAX_ORDER = 6
CHRF_BETA = 2.0  # recall weighted over precision, the sacreBLEU default

# Punctuation compared by Punctuation F1, in both scripts.
_PUNCTUATION = set(".,;:?!()-\"'«»،؛؟")

# Every Unicode block Persian is written in: Arabic, Arabic Supplement, Arabic
# Extended-A and -B, and the two presentation-form blocks.
_ARABIC_RANGES = [
    ("؀", "ۿ"), ("ݐ", "ݿ"), ("ࡰ", "࢟"),
    ("ࢠ", "ࣿ"), ("ﭐ", "﷿"), ("ﹰ", "﻿"),
]


# --- Edit distance ---------------------------------------------------------
@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int
    reference_length: int

    @property
    def errors(self):
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self):
        return self.errors / self.reference_length if self.reference_length else 0.0


def edit_counts(reference, hypothesis):
    """Levenshtein alignment, keeping the operation breakdown.

    Insertions are tracked separately because in a medical transcript they are
    the fabrication signal, not just noise.
    """
    rows, columns = len(reference), len(hypothesis)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distance[row][0] = row
    for column in range(columns + 1):
        distance[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if reference[row - 1] == hypothesis[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
            else:
                distance[row][column] = 1 + min(
                    distance[row - 1][column - 1],  # substitution
                    distance[row][column - 1],      # insertion
                    distance[row - 1][column],      # deletion
                )

    substitutions = insertions = deletions = 0
    row, column = rows, columns
    while row > 0 or column > 0:
        if row > 0 and column > 0 and reference[row - 1] == hypothesis[column - 1] \
                and distance[row][column] == distance[row - 1][column - 1]:
            row, column = row - 1, column - 1
        elif row > 0 and column > 0 and distance[row][column] == distance[row - 1][column - 1] + 1:
            substitutions += 1
            row, column = row - 1, column - 1
        elif column > 0 and distance[row][column] == distance[row][column - 1] + 1:
            insertions += 1
            column -= 1
        else:
            deletions += 1
            row -= 1
    return EditCounts(substitutions, insertions, deletions, rows)


def word_error_rate(reference_text, hypothesis_text):
    return edit_counts(tokenize(reference_text), tokenize(hypothesis_text))


def character_error_rate(reference_text, hypothesis_text):
    return edit_counts(_characters(reference_text), _characters(hypothesis_text))


def _characters(text):
    return list(normalize(text).replace(" ", ""))


# --- chrF ------------------------------------------------------------------
def chrf(reference_text, hypothesis_text, max_order=CHRF_MAX_ORDER, beta=CHRF_BETA):
    """Character n-gram F-score.

    Gives partial credit where WER does not: a morphological variant shares
    most of its characters, which matters for Persian's attached suffixes.
    """
    reference = _characters(reference_text)
    hypothesis = _characters(hypothesis_text)
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    precisions, recalls = [], []
    for order in range(1, max_order + 1):
        reference_grams = _ngrams(reference, order)
        hypothesis_grams = _ngrams(hypothesis, order)
        overlap = sum((reference_grams & hypothesis_grams).values())
        precisions.append(_ratio(overlap, sum(hypothesis_grams.values())))
        recalls.append(_ratio(overlap, sum(reference_grams.values())))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def _ngrams(items, order):
    return Counter(tuple(items[i:i + order]) for i in range(len(items) - order + 1))


# --- Failure-shape signals -------------------------------------------------
def hallucination_ratio(reference_text, hypothesis_text):
    """Output length over reference length. Above 1 means extra text appeared.

    A blunt instrument, but it sees fabrication built from words that are not
    in the clinical vocabulary -- which the concept-based metrics cannot.
    """
    reference = len(tokenize(reference_text))
    hypothesis = len(tokenize(hypothesis_text))
    return _ratio(hypothesis, reference)


def repetition_score(hypothesis_text, order=REPETITION_ORDER):
    """Share of repeated n-grams in the output, from 0 (none) to near 1.

    This is the loop detector. A model that degenerates into emitting the same
    sentence over and over scores near 1 here while every clinical metric stays
    quiet, because the repeated content is often individually plausible.
    """
    tokens = tokenize(hypothesis_text)
    if len(tokens) < order:
        return 0.0
    grams = [tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1)]
    return 1.0 - _ratio(len(set(grams)), len(grams))


def punctuation_f1(reference_text, hypothesis_text):
    """Agreement on punctuation, compared as a multiset.

    Computed on the raw text, since normalisation deliberately strips
    punctuation before any other metric runs. Models that punctuate (Whisper)
    and models that emit bare text (Wav2Vec2, MMS) are otherwise not
    comparable on it at all.
    """
    reference = _punctuation(reference_text)
    hypothesis = _punctuation(hypothesis_text)
    if not reference and not hypothesis:
        return 1.0  # neither punctuates: perfect agreement, not a failure
    matched = sum((reference & hypothesis).values())
    precision = _ratio(matched, sum(hypothesis.values()))
    recall = _ratio(matched, sum(reference.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _punctuation(text):
    return Counter(character for character in (text or "") if character in _PUNCTUATION)


def script_contamination(text):
    """Share of letters that are not Arabic-script.

    Computed on raw text, like punctuation F1, since normalisation does not
    remove Latin characters anyway.

    **Read this one against the reference, never on its own.** It was designed
    for corpora where any Latin character is leakage, and this one is not:
    these radiologists dictate English terms on purpose, so a perfectly correct
    transcript scores as heavily contaminated. What means something is the
    hypothesis figure compared with the reference figure -- a model drifting
    into Latin shows up as a gap between the two, not as a high number.
    """
    letters = [character for character in (text or "") if character.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for character in letters if not _is_arabic_script(character))
    return foreign / len(letters)


def _is_arabic_script(character):
    return any(low <= character <= high for low, high in _ARABIC_RANGES)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0
