"""The metrics themselves.

`evaluate()` is the whole public surface: give it a hypothesis and a reference
and it returns the report described in evaluation_schema.json.

The order matters. Nothing is counted until both texts have been normalised and
their entities aligned, because comparing raw text turns a single substitution
into one addition plus one omission -- inflating two metrics while the metric
that should have caught it reads zero.
"""
from dataclasses import dataclass

from extractors import ClinicalTerms, extract_measurements
from text_normalizer import normalize, tokenize

METRICS_VERSION = "1.0.0"

# Negation applies to findings, not to body parts: a report says "no stone",
# never "no kidney". Scoring anatomy for negation just measures how far the
# cue window happened to reach.
NEGATABLE_CATEGORIES = {"finding", "device"}


# --- General metrics -------------------------------------------------------
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
    reference = normalize(reference_text).replace(" ", "")
    hypothesis = normalize(hypothesis_text).replace(" ", "")
    return edit_counts(list(reference), list(hypothesis))


# --- Measurement alignment -------------------------------------------------
def align_measurements(reference, hypothesis):
    """Pair up measurements before judging them.

    Exact physical matches are paired first so an out-of-order but correct
    measurement is never mistaken for an error; whatever is left is paired in
    order of appearance within the same dimension. Leftovers on the reference
    side are omissions, leftovers on the hypothesis side are additions.
    """
    remaining_hypothesis = list(hypothesis)
    pairs = []
    unmatched_reference = []

    for measurement in reference:
        match = next((candidate for candidate in remaining_hypothesis
                      if measurement.same_quantity(candidate)), None)
        if match is not None:
            remaining_hypothesis.remove(match)
            pairs.append((measurement, match))
        else:
            unmatched_reference.append(measurement)

    still_unmatched = []
    for measurement in unmatched_reference:
        match = next((candidate for candidate in remaining_hypothesis
                      if candidate.dimension == measurement.dimension), None)
        if match is not None:
            remaining_hypothesis.remove(match)
            pairs.append((measurement, match))
        else:
            still_unmatched.append(measurement)

    return pairs, still_unmatched, remaining_hypothesis


def classify_measurement_error(reference, hypothesis):
    """Which counter a mismatched pair belongs to.

    Comparing physical quantities rather than unit strings means 10 mm and
    1 cm agree. When they genuinely differ, the literal tells us what went
    wrong: the same number with a different unit is a misheard unit, a
    different number with the same unit is a misheard value.
    """
    if reference.same_quantity(hypothesis):
        return set()
    errors = set()
    if reference.unit != hypothesis.unit:
        errors.add("unit")
    if reference.value != hypothesis.value:
        errors.add("number")
    if not errors:
        errors.add("number")
    return errors


# --- The report ------------------------------------------------------------
def evaluate(hypothesis_text, reference_text, terms=None):
    """Score one report against its reference.

    `hypothesis_text` is what the pipeline produced; `reference_text` is the
    radiologist-verified version.
    """
    terms = terms or ClinicalTerms()

    reference_tokens = tokenize(reference_text)
    hypothesis_tokens = tokenize(hypothesis_text)
    reference_mentions = terms.find(reference_tokens)
    hypothesis_mentions = terms.find(hypothesis_tokens)

    # One entry per concept: the first mention carries its negation and side.
    reference_concepts = {}
    for mention in reference_mentions:
        reference_concepts.setdefault(mention.concept_id, mention)
    hypothesis_concepts = {}
    for mention in hypothesis_mentions:
        hypothesis_concepts.setdefault(mention.concept_id, mention)

    matched = set(reference_concepts) & set(hypothesis_concepts)
    only_hypothesis = set(hypothesis_concepts) - set(reference_concepts)
    only_reference = set(reference_concepts) - set(hypothesis_concepts)

    critical_errors = []

    # Negation and laterality are only meaningful on concepts present in both.
    negation_errors = negation_scored = 0
    laterality_errors = laterality_scored = 0
    for concept_id in sorted(matched):
        expected, produced = reference_concepts[concept_id], hypothesis_concepts[concept_id]

        if expected.category in NEGATABLE_CATEGORIES:
            negation_scored += 1
            if expected.negated != produced.negated:
                negation_errors += 1
                critical_errors.append({
                    "type": "negation_flip", "concept": concept_id,
                    "reference": "negative" if expected.negated else "positive",
                    "prediction": "negative" if produced.negated else "positive",
                })

        if expected.laterality is not None:
            laterality_scored += 1
            if expected.laterality != produced.laterality:
                laterality_errors += 1
                critical_errors.append({
                    "type": "laterality_flip", "concept": concept_id,
                    "reference": expected.laterality,
                    "prediction": produced.laterality,
                })

    # Measurements
    reference_measurements = extract_measurements(reference_text)
    hypothesis_measurements = extract_measurements(hypothesis_text)
    pairs, missing_measurements, extra_measurements = align_measurements(
        reference_measurements, hypothesis_measurements)

    number_errors = unit_errors = 0
    for expected, produced in pairs:
        kinds = classify_measurement_error(expected, produced)
        if "number" in kinds:
            number_errors += 1
        if "unit" in kinds:
            unit_errors += 1
        if kinds:
            critical_errors.append({
                "type": "measurement_mismatch", "concept": "measurement",
                "reference": _format(expected), "prediction": _format(produced),
            })
    for expected in missing_measurements:
        critical_errors.append({
            "type": "measurement_omission", "concept": "measurement",
            "reference": _format(expected), "prediction": None,
        })

    # A missing concept is critical when the reference stated it with a
    # negation or a side -- that is content a reader would act on.
    critical_concept_omissions = [
        concept_id for concept_id in sorted(only_reference)
        if reference_concepts[concept_id].negated
        or reference_concepts[concept_id].laterality is not None
    ]
    critical_omissions = len(missing_measurements) + len(critical_concept_omissions)
    unsupported_additions = len(extra_measurements) + len(only_hypothesis)

    true_positives = len(matched)
    false_positives = len(only_hypothesis)
    false_negatives = len(only_reference)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)

    wer = word_error_rate(reference_text, hypothesis_text)
    cer = character_error_rate(reference_text, hypothesis_text)

    reasons = [name for name, count in (
        ("negation_errors", negation_errors),
        ("laterality_errors", laterality_errors),
        ("number_errors", number_errors),
        ("unit_errors", unit_errors),
        ("critical_omissions", critical_omissions),
        ("unsupported_additions", unsupported_additions),
    ) if count]

    return {
        "general": {
            "wer": round(wer.rate, 4),
            "cer": round(cer.rate, 4),
            "substitutions": wer.substitutions,
            "insertions": wer.insertions,
            "deletions": wer.deletions,
            "reference_words": wer.reference_length,
        },
        "clinical_counts": {
            "reference_entities": len(reference_concepts),
            "matched_entities": true_positives,
            "true_positive_terms": true_positives,
            "false_positive_terms": false_positives,
            "false_negative_terms": false_negatives,
            "reference_measurements": len(reference_measurements),
            "negation_errors": negation_errors,
            "laterality_errors": laterality_errors,
            "number_errors": number_errors,
            "unit_errors": unit_errors,
            "critical_omissions": critical_omissions,
            "unsupported_additions": unsupported_additions,
        },
        "clinical_metrics": {
            "medical_term_precision": round(precision, 4),
            "medical_term_recall": round(recall, 4),
            "medical_term_f1": round(f1, 4),
            "negation_error_rate": round(_ratio(negation_errors, negation_scored), 4),
            "laterality_error_rate": round(_ratio(laterality_errors, laterality_scored), 4),
            "number_error_rate": round(_ratio(number_errors, len(reference_measurements)), 4),
            "unit_error_rate": round(_ratio(unit_errors, len(reference_measurements)), 4),
            "critical_omission_rate": round(
                _ratio(critical_omissions,
                       len(reference_measurements) + len(critical_concept_omissions)), 4),
            "unsupported_addition_rate": round(
                _ratio(unsupported_additions,
                       len(hypothesis_measurements) + len(hypothesis_concepts)), 4),
        },
        "critical_errors": critical_errors,
        "requires_medical_review": bool(reasons),
        "review_reasons": reasons,
        "evaluation": {
            "metrics_version": METRICS_VERSION,
            "terms_version": terms.version,
            "terms_sha": terms.sha,
        },
    }


def _ratio(numerator, denominator):
    """Rates are 0.0 when nothing was scorable, never a division error."""
    return numerator / denominator if denominator else 0.0


def _format(measurement):
    number = int(measurement.value) if measurement.value.is_integer() else measurement.value
    return f"{number} {measurement.unit}" if measurement.unit else str(number)
