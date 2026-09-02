"""The metrics themselves.

`evaluate()` is the whole public surface: give it a hypothesis and a reference
and it returns the report described in evaluation_schema.json.

The order matters. Nothing is counted until both texts have been normalised and
their entities aligned, because comparing raw text turns a single substitution
into one addition plus one omission -- inflating two metrics while the metric
that should have caught it reads zero.
"""
from dataclasses import dataclass

import config
import general_metrics
import semantic_metrics
from extractors import ClinicalTerms, extract_measurements
from text_normalizer import tokenize

# 1.1.0 added the denominators to clinical_counts; 1.2.0 added the character
# counts and script contamination. Additive -- no metric value moved -- but
# results carry the version so the difference is never guesswork.
METRICS_VERSION = "1.2.0"

# Negation applies to findings, not to body parts: a report says "no stone",
# never "no kidney". Scoring anatomy for negation just measures how far the
# cue window happened to reach.
NEGATABLE_CATEGORIES = {"finding", "device"}


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


# --- Comparing the two texts -----------------------------------------------
@dataclass(frozen=True)
class ConceptComparison:
    """How the two texts agreed about clinical concepts."""
    reference: dict
    hypothesis: dict
    matched: set
    only_reference: set
    only_hypothesis: set
    negation_errors: int
    negation_scored: int
    laterality_errors: int
    laterality_scored: int
    critical_errors: list


@dataclass(frozen=True)
class MeasurementComparison:
    """How the two texts agreed about measured values."""
    reference_count: int
    hypothesis_count: int
    number_errors: int
    unit_errors: int
    missing: list
    extra: list
    critical_errors: list


def _index_by_concept(mentions):
    """One entry per concept; the first mention carries its negation and side."""
    indexed = {}
    for mention in mentions:
        indexed.setdefault(mention.concept_id, mention)
    return indexed


def _compare_concepts(reference, hypothesis):
    """Negation and laterality, judged only on concepts present in both texts.

    A concept in just one text is a term-level miss, already counted by
    precision and recall; comparing its negation against nothing would mean
    nothing.
    """
    matched = set(reference) & set(hypothesis)
    negation_errors = negation_scored = 0
    laterality_errors = laterality_scored = 0
    critical_errors = []

    for concept_id in sorted(matched):
        expected, produced = reference[concept_id], hypothesis[concept_id]

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

    return ConceptComparison(
        reference=reference, hypothesis=hypothesis, matched=matched,
        only_reference=set(reference) - set(hypothesis),
        only_hypothesis=set(hypothesis) - set(reference),
        negation_errors=negation_errors, negation_scored=negation_scored,
        laterality_errors=laterality_errors, laterality_scored=laterality_scored,
        critical_errors=critical_errors,
    )


def _compare_measurements(reference_text, hypothesis_text):
    """Align the measurements, then judge each pair."""
    reference = extract_measurements(reference_text)
    hypothesis = extract_measurements(hypothesis_text)
    pairs, missing, extra = align_measurements(reference, hypothesis)

    number_errors = unit_errors = 0
    critical_errors = []
    for expected, produced in pairs:
        kinds = classify_measurement_error(expected, produced)
        number_errors += "number" in kinds
        unit_errors += "unit" in kinds
        if kinds:
            critical_errors.append({
                "type": "measurement_mismatch", "concept": "measurement",
                "reference": _format(expected), "prediction": _format(produced),
            })
    for expected in missing:
        critical_errors.append({
            "type": "measurement_omission", "concept": "measurement",
            "reference": _format(expected), "prediction": None,
        })

    return MeasurementComparison(
        reference_count=len(reference), hypothesis_count=len(hypothesis),
        number_errors=number_errors, unit_errors=unit_errors,
        missing=missing, extra=extra, critical_errors=critical_errors,
    )


def _dropped_critical_concepts(concepts):
    """Concepts the reference stated with a negation or a side that the output
    dropped entirely -- content a reader would have acted on."""
    return [
        concept_id for concept_id in sorted(concepts.only_reference)
        if concepts.reference[concept_id].negated
        or concepts.reference[concept_id].laterality is not None
    ]


def _term_scores(true_positives, false_positives, false_negatives):
    """Precision, recall and F1 over concept identities."""
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return precision, recall, _ratio(2 * precision * recall, precision + recall)


# --- The report ------------------------------------------------------------
def evaluate(hypothesis_text, reference_text, terms=None, include_semantic=False):
    """Score one report against its reference.

    `hypothesis_text` is what the pipeline produced; `reference_text` is the
    radiologist-verified version. `include_semantic` adds the embedding
    metrics, which load a model and are therefore opt-in.
    """
    terms = terms or ClinicalTerms()

    concepts = _compare_concepts(
        _index_by_concept(terms.find(tokenize(reference_text))),
        _index_by_concept(terms.find(tokenize(hypothesis_text))))
    measurements = _compare_measurements(reference_text, hypothesis_text)

    dropped = _dropped_critical_concepts(concepts)
    critical_omissions = len(measurements.missing) + len(dropped)
    unsupported_additions = len(measurements.extra) + len(concepts.only_hypothesis)

    # How much there was to get wrong. Reported alongside the errors because a
    # rate cannot be re-derived across a batch without it: summing rates is not
    # the same as a rate over summed counts, and only the latter is meaningful
    # when reports vary in length.
    omission_scored = measurements.reference_count + len(dropped)
    addition_scored = measurements.hypothesis_count + len(concepts.hypothesis)

    true_positives = len(concepts.matched)
    false_positives = len(concepts.only_hypothesis)
    false_negatives = len(concepts.only_reference)
    precision, recall, f1 = _term_scores(true_positives, false_positives, false_negatives)

    wer = general_metrics.word_error_rate(reference_text, hypothesis_text)
    cer = general_metrics.character_error_rate(reference_text, hypothesis_text)

    # A degenerate output needs a human look even when every clinical counter
    # reads zero -- a model repeating one plausible sentence forever produces
    # no negation, laterality or number error at all.
    repetition = general_metrics.repetition_score(hypothesis_text)
    length_ratio = general_metrics.hallucination_ratio(reference_text, hypothesis_text)
    degenerate = [
        name for name, tripped in (
            ("repetition", repetition >= config.MAX_REPETITION),
            ("length_ratio", length_ratio >= config.MAX_LENGTH_RATIO
                             or (reference_text.strip() and length_ratio <= config.MIN_LENGTH_RATIO)),
        ) if tripped
    ]

    reasons = degenerate + [name for name, count in (
        ("negation_errors", concepts.negation_errors),
        ("laterality_errors", concepts.laterality_errors),
        ("number_errors", measurements.number_errors),
        ("unit_errors", measurements.unit_errors),
        ("critical_omissions", critical_omissions),
        ("unsupported_additions", unsupported_additions),
    ) if count]

    report = {
        "general": {
            "wer": round(wer.rate, 4),
            "cer": round(cer.rate, 4),
            "substitutions": wer.substitutions,
            "insertions": wer.insertions,
            "deletions": wer.deletions,
            "reference_words": wer.reference_length,
            "hypothesis_words": len(tokenize(hypothesis_text)),
            # Character-level counts, so a batch can report a corpus CER --
            # total edits over total characters -- rather than a mean of
            # per-report rates, which weights a one-line report like a page.
            "character_errors": cer.errors,
            "reference_chars": cer.reference_length,
            "chrf": round(general_metrics.chrf(reference_text, hypothesis_text), 4),
            "hallucination_ratio": round(length_ratio, 4),
            "repetition_score": round(repetition, 4),
            "punctuation_f1": round(
                general_metrics.punctuation_f1(reference_text, hypothesis_text), 4),
            # Reported as a pair on purpose. This corpus code-switches English
            # radiology terms deliberately, so the hypothesis figure alone says
            # nothing; the gap between the two is the part that does.
            "script_contamination": round(
                general_metrics.script_contamination(hypothesis_text), 4),
            "reference_script_contamination": round(
                general_metrics.script_contamination(reference_text), 4),
        },
        "clinical_counts": {
            "reference_entities": len(concepts.reference),
            "matched_entities": true_positives,
            "true_positive_terms": true_positives,
            "false_positive_terms": false_positives,
            "false_negative_terms": false_negatives,
            "reference_measurements": measurements.reference_count,
            "hypothesis_measurements": measurements.hypothesis_count,
            "negation_errors": concepts.negation_errors,
            "laterality_errors": concepts.laterality_errors,
            "number_errors": measurements.number_errors,
            "unit_errors": measurements.unit_errors,
            "critical_omissions": critical_omissions,
            "unsupported_additions": unsupported_additions,
            # Denominators, so every rate above can be rebuilt over a batch.
            "negation_scored": concepts.negation_scored,
            "laterality_scored": concepts.laterality_scored,
            "critical_omission_scored": omission_scored,
            "unsupported_addition_scored": addition_scored,
        },
        "clinical_metrics": {
            "medical_term_precision": round(precision, 4),
            "medical_term_recall": round(recall, 4),
            "medical_term_f1": round(f1, 4),
            "negation_error_rate": round(
                _ratio(concepts.negation_errors, concepts.negation_scored), 4),
            "laterality_error_rate": round(
                _ratio(concepts.laterality_errors, concepts.laterality_scored), 4),
            "number_error_rate": round(
                _ratio(measurements.number_errors, measurements.reference_count), 4),
            "unit_error_rate": round(
                _ratio(measurements.unit_errors, measurements.reference_count), 4),
            "critical_omission_rate": round(_ratio(critical_omissions, omission_scored), 4),
            "unsupported_addition_rate": round(_ratio(unsupported_additions, addition_scored), 4),
        },
        "critical_errors": concepts.critical_errors + measurements.critical_errors,
        "requires_medical_review": bool(reasons),
        "review_reasons": reasons,
        "evaluation": {
            "metrics_version": METRICS_VERSION,
            "terms_version": terms.version,
            "terms_sha": terms.sha,
        },
    }

    if include_semantic:
        report["semantic"] = semantic_metrics.compute(reference_text, hypothesis_text)
    return report


def _ratio(numerator, denominator):
    """Rates are 0.0 when nothing was scorable, never a division error."""
    return numerator / denominator if denominator else 0.0


def _format(measurement):
    number = int(measurement.value) if measurement.value.is_integer() else measurement.value
    return f"{number} {measurement.unit}" if measurement.unit else str(number)
