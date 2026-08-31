"""Request and response shapes for the evaluation API."""
from pydantic import BaseModel, ConfigDict, Field


class EvaluationRequest(BaseModel):
    """One report to score.

    The two texts are the whole computation; the identifying fields exist so a
    stored result can be traced back to the recording and the model that
    produced it, and so results can be grouped per file and per model.
    """
    model_config = ConfigDict(protected_namespaces=())

    asset_id: str = Field(description="Stable id of the recording, e.g. the audio filename stem")
    model: str = Field(description="Model that produced the transcript, e.g. whisper-large-v3")
    hypothesis: str = Field(description="The pipeline's final_text")
    reference: str = Field(description="The radiologist-verified text")
    pipeline: str | None = Field(default=None, description="separate | multimodal | hybrid")
    model_version: str | None = Field(default=None, description="Version/checkpoint of the model")


class GeneralMetrics(BaseModel):
    wer: float
    cer: float
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int


class ClinicalCounts(BaseModel):
    reference_entities: int
    matched_entities: int
    true_positive_terms: int
    false_positive_terms: int
    false_negative_terms: int
    reference_measurements: int
    negation_errors: int
    laterality_errors: int
    number_errors: int
    unit_errors: int
    critical_omissions: int
    unsupported_additions: int


class ClinicalMetrics(BaseModel):
    medical_term_precision: float
    medical_term_recall: float
    medical_term_f1: float
    negation_error_rate: float
    laterality_error_rate: float
    number_error_rate: float
    unit_error_rate: float
    critical_omission_rate: float
    unsupported_addition_rate: float


class CriticalError(BaseModel):
    """One clinically significant difference, kept for the audit trail."""
    type: str  # negation_flip | laterality_flip | measurement_mismatch | measurement_omission
    concept: str
    reference: str | None
    prediction: str | None


class EvaluationVersions(BaseModel):
    """Which ruler measured this result.

    Metric values only compare across reports scored with the same versions --
    adding terms or fixing a comparator moves the numbers on its own.
    """
    metrics_version: str
    terms_version: str
    terms_sha: str


class EvaluationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    asset_id: str
    model: str
    pipeline: str | None = None
    model_version: str | None = None
    general: GeneralMetrics
    clinical_counts: ClinicalCounts
    clinical_metrics: ClinicalMetrics
    critical_errors: list[CriticalError]
    requires_medical_review: bool
    review_reasons: list[str]
    evaluation: EvaluationVersions
