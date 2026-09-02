"""Embedding-based metrics -- optional, and off unless asked for.

These are the only metrics here that load a model. Everything else in this
service is pure text processing that runs in milliseconds on a CPU; BERTScore
and sentence similarity pull in torch and download weights, so they are opt-in
per request rather than part of the default score.

Install them with:
    pip install -r requirements-semantic.txt

`available()` reports whether that has been done, so a caller can find out
before asking rather than by failing.
"""
import config

_models = {}


def available():
    """(usable, reason) -- reason is empty when usable."""
    try:
        import bert_score  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        return False, (f"{exc}. Install with: pip install -r requirements-semantic.txt")
    return True, ""


def compute(reference_text, hypothesis_text):
    """Meaning-level agreement between the two texts.

    bertscore_f1        token-level match through contextual embeddings, so a
                        valid synonym is credited where WER penalises it.
    semantic_similarity whole-text cosine similarity: did the message survive.
    """
    return compute_batch([(reference_text, hypothesis_text)])[0]


def compute_batch(pairs):
    """The same two scores for many (reference, hypothesis) pairs at once.

    Both models are loaded once and every pair goes through in a single
    forward pass. Scoring a benchmark report-by-report instead means reloading
    weights and running a batch of one, several hundred times over -- which is
    the difference between a minute and an afternoon.
    """
    usable, reason = available()
    if not usable:
        raise RuntimeError(reason)
    if not pairs:
        return []

    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    f1_scores = _bertscore(references, hypotheses)
    similarities = _similarity(references, hypotheses)

    return [{"bertscore_f1": round(f1, 4), "semantic_similarity": round(similarity, 4)}
            for f1, similarity in zip(f1_scores, similarities)]


def _scorer():
    """One BERTScorer for the process.

    `bert_score.score()` builds a fresh scorer -- and reloads the weights --
    on every call, so calling it per report is the single slowest thing this
    service can do. The class form keeps the model.
    """
    if "bertscore" not in _models:
        from bert_score import BERTScorer

        _models["bertscore"] = BERTScorer(
            model_type=config.BERTSCORE_MODEL,
            num_layers=config.BERTSCORE_LAYERS,
            idf=False)
    return _models["bertscore"]


def _bertscore(references, hypotheses):
    _, _, f1 = _scorer().score(hypotheses, references, verbose=False)
    return [float(value) for value in f1]


def _similarity(references, hypotheses):
    from sentence_transformers import SentenceTransformer, util

    # Loading is the expensive part, so the model is kept for the process.
    if "similarity" not in _models:
        _models["similarity"] = SentenceTransformer(config.SIMILARITY_MODEL)
    model = _models["similarity"]

    reference_embeddings = model.encode(references, convert_to_tensor=True)
    hypothesis_embeddings = model.encode(hypotheses, convert_to_tensor=True)
    return [float(util.cos_sim(reference, hypothesis)[0][0])
            for reference, hypothesis in zip(reference_embeddings, hypothesis_embeddings)]
