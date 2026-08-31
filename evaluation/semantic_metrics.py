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
    usable, reason = available()
    if not usable:
        raise RuntimeError(reason)

    return {
        "bertscore_f1": round(_bertscore(reference_text, hypothesis_text), 4),
        "semantic_similarity": round(_similarity(reference_text, hypothesis_text), 4),
    }


def _bertscore(reference_text, hypothesis_text):
    from bert_score import score

    _, _, f1 = score([hypothesis_text], [reference_text],
                     model_type=config.BERTSCORE_MODEL,
                     num_layers=config.BERTSCORE_LAYERS,
                     verbose=False)
    return float(f1[0])


def _similarity(reference_text, hypothesis_text):
    from sentence_transformers import SentenceTransformer, util

    # Loading is the expensive part, so the model is kept for the process.
    if "similarity" not in _models:
        _models["similarity"] = SentenceTransformer(config.SIMILARITY_MODEL)
    model = _models["similarity"]

    embeddings = model.encode([reference_text, hypothesis_text], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
