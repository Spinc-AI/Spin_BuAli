"""Evaluation service -- scores a transcript against a verified reference.

Deliberately standalone and stateless: it takes two texts and returns numbers,
so it can be called at any point after transcription, re-run over old reports
when the metrics change, and deployed without touching the transcription path.

Run:
    python main.py            # or: uvicorn main:app --host 0.0.0.0 --port 8002
Interactive docs at http://<host>:8002/docs
"""
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

import config
from extractors import ClinicalTerms
from medical_metrics import METRICS_VERSION, evaluate
from schemas import EvaluationRequest, EvaluationResponse

app = FastAPI(title="Spin BuAli Evaluation Service")

# The vocabulary is read once at import; a term-list change is a deploy, which
# is what keeps `terms_sha` in the results meaningful.
TERMS = ClinicalTerms(config.CLINICAL_TERMS_PATH)


@app.get("/")
def health() -> dict:
    return {
        "evaluation_service": "ok",
        "metrics_version": METRICS_VERSION,
        "terms_version": TERMS.version,
        "terms_sha": TERMS.sha,
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_report(request: EvaluationRequest) -> dict:
    """Score one transcript against its radiologist-verified reference.

    Scoring is CPU-bound and synchronous, so it runs in a worker thread rather
    than blocking the event loop for every other request.
    """
    result = await run_in_threadpool(
        evaluate, request.hypothesis, request.reference, TERMS)
    return {
        "asset_id": request.asset_id,
        "model": request.model,
        "pipeline": request.pipeline,
        "model_version": request.model_version,
        **result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
