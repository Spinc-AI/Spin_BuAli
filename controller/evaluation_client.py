"""HTTP client for the evaluation service.

The controller is the only door into this system, so callers reach scoring
through it rather than talking to `evaluation/` directly. Nothing is
interpreted on the way past: the request body is forwarded as given and the
reply is returned as received, which keeps the metric contract owned by the
one module that implements it.
"""
import httpx

import config


def health() -> bool:
    try:
        with httpx.Client(timeout=config.HTTP_TIMEOUT) as client:
            return client.get(f"{config.EVALUATION_URL}/").status_code == 200
    except httpx.HTTPError:
        return False


def evaluate(payload: dict) -> tuple[int, dict]:
    """Forward one scoring request.

    Returns the service's own (status_code, body) so a rejected request comes
    back as the validation error it was, rather than a generic gateway
    failure. Transport problems still raise.
    """
    with httpx.Client(timeout=config.HTTP_TIMEOUT) as client:
        response = client.post(f"{config.EVALUATION_URL}/evaluate", json=payload)
    try:
        return response.status_code, response.json()
    except ValueError:
        raise RuntimeError(
            f"evaluation service returned a non-JSON reply "
            f"({response.status_code}): {response.text[:200]}"
        )
