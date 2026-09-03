"""Loading a language model at the precision its tier calls for, and asking it.

The STT half of this benchmark loads models in-process because HTTP per window
would dominate the measurement. The LLM half does the same, for the same
reason, plus one more: `core_llm/` has no quantization, so tier B could not run
through it at all. What is here is the loading policy the tier system decided,
made real.

Cloud models take a different road entirely -- `gemini:` and `openai:` are HTTP
calls with no weights to place -- so they route through the controller's own
client and never touch this file's loader.
"""
import time

import bridge
import settings

# bitsandbytes' two options. int8 keeps more of the weight and is slower to
# generate with; nf4 is the 4-bit normal-float, which holds a normally
# distributed weight better than a plain int4 would.
QUANTIZATION = {"int8": {"load_in_8bit": True}, "nf4": {"load_in_4bit": True}}


class LoadFailed(RuntimeError):
    """The model could not be placed on this hardware.

    Raised rather than swallowed: a tier-A model that will not load is a
    finding about the plan, not a run to quietly drop.
    """


def _quantization_config(precision: str):
    from transformers import BitsAndBytesConfig

    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Double quantization saves a further ~0.4 bits per weight, which is
        # the difference between fitting and not for the 30B tier.
        bnb_4bit_use_double_quant=True,
        # The compute dtype has to be fp16: Turing has no bf16, and asking for
        # it there is slow at best.
        bnb_4bit_compute_dtype=_torch().float16,
    )


def _torch():
    torch = bridge.torch_or_none()
    if torch is None:
        raise LoadFailed("torch is not installed")
    return torch


class LocalLLM:
    """One local model, loaded at a given precision across a given card count.

    Deliberately mirrors the STT model classes: `load()`, `generate()`,
    `unload()`. The runner does not care which of the two it is holding.
    """

    def __init__(self, model_key: str, model_id: str, precision: str = "fp16",
                 cards: int = 1, max_new_tokens: int | None = None):
        self.model_key = model_key
        self.model_id = model_id
        self.precision = precision
        self.cards = cards
        self.max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        self._model = None
        self._tokenizer = None
        self.load_seconds = 0.0

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = _torch()
        started = time.perf_counter()
        options = {
            # "auto" shards across every visible card when one is not enough;
            # a single card is pinned so the other stays free.
            "device_map": "auto" if self.cards > 1 else 0,
        }
        if self.precision in QUANTIZATION:
            options["quantization_config"] = _quantization_config(self.precision)
        else:
            options["dtype"] = torch.float16

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **options).eval()
        except Exception as error:
            raise LoadFailed(f"{self.model_key} at {self.precision}: {error}") from error
        self.load_seconds = time.perf_counter() - started
        return self

    def generate(self, system_prompt: str, user_text: str) -> str:
        """One completion. Only the newly generated tokens are decoded --
        several of these models echo the prompt otherwise, and a report that
        begins with its own instructions is not a report."""
        torch = _torch()
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text or ""}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        fresh = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(fresh, skip_special_tokens=True)

    def unload(self):
        self._model = None
        self._tokenizer = None
        torch = bridge.torch_or_none()
        if torch is not None and torch.cuda.is_available():
            import gc

            gc.collect()
            torch.cuda.empty_cache()


class CloudLLM:
    """A `gemini:` or `openai:` model, called through the controller's client.

    No weights, no placement, no tier. The controller already knows how to
    reach both providers and which credential each one needs, so this is a
    thin adapter rather than a second implementation.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_new_tokens: int | None = None):
        self.model_key = model
        self.model_id = model
        self.precision = "cloud"
        self.cards = 0
        self.load_seconds = 0.0
        self._api_key = api_key
        self._base_url = base_url

    def load(self):
        return self  # nothing to load

    def generate(self, system_prompt: str, user_text: str) -> str:
        client = _controller_llm_client()
        return client.complete(system_prompt, user_text, model=self.model_key,
                               api_key=self._api_key, base_url=self._base_url)

    def unload(self):
        pass


def _controller_llm_client():
    """The controller's provider client, imported on demand.

    Late because it pulls in the controller's own `config`, and `evaluation/`
    already owns that module name on the path -- so it is only reachable once
    the controller's directory has been put first, which happens here and
    nowhere else.
    """
    import sys

    path = str(settings.CONTROLLER_DIR)
    original = list(sys.path)
    sys.path.insert(0, path)
    for name in ("config", "providers", "llm_client"):
        sys.modules.pop(name, None)
    try:
        import llm_client

        return llm_client
    finally:
        sys.path[:] = original
        for name in ("config", "providers"):
            sys.modules.pop(name, None)


def build(model_key: str, precision: str = "fp16", cards: int = 1,
          model_id: str | None = None, **kwargs):
    """The right kind of model for `model_key`, not yet loaded."""
    if precision == "cloud" or ":" in model_key:
        return CloudLLM(model_key, **kwargs)
    return LocalLLM(model_key, model_id or _hugging_face_id(model_key),
                    precision=precision, cards=cards, **kwargs)


def _hugging_face_id(model_key: str) -> str:
    """The checkpoint behind a registry key, read from `core_llm/config.py`.

    Read rather than listed here, so adding a model there is enough to make it
    benchmarkable -- the same rule the STT side follows.
    """
    import os
    import re

    source = (settings.REPO_ROOT / "core_llm" / "config.py").read_text(encoding="utf-8")
    wanted = model_key.replace("-", "_").upper().replace("EXPANSE_", "")
    for variable, default in re.findall(r'^(\w+_MODEL_ID) = os\.getenv\(\s*"[^"]+",\s*"([^"]+)"',
                                        source, re.M):
        stem = variable[:-len("_MODEL_ID")]
        if stem.replace("_", "") in wanted.replace("_", ""):
            return os.getenv(variable, default)
    raise LoadFailed(f"no checkpoint known for {model_key!r} in core_llm/config.py")


class EchoLLM:
    """A stand-in that loads nothing and answers with the transcript it was given.

    The counterpart of `transcribe.EchoModel`. Together they let the whole
    campaign -- planning, tiering, the STT cache, scoring, the CSVs, resume --
    be exercised in seconds, so the only thing a real run can still get wrong
    is the models themselves.
    """

    model_key = "dry-run"
    model_id = "dry-run/llm"
    precision = "dry-run"
    cards = 0
    load_seconds = 0.0

    def load(self):
        return self

    def unload(self):
        pass

    def generate(self, system_prompt: str, user_text: str) -> str:
        import json

        text = (user_text or "").strip()
        return json.dumps({"raw_transcript": text, "corrected_transcript": text,
                           "final_text": text, "discrepancies_found": [], "notes": None})


def dry_run_factory(run: dict):
    return EchoLLM()
