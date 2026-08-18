"""Core_LLM's unified local model layer -- served directly via ``transformers``,
NOT Ollama.

Ollama was dropped entirely: it can't accept audio input at all (as of
2026-07: https://github.com/ollama/ollama/issues/11798), which meant running
two separate serving paths side by side (Ollama for text, transformers for
audio) even though several of these models can do both. Now there's ONE
manager holding at most one loaded model at a time, and that model serves
BOTH `/chat` (text-only) and `/chat_audio` (audio-capable models only) in
main.py -- load it once, use it for either role without a reload, as long as
the same registry key is requested.

Trade-off accepted deliberately: no quantization here (Ollama auto-quantized
for you; plain `transformers` loads at full bf16/fp16 precision), so VRAM
needs are higher per model than the old Ollama-served numbers. Fine on a
big card; revisit with bitsandbytes if VRAM becomes a real constraint.

Three model "shapes", each with their own transformers classes and
chat-template content-key conventions -- mirrors the STT module's swappable
BaseSTTModel/ModelManager pattern:
  - TextOnlyModel   -- Aya Expanse (8B/32B), Gemma 4 31B. No audio input --
                       26B-A4B/31B are Gemma 4's image/video/text-only tier.
  - GemmaAudioModel -- Gemma 4 E4B/12B ("Unified", encoder-free). Text AND
                       audio.
  - QwenOmniModel   -- Qwen3-Omni-30B, Thinker-only (text output, no speech
                       generation -- we don't need audio-out). Text AND
                       audio; confirmed via PARSA-Bench (an independent
                       Persian audio-LM benchmark) as the strongest tested
                       locally-runnable option for Persian audio.
"""
import gc
import tempfile
import threading
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

import config


def _generation_kwargs(temperature: float) -> dict:
    """Low temperature (near 0) -> greedy decoding; otherwise sampled.
    Mirrors the old Ollama-backed /chat's low-temperature-by-default
    behavior (medical use wants consistency, not creativity)."""
    if temperature <= 0.01:
        return {"do_sample": False}
    return {"do_sample": True, "temperature": temperature}


class BaseLLM(ABC):
    """A local model that can be loaded and chatted with, text-only or
    (if `supports_audio`) with an audio file attached to the last user turn."""

    supports_audio = False

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None  # tokenizer or AutoProcessor, depending on subclass

    @abstractmethod
    def load(self):
        """Pull weights and tokenizer/processor into memory."""

    @abstractmethod
    def chat(self, messages: list[dict], audio_path: str | None = None,
             temperature: float = 0.3, response_format: dict | None = None) -> str:
        """Return the model's text reply.

        `messages` is the standard OpenAI shape: [{"role": ..., "content": <str>}, ...].
        `audio_path` (only meaningful if `supports_audio`) attaches an audio
        file to the last user turn. `temperature` <= 0.01 means greedy
        decoding (see _generation_kwargs). `response_format` is accepted for
        interface parity with the old Ollama-backed /chat, but not enforced
        here -- there's no local equivalent of Ollama/OpenAI's JSON mode;
        rely on the prompt asking for JSON and the caller's tolerant parsing.
        """

    def unload(self):
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TextOnlyModel(BaseLLM):
    """Aya Expanse (8B/32B) and Gemma 4 31B -- plain causal LM, no audio."""

    supports_audio = False

    def load(self):
        self._processor = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=config.DEVICE_MAP, dtype="auto"
        )

    def chat(self, messages, audio_path=None, temperature=0.3, response_format=None):
        if audio_path:
            raise ValueError(f"{self.model_id} is text-only and can't accept audio input")
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                           **_generation_kwargs(temperature))
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    raise ValueError("messages must include at least one user turn")


class GemmaAudioModel(BaseLLM):
    """Gemma 4's "Unified" (encoder-free) models, via AutoModelForMultimodalLM
    -- covers E4B and 12B (the largest audio-capable Gemma 4 variant)."""

    supports_audio = True

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id, device_map=config.DEVICE_MAP, attn_implementation="sdpa"
        )

    def chat(self, messages, audio_path=None, temperature=0.3, response_format=None):
        last_user = _last_user_index(messages) if audio_path else -1
        converted = []
        for i, m in enumerate(messages):
            if m["role"] == "system":
                converted.append({"role": "system", "content": m["content"]})
                continue
            content = [{"type": "text", "text": m["content"]}]
            if audio_path and i == last_user:
                content.append({"type": "audio", "url": audio_path})
            converted.append({"role": m["role"], "content": content})

        inputs = self._processor.apply_chat_template(
            converted, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                           **_generation_kwargs(temperature))
        # skip_special_tokens=True (unlike the reference docs example) -- callers
        # often parse this as JSON, so stray special-token text would break that.
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)


class QwenOmniModel(BaseLLM):
    """Qwen3-Omni, Thinker-only (text output, no speech generation -- we don't
    need audio-out, and this skips loading the Talker's audio-codec weights).

    Note: transformers' own docs flag that MoE inference through `transformers`
    (as opposed to vLLM) can be slow. Fine for now since it matches Core_LLM's
    existing serving pattern; revisit if latency becomes a real problem.
    """

    supports_audio = True

    def load(self):
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
        self._model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.model_id, device_map=config.DEVICE_MAP
        )

    def chat(self, messages, audio_path=None, temperature=0.3, response_format=None):
        last_user = _last_user_index(messages) if audio_path else -1
        converted = []
        for i, m in enumerate(messages):
            content = [{"type": "text", "text": m["content"]}]
            if audio_path and i == last_user:
                # audio part first, matching the model's own reference examples
                content = [{"type": "audio", "path": audio_path}] + content
            converted.append({"role": m["role"], "content": content})

        # load_audio_from_video (from the reference docs example) deliberately
        # dropped -- we never pass video, only audio, and this kwarg was the
        # likely cause of a "coroutine raised StopIteration" failure on this
        # transformers version (its processor.__call__ kwarg-passing
        # convention changed; this parameter isn't needed for our use case
        # anyway, so removing it sidesteps the incompatibility entirely).
        inputs = self._processor.apply_chat_template(
            converted, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", padding=True,
        ).to(self._model.device, dtype=self._model.dtype)
        # Slicing off input_len (unlike the reference docs snippet, which decodes
        # the full sequence) so the reply doesn't echo the prompt back -- same
        # reasoning as GemmaAudioModel above.
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            text_ids = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                            **_generation_kwargs(temperature))
        return self._processor.batch_decode(
            text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ============================================================
# Registry
# ============================================================
MODEL_REGISTRY = {
    "aya-expanse-8b": (TextOnlyModel, config.AYA_8B_MODEL_ID),
    "aya-expanse-32b": (TextOnlyModel, config.AYA_32B_MODEL_ID),
    "gemma-4-31b": (TextOnlyModel, config.GEMMA_31B_MODEL_ID),
    "gemma-4-e4b": (GemmaAudioModel, config.GEMMA_E4B_MODEL_ID),
    "gemma-4-12b": (GemmaAudioModel, config.GEMMA_12B_MODEL_ID),
    "qwen3-omni-30b": (QwenOmniModel, config.QWEN_OMNI_MODEL_ID),
}


class LLMManager:
    """Holds at most one loaded model (of any registry kind), swapping as
    needed -- shared by BOTH /chat and /chat_audio in main.py, so loading a
    model via one endpoint means it's already warm for the other, as long as
    the same registry key is requested.

    A single lock guards both loading and generation so concurrent requests
    can't swap the model out from under an in-flight generation.
    """

    def __init__(self):
        self._current_key = None
        self._current_model = None
        self._lock = threading.Lock()

    def available(self, audio_only: bool = False) -> list[str]:
        if audio_only:
            return [k for k, (cls, _) in MODEL_REGISTRY.items() if cls.supports_audio]
        return list(MODEL_REGISTRY.keys())

    @property
    def loaded(self):
        return self._current_key

    def _ensure_loaded(self, key: str):
        if key not in MODEL_REGISTRY:
            raise KeyError(f"unknown model '{key}' -- available: {self.available()}")
        if self._current_key != key:
            if self._current_model is not None:
                self._current_model.unload()
            cls, model_id = MODEL_REGISTRY[key]
            model = cls(model_id)
            model.load()
            self._current_model = model
            self._current_key = key

    def chat(self, key: str, messages: list[dict], audio: bytes | None = None,
             audio_format: str | None = None, temperature: float = 0.3,
             response_format: dict | None = None) -> str:
        with self._lock:
            self._ensure_loaded(key)
            if audio is None:
                return self._current_model.chat(messages, temperature=temperature,
                                                 response_format=response_format)
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}") as f:
                f.write(audio)
                f.flush()
                return self._current_model.chat(messages, audio_path=f.name, temperature=temperature,
                                                response_format=response_format)

    def unload(self):
        with self._lock:
            if self._current_model is not None:
                self._current_model.unload()
                self._current_model = None
                self._current_key = None


MANAGER = LLMManager()
