"""Local model loading and generation, served directly via ``transformers``.

One manager holds at most one model at a time, and that model serves both
``/chat`` and ``/chat_audio`` -- load it once and either endpoint can use it
without a reload, as long as the same registry key is requested.

Models come in three shapes, each with its own transformers classes and
chat-template conventions. Add a model by writing a ``BaseLLM`` subclass (or
reusing one) and adding a ``MODEL_REGISTRY`` entry; nothing in main.py changes.

  TextOnlyModel        Aya Expanse 8B/32B, Gemma 4 31B. No audio input.
  GemmaAudioModel      Gemma 4 E4B/12B ("Unified", encoder-free). Text and audio.
  QwenOmniModel        Qwen3-Omni-30B, Thinker-only -- text out, no speech
                       generation, which also skips the Talker's codec weights.
  MedGemmaTextModel    MedGemma 1.5 4B, used text-only here. The checkpoint is
                       image+text (AutoModelForImageTextToText, not a plain
                       causal LM), but nothing in this codebase sends it an
                       image, so no image content is ever attached.
  Phi4MultimodalModel  Phi-4-multimodal-instruct. Text and audio, via
                       Microsoft's own custom modeling code
                       (trust_remote_code=True) rather than a standard
                       transformers architecture.

Two deliberate trade-offs:
  - No Ollama. It cannot accept audio input at all, so keeping it would mean
    two serving paths side by side when several of these models do both roles.
  - No quantization, so models load at full bf16/fp16 precision and need more
    VRAM than a quantized equivalent. Revisit with bitsandbytes if that bites.
"""
import gc
import tempfile
import threading
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

import config


def _generation_kwargs(temperature: float) -> dict:
    """Near-zero temperature means greedy decoding; anything higher samples.

    Medical use wants consistency over creativity, hence the low default."""
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
             temperature: float = 0.3) -> str:
        """Return the model's text reply.

        `messages` is the standard OpenAI shape: [{"role": ..., "content": <str>}, ...].
        `audio_path` (only meaningful if `supports_audio`) attaches an audio
        file to the last user turn. `temperature` <= 0.01 means greedy
        decoding (see _generation_kwargs).

        There is no JSON mode: callers that need JSON ask for it in the
        prompt and parse the reply tolerantly.
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

    def chat(self, messages, audio_path=None, temperature=0.3):
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

    def chat(self, messages, audio_path=None, temperature=0.3):
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

    def chat(self, messages, audio_path=None, temperature=0.3):
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


class MedGemmaTextModel(BaseLLM):
    """MedGemma 1.5 4B, used text-only.

    The checkpoint is image+text (loads via AutoModelForImageTextToText, not
    AutoModelForCausalLM -- attempting the latter fails at load time), but
    nothing here ever attaches an image, so the chat template only ever sees
    a text content part.
    """

    supports_audio = False

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, device_map=config.DEVICE_MAP, dtype="auto"
        )

    def chat(self, messages, audio_path=None, temperature=0.3):
        if audio_path:
            raise ValueError(f"{self.model_id} is text-only and can't accept audio input")
        converted = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in messages]
        inputs = self._processor.apply_chat_template(
            converted, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                           **_generation_kwargs(temperature))
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)


class Phi4MultimodalModel(BaseLLM):
    """Phi-4-multimodal-instruct. Text and audio, via Microsoft's own custom
    modeling code (trust_remote_code=True) rather than a standard
    transformers architecture class -- unlike every other model in this
    file, `AutoModelForCausalLM` here resolves to that custom code, not the
    plain-causal-LM path TextOnlyModel uses.

    The prompt format is the vendor's own: an inline `<|audio_1|>` placeholder
    in the text where the audio should be attended to, with the actual audio
    array passed alongside via the `audios` kwarg -- not a chat-template
    content list like GemmaAudioModel/QwenOmniModel use.
    """

    supports_audio = True

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=config.DEVICE_MAP, dtype="auto",
            trust_remote_code=True,
        )
        self._generation_config = GenerationConfig.from_pretrained(self.model_id)

    def chat(self, messages, audio_path=None, temperature=0.3):
        import soundfile as sf

        last_user = _last_user_index(messages) if audio_path else -1
        prompt_parts = []
        for i, m in enumerate(messages):
            text = m["content"]
            if audio_path and i == last_user:
                text = f"<|audio_1|>{text}"
            prompt_parts.append(f"<|{m['role']}|>{text}<|end|>")
        prompt = "".join(prompt_parts) + "<|assistant|>"

        audios = [sf.read(audio_path)] if audio_path else None
        inputs = self._processor(text=prompt, audios=audios, return_tensors="pt").to(
            self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                generation_config=self._generation_config,
                **_generation_kwargs(temperature))
        return self._processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]


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
    "medgemma-1.5-4b": (MedGemmaTextModel, config.MEDGEMMA_4B_MODEL_ID),
    "phi-4-multimodal": (Phi4MultimodalModel, config.PHI4_MULTIMODAL_MODEL_ID),
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
             audio_format: str | None = None, temperature: float = 0.3) -> str:
        with self._lock:
            self._ensure_loaded(key)
            if audio is None:
                return self._current_model.chat(messages, temperature=temperature)
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}") as f:
                f.write(audio)
                f.flush()
                return self._current_model.chat(messages, audio_path=f.name,
                                                temperature=temperature)

    def unload(self):
        with self._lock:
            if self._current_model is not None:
                self._current_model.unload()
                self._current_model = None
                self._current_key = None


MANAGER = LLMManager()
