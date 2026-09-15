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
  VoxtralModel         Mistral Voxtral Mini 3B. Text and audio; the lightest
                       audio-in model here, and tokenized through
                       mistral-common rather than a Jinja template.
  Qwen2AudioModel      Qwen2-Audio-7B-Instruct. Text and audio; the one class
                       whose processor does not load the audio file itself.

Precision is set by ``config.QUANTIZATION`` -- None for native fp16/bf16, or
"int8"/"nf4" through bitsandbytes. The service leaves it None; the benchmark
sets it per run, because its tier system places some models at a compressed
precision to fit a 16 GB card at all. Every ``from_pretrained`` here goes
through ``_load_kwargs()`` so that setting cannot be missed by one call site.

One deliberate trade-off remains: no Ollama. It cannot accept audio input at
all, so keeping it would mean two serving paths side by side when most of
these models do both roles.
"""
import gc
import tempfile
import threading
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    Qwen2AudioForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    VoxtralForConditionalGeneration,
)

import config


def _quantization_config():
    """A `BitsAndBytesConfig` for `config.QUANTIZATION`, or None for native.

    fp16 compute rather than bf16 because the benchmark's cards are Turing
    (T4), which has no bf16 units at all -- asking for it there is slow at
    best. Double quantization on the 4-bit path saves a further ~0.4 bits per
    weight, which is the difference between fitting and not at the 30B tier.
    """
    if not config.QUANTIZATION:
        return None
    if config.QUANTIZATION == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if config.QUANTIZATION == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    raise ValueError(
        f"unknown QUANTIZATION {config.QUANTIZATION!r} -- expected None, 'int8' or 'nf4'")


def _load_kwargs(**extra) -> dict:
    """The `from_pretrained` kwargs every model class here shares.

    One place, so a model added later cannot quietly ignore `DEVICE_MAP` or
    `QUANTIZATION` the way each hand-written call site could.
    """
    kwargs = {"device_map": config.DEVICE_MAP, **extra}
    quantization = _quantization_config()
    if quantization is not None:
        kwargs["quantization_config"] = quantization
        # bitsandbytes picks its own storage dtype for the quantized weights;
        # a dtype= alongside it is either ignored or an outright conflict.
        kwargs.pop("dtype", None)
    return kwargs


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
            self.model_id, **_load_kwargs(dtype="auto")
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
            self.model_id, **_load_kwargs(attn_implementation="sdpa")
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
                # str(), not the Path itself: the processor accepts a numpy
                # array or a string (URL, local path, base64) and rejects a
                # pathlib.Path with "Incorrect format used for `audio`".
                # dataset.Item.audio is a Path, so this is the normal case.
                content.append({"type": "audio", "url": str(audio_path)})
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
            self.model_id, **_load_kwargs()
        )

    def chat(self, messages, audio_path=None, temperature=0.3):
        last_user = _last_user_index(messages) if audio_path else -1
        converted = []
        for i, m in enumerate(messages):
            content = [{"type": "text", "text": m["content"]}]
            if audio_path and i == last_user:
                # audio part first, matching the model's own reference examples
                # str() for the same reason as GemmaAudioModel above -- a
                # pathlib.Path is not one of the accepted audio formats.
                content = [{"type": "audio", "path": str(audio_path)}] + content
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

    Every role, `system` included, gets `content` as a list of typed parts
    (`[{"type": "text", "text": ...}]`), unlike GemmaAudioModel's plain
    string for `system` -- an earlier version copied that convention on the
    assumption the two share a template, and it does not: MedGemma's own
    `chat_template.jinja` iterates `message['content']` expecting a list,
    and a plain string there gets iterated character by character, each
    character then failing `item['type']` with `TypeError: string indices
    must be integers, not 'str'`. Confirmed against a live run -- the bug
    this file had before that (empty replies) was unrelated, a routing bug
    in llm.py that meant this class was never even reached; do not
    reintroduce the plain-string branch based on that history repeating.
    """

    supports_audio = False

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **_load_kwargs(dtype="auto")
        )

    def chat(self, messages, audio_path=None, temperature=0.3):
        if audio_path:
            raise ValueError(f"{self.model_id} is text-only and can't accept audio input")
        converted = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in messages]
        inputs = self._processor.apply_chat_template(
            converted, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        # Device only, deliberately no dtype cast: unlike GemmaAudioModel
        # (which has real floating-point audio tensors to cast), this class
        # never sends an image, so the only tensors here are input_ids /
        # attention_mask -- integer tensors that a blind dtype=float16/bf16
        # cast has no legitimate reason to touch.
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        eos_id = self._processor.tokenizer.eos_token_id
        tok_pad_id = self._processor.tokenizer.pad_token_id
        pad_id = tok_pad_id if tok_pad_id is not None else eos_id
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                eos_token_id=eos_id, pad_token_id=pad_id,
                **_generation_kwargs(temperature))
        new_tokens = outputs[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        if not text.strip() and new_tokens.numel() > 0:
            # skip_special_tokens=True stripped everything -- the model
            # generated something, just not ordinary text. Falling back to
            # the raw decode turns a silent empty reply into a diagnosable
            # one (pipeline.py's error snippet then shows what was actually
            # produced instead of '').
            text = self._processor.decode(new_tokens, skip_special_tokens=False)
        return text


def _patch_sliding_window_cache():
    """Restore an importable `SlidingWindowCache` name in
    `transformers.cache_utils`, if the installed transformers removed it.

    Idempotent and scoped to just that one name -- runs every time
    Phi4MultimodalModel.load() does, cheap, and harmless to call again if
    the name already exists (real or already patched).
    """
    import transformers.cache_utils as cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        cache_utils.SlidingWindowCache = cache_utils.DynamicCache


class Phi4MultimodalModel(BaseLLM):
    """Phi-4-multimodal-instruct. Text and audio, via Microsoft's own custom
    modeling code (trust_remote_code=True).

    Confirmed live: the installed transformers has no native Phi-4-multimodal
    support (omitting trust_remote_code produced an interactive "run custom
    code? [y/N]" prompt, which hangs forever in a non-interactive notebook
    cell -- there is no native path to fall back to here). So the vendor code
    path is mandatory, which means its stale import has to be worked around
    directly rather than avoided: `modeling_phi4mm.py` does
    `from transformers.cache_utils import Cache, DynamicCache,
    SlidingWindowCache, StaticCache`, and `SlidingWindowCache` was removed
    from that module's public API in transformers v4.48.

    Patched in, not pinned: downgrading transformers globally to get
    SlidingWindowCache back risks breaking every other model in this file,
    several of which need a fairly recent transformers already (MedGemma's
    "fast" image processor, Gemma 4's AutoModelForMultimodalLM). Instead,
    `_patch_sliding_window_cache()` aliases `SlidingWindowCache` to
    `DynamicCache` (unbounded, not size-limited the way a real sliding
    window is) only if the name is missing, only in this process, before the
    vendor file ever imports it -- enough for the import itself to succeed.
    Correctness caveat: if Phi-4-multimodal's forward pass actually depends
    on sliding-window *behaviour* (not just the class existing), this is a
    functional approximation, not a faithful implementation -- worth
    revisiting if generation quality looks off specifically for this model.
    """

    supports_audio = True

    def load(self):
        _patch_sliding_window_cache()
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # The checkpoint's own config defaults to flash_attention_2, which
        # needs the flash_attn package -- slow to build on Kaggle (CUDA/torch
        # version matching, long compile) and not installed. Passing
        # attn_implementation= directly to from_pretrained() did not
        # override it (confirmed live: identical error either way) -- this
        # custom model's config class evidently does not honour that kwarg
        # the standard way. Forcing it on a pre-loaded AutoConfig instead,
        # before the model ever sees it, is the more direct path. "eager"
        # rather than "sdpa": this custom architecture's own attention class
        # may not have a working SDPA path registered, since trust_remote_code
        # repos don't always implement every backend transformers supports --
        # eager needs no optimized kernel at all, so it works regardless.
        model_config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        model_config._attn_implementation = "eager"
        # No device_map here, deliberately: transformers' default
        # from_pretrained() path constructs the model on the meta device
        # first (real allocation deferred until weights load, the standard
        # fast-init transformers/accelerate use for every model now) --
        # fine for ordinary modules, but this checkpoint's own
        # speech_conformer_encoder.py computes a real value inside __init__
        # and calls .item() on it, which meta tensors cannot do
        # ("Tensor.item() cannot be called on meta tensors", confirmed
        # live). low_cpu_mem_usage=False disables that fast-init path, but
        # recent transformers raises if device_map and
        # low_cpu_mem_usage=False are passed together -- so device_map is
        # dropped here and the whole model is moved to the target device
        # afterward instead, once real (non-meta) weights exist to move.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, config=model_config, dtype="auto",
            trust_remote_code=True, low_cpu_mem_usage=False,
        ).to(config.DEVICE_MAP)
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


class VoxtralModel(BaseLLM):
    """Mistral's Voxtral -- a Whisper encoder, a projector and a Ministral
    language model, exposed as one `VoxtralForConditionalGeneration`.

    The lightest audio-in chat model here by some margin (~4.7B all in), which
    is the reason it is registered: it is the only one that fits a single 16 GB
    card at fp16 with room to spare for a long generation.

    Two conventions differ from the Gemma/Qwen classes above:

    * `system` content stays a plain string. Voxtral tokenizes through
      mistral-common rather than a Jinja template, and mistral-common's
      `SystemMessage` takes text, not a list of typed parts.
    * `apply_chat_template` returns model-ready inputs directly -- no
      `tokenize=`/`return_dict=` arguments, and the audio named by `path` is
      loaded by the processor itself.

    That audio loading is why `mistral-common[audio]` is a real install-time
    dependency, not an optional extra: without it the `path` part raises
    rather than degrading to text-only.
    """

    supports_audio = True

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_id, **_load_kwargs(dtype=torch.float16)
        )

    def chat(self, messages, audio_path=None, temperature=0.3):
        last_user = _last_user_index(messages) if audio_path else -1
        converted = []
        for i, m in enumerate(messages):
            if m["role"] == "system":
                converted.append({"role": "system", "content": m["content"]})
                continue
            content = []
            if audio_path and i == last_user:
                # Audio first, matching the model's own reference examples.
                # str(), not the Path itself -- see GemmaAudioModel.
                content.append({"type": "audio", "path": str(audio_path)})
            if m["content"]:
                content.append({"type": "text", "text": m["content"]})
            converted.append({"role": m["role"], "content": content})

        inputs = self._processor.apply_chat_template(converted).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                           **_generation_kwargs(temperature))
        return self._processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True)[0]


class Qwen2AudioModel(BaseLLM):
    """Qwen2-Audio-7B-Instruct, via `Qwen2AudioForConditionalGeneration`.

    Alone among the audio classes here, its processor does **not** load the
    audio file itself: `apply_chat_template` is text-only (`tokenize=False`)
    and the waveform goes to `processor(text=..., audio=[...])` separately,
    already decoded and resampled to the feature extractor's rate. That is the
    shape the model's own documentation uses, and skipping the resample is not
    optional -- the mel features are defined at 16 kHz and a 44.1 kHz array
    silently produces a four-times-too-long spectrogram rather than an error.

    `system` content is a plain string, as in that same documentation.
    """

    supports_audio = True

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, **_load_kwargs(dtype=torch.float16)
        )

    def chat(self, messages, audio_path=None, temperature=0.3):
        import librosa

        last_user = _last_user_index(messages) if audio_path else -1
        converted = []
        for i, m in enumerate(messages):
            if m["role"] == "system":
                converted.append({"role": "system", "content": m["content"]})
                continue
            content = []
            if audio_path and i == last_user:
                content.append({"type": "audio", "audio_url": str(audio_path)})
            if m["content"]:
                content.append({"type": "text", "text": m["content"]})
            converted.append({"role": m["role"], "content": content})

        text = self._processor.apply_chat_template(
            converted, add_generation_prompt=True, tokenize=False)
        audios = None
        if audio_path:
            waveform, _ = librosa.load(
                str(audio_path), sr=self._processor.feature_extractor.sampling_rate)
            audios = [waveform]

        inputs = self._processor(text=text, audio=audios, return_tensors="pt",
                                 padding=True).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
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
    "voxtral-mini-3b": (VoxtralModel, config.VOXTRAL_MINI_MODEL_ID),
    "qwen2-audio-7b": (Qwen2AudioModel, config.QWEN2_AUDIO_MODEL_ID),
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
