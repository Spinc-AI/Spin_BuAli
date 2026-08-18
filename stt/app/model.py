'''Model loading, swapping and transcription.

Each supported architecture is a subclass of ``BaseSTTModel``. A new model is
added by writing one subclass, registering it in ``_MODEL_TYPES`` and adding an
entry to ``config.MODEL_REGISTRY`` — nothing in the API layer changes.
'''

import gc
import threading
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    SeamlessM4TModel,
    SeamlessM4Tv2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from app import config

# Map the API's simple language codes onto what each model family expects.
WHISPER_LANGUAGE_NAMES = {"fa": "persian", "en": "english"}
SEAMLESS_LANGUAGE_CODES = {"fa": "pes", "en": "eng"}


# ============================================================
# Audio utilities
# ============================================================

def resample(audio, sr, target_sr=config.TARGET_SAMPLE_RATE):
    '''Resample a mono float32 array to ``target_sr`` if needed.'''
    if sr == target_sr:
        return audio
    import torchaudio

    tensor = torch.from_numpy(np.asarray(audio, np.float32)).unsqueeze(0)
    out = torchaudio.functional.resample(tensor, sr, target_sr)
    return out.squeeze(0).numpy()


# ============================================================
# Model implementations
# ============================================================

class BaseSTTModel(ABC):
    '''An STT model that can be loaded into memory and run on audio.'''

    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    @abstractmethod
    def load(self):
        '''Pull weights and processor into memory on ``self.device``.'''

    @abstractmethod
    def transcribe(self, audio, sr, language=None):
        '''Return the transcription of a mono float32 array sampled at ``sr``.

        ``language`` is one of ``config.SUPPORTED_LANGUAGES`` (e.g. "fa", "en"),
        or ``None`` to use the model's own default behaviour.
        '''

    def unload(self):
        '''Release references and free GPU memory.'''
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class WhisperModel(BaseSTTModel):
    '''Whisper-family conditional generation model.'''

    def load(self):
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self._model.generation_config.forced_decoder_ids = None

    def transcribe(self, audio, sr, language=None):
        audio = resample(audio, sr)
        features = self._processor(
            audio, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)
        if self.device == "cuda":
            features = features.half()

        forced_decoder_ids = None
        if language:
            name = WHISPER_LANGUAGE_NAMES.get(language, language)
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=name, task="transcribe"
            )

        with torch.no_grad():
            ids = self._model.generate(features, forced_decoder_ids=forced_decoder_ids)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]


class SeamlessV2Model(BaseSTTModel):
    '''SeamlessM4T v2 speech-to-text model.'''

    def __init__(self, model_id, device, tgt_lang="pes"):
        super().__init__(model_id, device)
        self.tgt_lang = tgt_lang

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = SeamlessM4Tv2Model.from_pretrained(self.model_id).to(self.device).eval()

    def transcribe(self, audio, sr, language=None):
        audio = resample(audio, sr)
        inputs = self._processor(
            audio=audio, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tgt_lang = SEAMLESS_LANGUAGE_CODES.get(language, language) if language else self.tgt_lang
        with torch.no_grad():
            out = self._model.generate(
                **inputs, tgt_lang=tgt_lang, generate_speech=False
            )
        seqs = out.sequences if hasattr(out, "sequences") else out
        return self._processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)[0]


class SeamlessV1Model(BaseSTTModel):
    '''SeamlessM4T v1 speech-to-text model (e.g. hf-seamless-m4t-medium).

    v1's ``generate(..., generate_speech=False)`` returns token ids directly
    (no ``.sequences`` wrapper like v2), so decoding differs slightly from
    ``SeamlessV2Model``.
    '''

    def __init__(self, model_id, device, tgt_lang="pes"):
        super().__init__(model_id, device)
        self.tgt_lang = tgt_lang

    def load(self):
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = SeamlessM4TModel.from_pretrained(self.model_id).to(self.device).eval()

    def transcribe(self, audio, sr, language=None):
        audio = resample(audio, sr)
        inputs = self._processor(
            audios=audio, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tgt_lang = SEAMLESS_LANGUAGE_CODES.get(language, language) if language else self.tgt_lang
        with torch.no_grad():
            out = self._model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        tokens = out[0] if isinstance(out, (list, tuple)) else out
        return self._processor.decode(tokens.squeeze().tolist(), skip_special_tokens=True)


class CTCModel(BaseSTTModel):
    '''Plain wav2vec2-family CTC model (no language adapter).'''

    def load(self):
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device).eval()

    def transcribe(self, audio, sr, language=None):
        audio = resample(audio, sr)
        inputs = self._processor(
            audio, sampling_rate=config.TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self._model(input_values).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0]


class MMSModel(CTCModel):
    '''Meta MMS CTC model — same as ``CTCModel`` but loads a target-language
    adapter first (MMS ships one shared backbone with per-language adapter
    weights; see https://huggingface.co/facebook/mms-1b-all).
    '''

    def __init__(self, model_id, device, target_lang="fas"):
        super().__init__(model_id, device)
        self.target_lang = target_lang

    def load(self):
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id, target_lang=self.target_lang)
        self._model = (
            Wav2Vec2ForCTC.from_pretrained(self.model_id, target_lang=self.target_lang)
            .to(self.device)
            .eval()
        )
        self._model.load_adapter(self.target_lang)


# ============================================================
# Factory
# ============================================================

_MODEL_TYPES = {
    "whisper": WhisperModel,
    "seamless": SeamlessV2Model,
    "seamless_v2": SeamlessV2Model,
    "seamless_v1": SeamlessV1Model,
    "ctc": CTCModel,
    "mms": MMSModel,
}


def build_model(key):
    '''Instantiate (without loading) the model registered under ``key``.'''
    if key not in config.MODEL_REGISTRY:
        raise KeyError(key)
    spec = dict(config.MODEL_REGISTRY[key])
    model_type = spec.pop("type")
    model_id = spec.pop("model_id")
    cls = _MODEL_TYPES[model_type]
    return cls(model_id=model_id, device=config.DEVICE, **spec)


# ============================================================
# Manager
# ============================================================

class ModelManager:
    '''Holds at most one loaded model and serializes access to it.

    A single lock guards both loading and transcription so that concurrent
    requests cannot swap the model out from under an in-flight transcription.
    '''

    def __init__(self):
        self._current_key = None
        self._current_model = None
        self._lock = threading.Lock()

    def available(self):
        '''Return the list of registered model keys.'''
        return list(config.MODEL_REGISTRY.keys())

    @property
    def loaded(self):
        '''Return the key of the loaded model, or ``None``.'''
        return self._current_key

    def load(self, key):
        '''Load ``key`` into memory, unloading any currently loaded model.'''
        if key not in config.MODEL_REGISTRY:
            raise KeyError(key)
        with self._lock:
            if self._current_key == key:
                return
            if self._current_model is not None:
                self._current_model.unload()
                self._current_model = None
                self._current_key = None
            model = build_model(key)
            model.load()
            self._current_model = model
            self._current_key = key

    def transcribe(self, audio, sr, language=None):
        '''Transcribe audio with the loaded model, erroring if none is loaded.'''
        with self._lock:
            if self._current_model is None:
                raise RuntimeError("no model loaded")
            return self._current_model.transcribe(audio, sr, language=language)

    def unload(self):
        '''Unload the current model and free its memory. No-op if none loaded.'''
        with self._lock:
            if self._current_model is not None:
                self._current_model.unload()
                self._current_model = None
                self._current_key = None
