"""Running the models: windowing, stitching, and one replica per GPU.

Three things here exist to keep a comparison honest rather than to make it
fast:

* **Windowing.** Whisper's encoder is fixed at 30 seconds. Left alone it
  transcribes the first half-minute of a four-minute dictation and returns,
  which reads as a catastrophic WER caused by the model rather than by the
  harness. Every model gets the same overlapping windows.
* **One replica per device.** Both T4s run the same model over different halves
  of the batch, so throughput doubles without changing a single number in the
  result -- the alternative, one model per GPU, would have the two models
  competing for bandwidth and make the latency figures meaningless.
* **Failures are recorded, not raised.** One model that OOMs on one recording
  must not discard the other nine models' results.
"""
import threading
import time
from dataclasses import asdict, dataclass, field

import bridge
import dataset
import settings


@dataclass
class Transcript:
    """One model's attempt at one recording."""
    asset_id: str
    model: str
    text: str = ""
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    windows: int = 0
    device: str = ""
    error: str | None = None

    @property
    def real_time_factor(self):
        """Seconds of compute per second of audio. Below 1.0 is faster than
        real time, which is the bar for dictating and reading back in a clinic."""
        return self.elapsed_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def as_dict(self):
        return {**asdict(self), "real_time_factor": round(self.real_time_factor, 4)}


@dataclass
class ModelRun:
    """Everything one model produced over the whole batch."""
    model: str
    model_id: str = ""
    devices: list = field(default_factory=list)
    transcripts: list = field(default_factory=list)
    load_seconds: float = 0.0
    peak_vram_bytes: int = 0

    def summary(self):
        done = [t for t in self.transcripts if t.error is None]
        audio = sum(t.audio_seconds for t in done)
        compute = sum(t.elapsed_seconds for t in done)
        return {
            "model": self.model,
            "model_id": self.model_id,
            "devices": self.devices,
            "transcribed": len(done),
            "failed": len(self.transcripts) - len(done),
            "load_seconds": round(self.load_seconds, 2),
            "peak_vram_gb": round(self.peak_vram_bytes / 1024 ** 3, 2),
            "audio_seconds": round(audio, 1),
            # Aggregate RTF from summed totals, not a mean of per-file ratios:
            # a two-second clip with a fixed startup cost would otherwise
            # dominate an average built from hours of real dictation.
            "real_time_factor": round(compute / audio, 4) if audio else 0.0,
        }


# --- Windowing -------------------------------------------------------------
def plan_for(preprocessing, duration, audio=None, sample_rate=None,
             window_sec=None, overlap_sec=None):
    """The windows one preprocessing variant asks for.

    `fixed` is this module's own even windowing. The others come from
    `preprocessing/chunking.py`, which is where the real strategies live --
    `adaptive` listens to the recording and nudges every boundary onto the
    quietest moment nearby, so a cut lands between words rather than through
    one. Reimplementing that here would be a second copy of the thing the
    benchmark exists to compare.

    Falls back to even windowing if preprocessing is unavailable, because a
    missing optional dependency should cost a variant, not the run.
    """
    if preprocessing in (None, "fixed"):
        return plan_windows(duration, window_sec, overlap_sec)

    try:
        plan_chunks, config = bridge.chunk_planner()
    except Exception:
        return plan_windows(duration, window_sec, overlap_sec)

    strategy = "adaptive" if preprocessing.startswith("adaptive") else preprocessing
    settings_for_run = {**config, "chunking": {**config["chunking"], "strategy": strategy}}

    # The "-vad" variants chunk within the detected speech regions instead of
    # across the whole recording, so a long pause is a boundary rather than
    # something a window has to spend itself on.
    regions = None
    if preprocessing.endswith("-vad") and audio is not None:
        regions = bridge.speech_regions(audio, sample_rate)

    # Adaptive needs the samples to find the quiet moments; without them
    # preprocessing itself downgrades to uniform and says so.
    windows, _ = plan_chunks(duration, settings_for_run, regions=regions,
                             audio=audio, sr=sample_rate)
    return windows or plan_windows(duration, window_sec, overlap_sec)


def plan_windows(duration, window_sec=None, overlap_sec=None):
    """Cut `duration` into overlapping windows, as [(start, end)] in seconds.

    The overlap is what makes stitching possible: a word landing on a cut is
    spoken fully inside one of the two neighbours.
    """
    window_sec = settings.WINDOW_SEC if window_sec is None else window_sec
    overlap_sec = settings.OVERLAP_SEC if overlap_sec is None else overlap_sec
    if duration <= window_sec:
        return [(0.0, duration)]
    if overlap_sec >= window_sec:
        raise ValueError("overlap must be shorter than the window")

    step = window_sec - overlap_sec
    windows = []
    start = 0.0
    while start < duration:
        end = min(start + window_sec, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def stitch(parts, max_overlap_words=None):
    """Join per-window transcripts, dropping what the overlap said twice.

    Where one window ends with the same words the next begins with, that is the
    shared audio being transcribed once each. The longest such run is the seam;
    no match means the two windows disagreed about the overlap, in which case
    keeping both is the safer error -- an omission is invisible to a reader, a
    duplication is not.
    """
    cap = settings.MAX_STITCH_OVERLAP_WORDS if max_overlap_words is None else max_overlap_words
    merged = []
    for part in parts:
        words = part.split()
        if not words:
            continue
        if not merged:
            merged = words
            continue
        merged.extend(words[_seam(merged, words, cap):])
    return " ".join(merged)


def _seam(left, right, cap):
    """Length of the longest suffix of `left` that is also a prefix of `right`."""
    limit = min(cap, len(left), len(right))
    for length in range(limit, 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


# --- Running one model over a batch ----------------------------------------
def transcribe_batch(model_key, items, devices=None, language=None,
                     model_factory=None, on_progress=None, preprocessing=None,
                     **window_kwargs):
    """Load `model_key` on every device and transcribe `items` across them.

    `model_factory(key, device)` is injected so the tests can run the whole
    scheduling, windowing and stitching path against a stub, with no weights
    and no GPU.
    """
    devices = list(devices or resolve_devices())
    factory = model_factory or bridge.build_stt_model
    language = language or settings.DEFAULT_LANGUAGE

    run = ModelRun(model=model_key, devices=devices)
    shards = [items[index::len(devices)] for index in range(len(devices))]
    lock = threading.Lock()

    _reset_vram(devices)
    threads = [
        threading.Thread(
            target=_work_shard, name=f"{model_key}@{device}",
            args=(factory, model_key, device, shard, language, run, lock,
                  on_progress, preprocessing, window_kwargs))
        for device, shard in zip(devices, shards) if shard
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    run.peak_vram_bytes = _peak_vram(devices)
    # Ordered as the caller supplied them; the shards finish interleaved.
    order = {item.asset_id: index for index, item in enumerate(items)}
    run.transcripts.sort(key=lambda t: order.get(t.asset_id, 0))
    run.load_seconds = round(run.load_seconds, 2)
    return run


def _work_shard(factory, model_key, device, shard, language, run, lock,
                on_progress, preprocessing, window_kwargs):
    """One device's share of the batch: load once, then transcribe in turn."""
    try:
        loading = time.perf_counter()
        model = factory(model_key, device)
        model.load()
        load_seconds = time.perf_counter() - loading
    except Exception as error:  # noqa: BLE001 - a model that will not load is a result
        with lock:
            run.transcripts.extend(
                Transcript(asset_id=item.asset_id, model=model_key, device=device,
                           error=f"load failed: {error}")
                for item in shard)
        return

    with lock:
        run.load_seconds = max(run.load_seconds, load_seconds)
        run.model_id = getattr(model, "model_id", "")

    try:
        for item in shard:
            transcript = _transcribe_one(model, model_key, device, item, language,
                                         preprocessing, window_kwargs)
            with lock:
                run.transcripts.append(transcript)
            if on_progress:
                on_progress(transcript)
    finally:
        # Free the replica before the next model loads; two large models
        # resident at once is how a 16 GB T4 runs out.
        model.unload()


def _transcribe_one(model, model_key, device, item, language, preprocessing,
                    window_kwargs):
    transcript = Transcript(asset_id=item.asset_id, model=model_key, device=device)
    try:
        audio, sr = dataset.load_audio(item.audio)
        transcript.audio_seconds = len(audio) / sr
        windows = plan_for(preprocessing, transcript.audio_seconds,
                           audio=audio, sample_rate=sr, **window_kwargs)
        transcript.windows = len(windows)

        started = time.perf_counter()
        parts = [
            model.transcribe(audio[int(start * sr):int(end * sr)], sr, language=language)
            for start, end in windows
        ]
        transcript.elapsed_seconds = time.perf_counter() - started
        transcript.text = stitch(parts)
    except Exception as error:  # noqa: BLE001 - recorded per recording, never fatal
        transcript.error = f"{type(error).__name__}: {error}"
    return transcript


class EchoModel:
    """A stand-in that loads nothing and transcribes nothing.

    A benchmark run downloads gigabytes of weights and then spends hours on
    them, which is a long way to travel before finding out a path was wrong.
    Swapping this in exercises the whole harness -- decoding, windowing,
    stitching, scoring, writing -- in seconds, so the only thing left to be
    wrong is the models themselves.
    """

    def __init__(self, key="dry-run", device="cpu"):
        self.model_id = f"dry-run/{key}"
        self.device = device

    def load(self):
        pass

    def unload(self):
        pass

    def transcribe(self, audio, sr, language=None):
        return ""


def dry_run_factory(key, device):
    return EchoModel(key, device)


# --- Devices ---------------------------------------------------------------
def resolve_devices(spec=None):
    """Turn the `DEVICES` setting into a concrete list of torch devices."""
    spec = settings.DEVICES if spec is None else spec
    if spec and spec != "auto":
        return [device.strip() for device in spec.split(",") if device.strip()]

    torch = bridge.torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def describe_devices(devices=None):
    """What the run is about to use -- printed so a Kaggle notebook that
    quietly fell back to CPU says so before spending an hour proving it."""
    devices = devices or resolve_devices()
    torch = bridge.torch_or_none()
    described = []
    for device in devices:
        if torch is not None and device.startswith("cuda"):
            index = int(device.split(":")[1]) if ":" in device else 0
            properties = torch.cuda.get_device_properties(index)
            described.append({"device": device, "name": properties.name,
                              "total_vram_gb": round(properties.total_memory / 1024 ** 3, 1)})
        else:
            described.append({"device": device, "name": "cpu", "total_vram_gb": 0.0})
    return described


def _reset_vram(devices):
    torch = bridge.torch_or_none()
    if torch is None:
        return
    for device in devices:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)


def _peak_vram(devices):
    """Peak across devices, not the sum: the replicas are identical, so the
    number to report is what one GPU had to hold."""
    torch = bridge.torch_or_none()
    if torch is None:
        return 0
    peaks = [torch.cuda.max_memory_allocated(device)
             for device in devices if device.startswith("cuda")]
    return max(peaks, default=0)
