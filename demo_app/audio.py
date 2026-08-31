"""Microphone capture.

`sounddevice` needs PortAudio and a real input device; when either is missing
the import fails, and the app stays usable for file-based runs (the button
reports why on click instead of crashing at startup).
"""
import queue
import tempfile
import wave
from tkinter import messagebox, ttk

from config import MIC_SAMPLE_RATE

try:
    import sounddevice
    MIC_UNAVAILABLE_REASON = None
except Exception as exc:  # missing PortAudio, no input device, ...
    sounddevice = None
    MIC_UNAVAILABLE_REASON = exc


class MicRecorder:
    """Records mono 16-bit PCM from the default input until stopped."""

    def __init__(self, samplerate=MIC_SAMPLE_RATE, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self._chunks: queue.Queue[bytes] = queue.Queue()
        self._stream = None

    def start(self) -> None:
        self._chunks = queue.Queue()
        self._stream = sounddevice.RawInputStream(
            samplerate=self.samplerate, channels=self.channels, dtype="int16",
            callback=lambda data, *_: self._chunks.put(bytes(data)),
        )
        self._stream.start()

    def stop_and_save(self) -> str:
        """Stop recording and write a temporary .wav, returning its path."""
        self._stream.stop()
        self._stream.close()
        frames = []
        while not self._chunks.empty():
            frames.append(self._chunks.get())

        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(path, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # int16
            wav.setframerate(self.samplerate)
            wav.writeframes(b"".join(frames))
        return path


class RecordButton(ttk.Button):
    """Toggles recording, writing the finished file's path into `path_var`."""

    def __init__(self, parent, path_var):
        super().__init__(parent, text="Record mic", command=self._toggle)
        self._path_var = path_var
        self._recorder: MicRecorder | None = None

    def _toggle(self) -> None:
        if self._recorder is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if sounddevice is None:
            messagebox.showerror(
                "Microphone", f"Microphone support unavailable: {MIC_UNAVAILABLE_REASON}")
            return
        try:
            self._recorder = MicRecorder()
            self._recorder.start()
        except Exception as exc:
            messagebox.showerror("Microphone", f"Could not start recording: {exc}")
            self._recorder = None
            return
        self.config(text="Stop recording")

    def _stop(self) -> None:
        try:
            self._path_var.set(self._recorder.stop_and_save())
        except Exception as exc:
            messagebox.showerror("Microphone", f"Could not save recording: {exc}")
        finally:
            self._recorder = None
            self.config(text="Record mic")
