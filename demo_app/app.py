"""Spin BuAli — a desktop client for the BuAli controller.

One window: pick a pipeline, configure the STT slots and/or an audio-capable
LLM, choose or record a recording, run it, and save the report.

    python app.py

The controller must be running (default localhost:9002).
"""
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from api import BuAliClient, ControllerError
from audio import RecordButton
from config import (
    AUDIO_FILETYPES,
    DEFAULT_CLOUD_LLM_MODEL,
    GEMINI_HINT,
    MAX_STT_SLOTS,
    PIPELINE_LABELS,
)
from export import TranscriptSaver
from widgets import ConnectionBar, CloudFields, ModeSelector, OutputBox, ScrollableFrame, SttSlotWidget


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spin BuAli — Demo")
        self.geometry("880x760")
        self.minsize(700, 500)

        self.client = BuAliClient(lambda: self.connection.base_url)
        self._audio_path: str | None = None
        self._report: dict = {}
        self._local_stt_models: list[str] = []
        self._local_llm_models: list[str] = []
        self._audio_llm_models: list[str] = []

        scroller = ScrollableFrame(self)
        scroller.pack(fill="both", expand=True)
        self._build(scroller.body)

        self._on_pipeline_change()
        # Deferred: the worker hands results back via after(), which needs the
        # event loop to already be running.
        self.after(0, self.refresh_models)

    # --- layout ------------------------------------------------------------
    def _build(self, root):
        self.connection = ConnectionBar(root, on_check=self.check_connection)
        self.connection.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(root, text="Pipeline:").grid(row=1, column=0, sticky="w")
        self.pipeline_label = tk.StringVar(value=next(iter(PIPELINE_LABELS)))
        ttk.Combobox(root, textvariable=self.pipeline_label, width=28, state="readonly",
                     values=list(PIPELINE_LABELS)).grid(row=1, column=1, sticky="w", padx=(6, 0))
        self.pipeline_label.trace_add("write", lambda *_: self._on_pipeline_change())

        self.slots_frame = ttk.LabelFrame(root, text="STT slots", padding=8)
        self.slots_frame.grid(row=2, column=0, columnspan=4, sticky="we", pady=(10, 0))
        self.slots = []
        for index in range(MAX_STT_SLOTS):
            slot = SttSlotWidget(self.slots_frame, f"STT slot {index + 1} source:",
                                 on_refresh=self.refresh_models)
            slot.grid(row=index, column=0, sticky="w", pady=(4 if index else 0, 0))
            self.slots.append(slot)

        self._build_llm_section(root)

        ttk.Button(root, text="Start session", command=self.start_session).grid(
            row=4, column=0, sticky="w", pady=10)
        ttk.Button(root, text="Unload session", command=self.unload_session).grid(
            row=4, column=1, sticky="w")
        self.session_status = ttk.Label(root, text="session: none")
        self.session_status.grid(row=5, column=0, columnspan=4, sticky="w")

        ttk.Label(root, text="Audio file:").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.audio_path = tk.StringVar()
        ttk.Entry(root, textvariable=self.audio_path, width=45, state="readonly").grid(
            row=6, column=1, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(root, text="Browse...", command=self.browse).grid(
            row=6, column=3, sticky="w", pady=(10, 0))
        RecordButton(root, self.audio_path).grid(row=7, column=0, sticky="w", pady=(4, 0))
        ttk.Button(root, text="Clear", command=lambda: self.audio_path.set("")).grid(
            row=7, column=1, sticky="w", pady=(4, 0))

        ttk.Button(root, text="Run", command=self.run).grid(row=8, column=0, sticky="w", pady=10)

        ttk.Label(root, text="Output:").grid(row=9, column=0, sticky="nw")
        self.output = OutputBox(root)
        self.output.container.grid(row=10, column=0, columnspan=4, sticky="we", pady=(0, 10))

        self.saver = TranscriptSaver(root, self._report_for_saving)
        self.saver.grid(row=11, column=0, columnspan=4, sticky="w")

    def _build_llm_section(self, root):
        frame = ttk.LabelFrame(root, text="LLM", padding=8)
        frame.grid(row=3, column=0, columnspan=4, sticky="we", pady=(10, 0))

        self.llm_mode = ModeSelector(frame, "LLM source:")
        self.llm_mode.grid(row=0, column=0, columnspan=4, sticky="w")
        self.llm_mode.on_change(self._on_pipeline_change)

        self.llm_local_model = tk.StringVar()
        self.llm_local_box = ttk.Combobox(frame, textvariable=self.llm_local_model, width=18,
                                          state="readonly")
        self.llm_cloud = CloudFields(frame, DEFAULT_CLOUD_LLM_MODEL)
        self.gemini_hint = ttk.Label(frame, text=GEMINI_HINT, foreground="gray",
                                     wraplength=520, justify="left")

    # --- threading ----------------------------------------------------------
    def in_background(self, work, on_success=None, on_error=None):
        """Run `work()` off the UI thread and deliver the outcome back on it.

        Tk is single-threaded, so results come back through `after()` rather
        than touching widgets from the worker.
        """
        def run():
            try:
                result = work()
            except ControllerError as exc:
                if on_error:
                    self.after(0, on_error, str(exc))
                return
            if on_success:
                self.after(0, on_success, result)

        threading.Thread(target=run, daemon=True).start()

    # --- state --------------------------------------------------------------
    @property
    def pipeline(self) -> str:
        return PIPELINE_LABELS[self.pipeline_label.get()]

    @property
    def needs_audio_llm(self) -> bool:
        return self.pipeline in ("multimodal", "hybrid")

    @property
    def llm_model(self) -> str:
        """The model name, with the prefix the controller routes on."""
        if not self.llm_mode.is_cloud():
            return self.llm_local_model.get().strip()
        model = self.llm_cloud.model.get().strip()
        return model if model.startswith("gemini:") else "openai:" + model

    def _on_pipeline_change(self):
        if self.pipeline == "multimodal":
            self.slots_frame.grid_remove()
        else:
            self.slots_frame.grid()

        models = self._audio_llm_models if self.needs_audio_llm else self._local_llm_models
        self.llm_local_box["values"] = models
        if models and self.llm_local_model.get() not in models:
            self.llm_local_model.set(models[0])

        if self.llm_mode.is_cloud():
            self.llm_local_box.grid_remove()
            self.llm_cloud.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.llm_cloud.grid_remove()
            self.llm_local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))

        if self.needs_audio_llm and self.llm_mode.is_cloud():
            self.gemini_hint.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.gemini_hint.grid_remove()

    # --- controller calls ---------------------------------------------------
    def check_connection(self):
        self.connection.show_checking()
        self.in_background(self.client.is_reachable, on_success=self.connection.show_result)

    def refresh_models(self):
        """Pull both model registries from the controller, so nothing is hardcoded."""
        def fetch():
            return self.client.stt_models(), self.client.llm_models()

        def apply(result):
            stt_models, (llm_models, audio_models) = result
            self._local_stt_models = stt_models
            self._local_llm_models = llm_models
            self._audio_llm_models = audio_models
            for slot in self.slots:
                slot.set_local_models(stt_models)
            self._on_pipeline_change()

        self.in_background(fetch, on_success=apply,
                      on_error=lambda detail: messagebox.showerror(
                          "BuAli", f"Could not fetch models: {detail}"))

    def start_session(self):
        if not self.llm_model.strip(":"):
            messagebox.showwarning(
                "BuAli",
                "No LLM model selected. Local models are listed from the controller — "
                "check the connection and press Refresh, or switch to a cloud model.")
            return

        payload = {"pipeline": self.pipeline, "llm_model": self.llm_model}
        if self.pipeline != "multimodal":
            payload["stt_slots"] = [slot.as_slot_config() for slot in self.slots]
        if self.llm_mode.is_cloud():
            payload.update({f"llm_{key}": value
                            for key, value in self.llm_cloud.credentials().items()})

        self.session_status.config(text="starting session...")
        self.in_background(
            lambda: self.client.start_session(payload),
            on_success=lambda body: self.session_status.config(
                text=f"session: pipeline={body.get('pipeline')}, llm={body.get('llm_model')}"),
            on_error=lambda detail: self.session_status.config(text=f"session failed: {detail}"),
        )

    def unload_session(self):
        self.in_background(
            self.client.unload_session,
            on_success=lambda _: self.session_status.config(text="session: none"),
            on_error=lambda detail: messagebox.showerror("BuAli", f"Unload failed: {detail}"),
        )

    def browse(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
        if path:
            self.audio_path.set(path)

    def run(self):
        path = self.audio_path.get()
        if not path:
            messagebox.showwarning("BuAli", "Choose or record an audio file first.")
            return
        self._audio_path = path
        self.saver.suggest_dir_from(path)

        overrides = ({f"llm_{key}": value
                      for key, value in self.llm_cloud.credentials().items()}
                     if self.llm_mode.is_cloud() else {})

        self.output.write("running...")

        def show(body):
            self._report = body.get("result", {})
            self.output.write(json.dumps(body, indent=2, ensure_ascii=False))

        self.in_background(lambda: self.client.run(path, overrides), on_success=show,
                      on_error=lambda detail: self.output.write(f"Error: {detail}"))

    def _report_for_saving(self):
        text = self._report.get("final_text") or self._report.get("corrected_transcript") or ""
        return text, self._audio_path, [self.llm_model]


if __name__ == "__main__":
    app = App()
    app.mainloop()
