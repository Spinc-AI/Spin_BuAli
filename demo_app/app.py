"""Spin BuAli — a desktop client for the BuAli controller.

One window: pick a pipeline, configure the STT slots and/or an audio-capable
LLM, choose or record a recording, run it, and save the report.

    python app.py

The controller must be running (default localhost:9002).
"""
import json
import threading
import tkinter as tk
import uuid
from tkinter import filedialog, messagebox, ttk

from api import BuAliClient, ControllerError
from audio import RecordButton
from config import (
    AUDIO_FILETYPES,
    DEFAULT_CLOUD_LLM_MODEL,
    GEMINI_HINT,
    MAX_STT_SLOTS,
    PIPELINE_LABELS,
    PREPROCESSING_VERSION = "legacy-v1"
    PIPELINE_VERSION = "buali-v1"
    PROMPT_VERSION = "radiology-v1"


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



    def _build_processing_config_and_credentials(self):
    pipeline = self._pipeline_value()

    stt_slots = []
    stt_credentials = {}

    if pipeline != "multimodal":
        for index, widget in enumerate(self.slot_widgets):
            if not widget.enabled.get():
                stt_slots.append(None)
                continue

            slot_id = f"stt_{index + 1}"

            slot_config = {
                "slot_id": slot_id,
                "model": widget.effective_model(),
            }

            stt_slots.append(slot_config)

            if widget.is_cloud():
                credential = {}

                api_key = widget.cloud.api_key.get().strip()
                base_url = widget.cloud.base_url.get().strip()

                if api_key:
                    credential["api_key"] = api_key

                if base_url:
                    credential["base_url"] = base_url

                if credential:
                    stt_credentials[slot_id] = credential

    processing_config = {
        "pipeline": pipeline,
        "language": "fa",
        "stt_slots": stt_slots,
        "llm_model": self._effective_llm_model(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dictionary_version": None,
    }

    execution_credentials = {
        "stt": stt_credentials,
    }

    if self.llm_mode.is_cloud():
        llm_credential = {}

        api_key = self.llm_cloud.api_key.get().strip()
        base_url = self.llm_cloud.base_url.get().strip()

        if api_key:
            llm_credential["api_key"] = api_key

        if base_url:
            llm_credential["base_url"] = base_url

        if llm_credential:
            execution_credentials["llm"] = llm_credential

    return processing_config, execution_credentials


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
    """Development-only local validation; no Session is created in AI."""

    try:
        config_data, _ = self._build_processing_config_and_credentials()

        pipeline = config_data["pipeline"]
        configured_slots = [
            slot
            for slot in config_data["stt_slots"]
            if slot is not None
        ]

        if pipeline in ("separate", "hybrid") and not configured_slots:
            raise ValueError(
                f"{pipeline} requires at least one STT slot"
            )

        self.session_status.config(
            text=(
                "settings ready: "
                f"pipeline={pipeline}, "
                f"llm={config_data['llm_model']}"
            )
        )

    except Exception as exc:
        self.session_status.config(
            text=f"invalid settings: {exc}"
        )

    
    def unload_session(self):
        """Clear only the local Demo status."""
    
        self.session_status.config(
            text="settings: not validated"
        )


    def browse(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
        if path:
            self.audio_path.set(path)


    def _run_bg(
        self,
        path,
        processing_config,
        credentials,
        token,
    ):
        job_id = f"dev-{uuid.uuid4().hex}"
    
        endpoint = (
            f"{self.conn.base_url}"
            f"/internal/jobs/{job_id}/execute"
        )
    
        headers = {
            "X-Internal-Token": token,
        }
    
        data = {
            "config_json": json.dumps(
                processing_config,
                ensure_ascii=False,
            ),
            "credentials_json": json.dumps(
                credentials,
                ensure_ascii=False,
            ),
        }
    
        try:
            with open(path, "rb") as audio_file:
                files = {
                    "file": (
                        os.path.basename(path),
                        audio_file,
                    )
                }
    
                response = requests.post(
                    endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=TIMEOUT_LOAD,
                )
    
            response.raise_for_status()
    
            body = response.json()
    
            self._last_result = body.get(
                "result",
                {},
            )
    
            pretty = json.dumps(
                body,
                indent=2,
                ensure_ascii=False,
            )
    
            self.after(
                0,
                self.output.write,
                pretty,
            )
    
        except requests.RequestException as exc:
            self.after(
                0,
                self.output.write,
                f"Error: {error_detail(exc)}",
            )
    
    def run(self):
        path = self.file_path.get().strip()
    
        if not path:
            messagebox.showwarning(
                "BuAli",
                "Choose or record an audio file first.",
            )
            return
    
        token = self.conn.internal_token.get().strip()
    
        if not token:
            messagebox.showwarning(
                "BuAli",
                "Enter the AI Controller internal token.",
            )
            return
    
        try:
            processing_config, credentials = (
                self._build_processing_config_and_credentials()
            )
        except Exception as exc:
            messagebox.showerror(
                "BuAli",
                f"Invalid processing configuration: {exc}",
            )
            return
    
        self._last_audio_path = path
        self.saver.note_audio_path(path)
    
        self.output.write("running...")
    
        run_bg(
            self._run_bg,
            path,
            processing_config,
            credentials,
            token,
        )

        
    

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
