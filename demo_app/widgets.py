"""Reusable Tk widgets.

These are deliberately passive: they hold and display state, and hand work
back to the app through callbacks. Nothing here does HTTP or threading.
"""
import tkinter as tk
from tkinter import ttk

from config import (
    CLOUD_LABEL,
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_CLOUD_STT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOCAL_LABEL,
)


class ModeSelector(ttk.Frame):
    """Label plus a dropdown choosing a local or a cloud model."""

    def __init__(self, parent, label_text):
        super().__init__(parent)
        ttk.Label(self, text=label_text).grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value=LOCAL_LABEL)
        ttk.Combobox(self, textvariable=self.mode, width=20, state="readonly",
                     values=[LOCAL_LABEL, CLOUD_LABEL]).grid(row=0, column=1, sticky="w",
                                                             padx=(6, 0))

    def is_cloud(self) -> bool:
        return self.mode.get() == CLOUD_LABEL

    def on_change(self, callback):
        self.mode.trace_add("write", lambda *_: callback())


class CloudFields(ttk.Frame):
    """Model name, API key and base URL for a cloud provider."""

    def __init__(self, parent, default_model):
        super().__init__(parent)
        ttk.Label(self, text="Cloud model:").grid(row=0, column=0, sticky="w")
        self.model = tk.StringVar(value=default_model)
        ttk.Entry(self, textvariable=self.model, width=18).grid(row=0, column=1, sticky="w",
                                                                padx=(0, 12))

        ttk.Label(self, text="API key:").grid(row=0, column=2, sticky="w")
        self.api_key = tk.StringVar()
        ttk.Entry(self, textvariable=self.api_key, show="*", width=24).grid(row=0, column=3,
                                                                           sticky="w")

        ttk.Label(self, text="Base URL:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.base_url = tk.StringVar(value=DEFAULT_CLOUD_BASE_URL)
        ttk.Entry(self, textvariable=self.base_url, width=40).grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))

    def credentials(self) -> dict:
        """The non-empty api_key/base_url, ready to merge into a request body."""
        return {key: value for key, value in
                (("api_key", self.api_key.get().strip()),
                 ("base_url", self.base_url.get().strip())) if value}


class SttSlotWidget(ttk.Frame):
    """One STT engine slot: on/off, local or cloud, and the fields for either."""

    def __init__(self, parent, label_text, on_refresh):
        super().__init__(parent)
        self._on_refresh = on_refresh

        self.enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Use this slot", variable=self.enabled,
                        command=self._show_relevant_fields).grid(row=0, column=0, sticky="w")

        self.mode = ModeSelector(self, label_text)
        self.mode.grid(row=0, column=1, columnspan=3, sticky="w", padx=(10, 0))
        self.mode.on_change(self._show_relevant_fields)

        self.local_model = tk.StringVar()
        self.local_box = ttk.Combobox(self, textvariable=self.local_model, width=18,
                                      state="readonly")
        self.refresh_button = ttk.Button(self, text="Refresh", command=on_refresh)
        self.cloud = CloudFields(self, DEFAULT_CLOUD_STT_MODEL)

        self._show_relevant_fields()

    def set_local_models(self, models: list[str]) -> None:
        self.local_box["values"] = models
        if models and self.local_model.get() not in models:
            self.local_model.set(models[0])

    def _show_relevant_fields(self) -> None:
        for widget in (self.local_box, self.refresh_button, self.cloud):
            widget.grid_remove()
        if not self.enabled.get():
            return
        if self.mode.is_cloud():
            self.cloud.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            self.local_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.refresh_button.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(4, 0))

    def as_slot_config(self, slot_id: str) -> dict | None:
        """This slot as a ProcessingConfig entry, or None if unused.

        Deliberately without the API key: configuration is storable and
        hashable, credentials are not. `as_credential` returns that half.
        """
        if not self.enabled.get():
            return None
        config = {"slot_id": slot_id}
        if self.mode.is_cloud():
            config["model"] = "openai:" + self.cloud.model.get().strip()
            base_url = self.cloud.base_url.get().strip()
            if base_url:
                config["base_url"] = base_url
        else:
            config["model"] = self.local_model.get().strip()
        return config

    def as_credential(self) -> dict | None:
        """This slot's secret, sent per request and never stored."""
        if not self.enabled.get() or not self.mode.is_cloud():
            return None
        return self.cloud.credentials() or None


class ConnectionBar(ttk.Frame):
    """Host/port entry and a connection indicator. The app performs the check."""

    def __init__(self, parent, on_check):
        super().__init__(parent)
        ttk.Label(self, text="Host:").grid(row=0, column=0, padx=(0, 4))
        self.host = tk.StringVar(value=DEFAULT_HOST)
        ttk.Entry(self, textvariable=self.host, width=16).grid(row=0, column=1, padx=(0, 8))

        ttk.Label(self, text="Port:").grid(row=0, column=2, padx=(0, 4))
        self.port = tk.StringVar(value=str(DEFAULT_PORT))
        ttk.Entry(self, textvariable=self.port, width=6).grid(row=0, column=3, padx=(0, 8))

        ttk.Button(self, text="Check connection", command=on_check).grid(row=0, column=4,
                                                                        padx=(0, 8))
        self._indicator = ttk.Label(self, text="● unknown", foreground="gray")
        self._indicator.grid(row=0, column=5)

        # Every route but the health check needs this. Shown masked, and never
        # written anywhere -- it lives only in this field.
        ttk.Label(self, text="Token:").grid(row=1, column=0, padx=(0, 4), pady=(6, 0))
        self.token = tk.StringVar()
        ttk.Entry(self, textvariable=self.token, width=34, show="*").grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(6, 0))

    @property
    def base_url(self) -> str:
        return f"http://{self.host.get().strip()}:{self.port.get().strip()}"

    def show_checking(self) -> None:
        self._indicator.config(text="● checking...", foreground="gray")

    def show_result(self, reachable: bool) -> None:
        self._indicator.config(text="● connected" if reachable else "● unreachable",
                               foreground="green" if reachable else "red")


class OutputBox(tk.Text):
    """A read-only scrolling text area."""

    def __init__(self, parent, height=12):
        frame = ttk.Frame(parent)
        super().__init__(frame, width=90, height=height, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.yview)
        self.configure(yscrollcommand=scrollbar.set)
        self.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.container = frame

    def write(self, text: str) -> None:
        self.config(state="normal")
        self.delete("1.0", tk.END)
        self.insert(tk.END, text)
        self.config(state="disabled")


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container; put content in `.body`."""

    def __init__(self, parent):
        super().__init__(parent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.body = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>",
                       lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfig(window, width=event.width))

        # Bind the wheel only while the pointer is inside, so other scrollable
        # widgets keep their own wheel behaviour.
        def scroll(event):
            canvas.yview_scroll(-event.delta // 120, "units")

        canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", scroll))
        canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
