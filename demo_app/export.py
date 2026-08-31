"""Saving a finished report as a Word document.

Persian reports need per-paragraph right-to-left marking, which python-docx
doesn't expose directly -- the `w:bidi` / `w:rtl` elements below are the
underlying OOXML for it.
"""
import os
import re
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement

# Arabic-script Unicode blocks: base, presentation forms A and B.
_RTL_RANGES = ((0x0590, 0x08FF), (0xFB1D, 0xFDFF), (0xFE70, 0xFEFF))
_UNSAFE_IN_FILENAME = re.compile(r'[\\/:*?"<>|]')


def looks_rtl(text: str) -> bool:
    """True if the line contains any Arabic-script character."""
    return any(low <= ord(char) <= high
               for char in text or "" for low, high in _RTL_RANGES)


def sanitize_filename_part(name: str) -> str:
    """A model name reduced to something safe to put in a filename."""
    if not name:
        return ""
    _, _, without_prefix = name.rpartition(":")
    return _UNSAFE_IN_FILENAME.sub("-", without_prefix).strip("-_")


def build_filename(audio_path: str | None, model_parts, extension=".docx") -> str:
    """`<recording>_<model>.docx`, falling back to `report` with no recording."""
    stem = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else "report"
    parts = [stem] + [sanitize_filename_part(part) for part in model_parts if part]
    return "_".join(part for part in parts if part) + extension


def save_as_docx(text: str, path: str) -> None:
    """Write `text` to `path`, marking Arabic-script paragraphs right-to-left."""
    document = Document()
    for line in (text or "").split("\n"):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(line)
        if looks_rtl(line):
            paragraph._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
            run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    document.save(path)


class TranscriptSaver(ttk.Frame):
    """Destination picker plus a save button.

    Defaults to saving next to the recording, until the user picks a folder.
    """

    def __init__(self, parent, get_report):
        super().__init__(parent)
        self._get_report = get_report
        self._chosen_by_user = False
        self.save_dir = tk.StringVar()

        ttk.Label(self, text="Save to:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.save_dir, width=40, state="readonly").grid(
            row=0, column=1, sticky="w")
        ttk.Button(self, text="Browse...", command=self._browse).grid(
            row=0, column=2, sticky="w", padx=4)
        ttk.Button(self, text="Save Transcript (.docx)", command=self._save).grid(
            row=0, column=3, sticky="w", padx=(8, 0))

    def suggest_dir_from(self, audio_path: str) -> None:
        if audio_path and not self._chosen_by_user:
            self.save_dir.set(os.path.dirname(os.path.abspath(audio_path)))

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(title="Choose where to save the transcript")
        if chosen:
            self.save_dir.set(chosen)
            self._chosen_by_user = True

    def _save(self) -> None:
        text, audio_path, model_parts = self._get_report()
        if not (text or "").strip():
            messagebox.showwarning("Save Transcript", "Nothing to save yet — run BuAli first.")
            return

        directory = self.save_dir.get().strip() or (
            os.path.dirname(os.path.abspath(audio_path)) if audio_path else os.getcwd())
        path = os.path.join(directory, build_filename(audio_path, model_parts))
        try:
            os.makedirs(directory, exist_ok=True)
            save_as_docx(text, path)
        except OSError as exc:
            messagebox.showerror("Save Transcript", f"Could not save: {exc}")
            return
        messagebox.showinfo("Save Transcript", f"Saved to:\n{path}")
