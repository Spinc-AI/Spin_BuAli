"""Turning transcripts into a report, the way the controller does.

This is the stage that makes the benchmark measure the product rather than one
component of it. The reference labels are signed radiology reports, so the
thing to score is a report -- and a report is what comes out of here.

Every prompt is imported from `controller/prompts.py`. None is written here.
A benchmark that phrased the instruction its own way would be ranking models on
a system nobody ships, and the difference would be invisible in the results.

The three pipelines differ only in what the single LLM call is given:

    separate    transcripts, no audio    -> RECONCILE
    multimodal  audio, no transcripts    -> TRANSCRIBE_FROM_AUDIO
    hybrid      audio AND transcripts    -> TRANSCRIBE_FROM_AUDIO

which is `controller/pipelines.py`'s structure, kept deliberately parallel so
the two can be read against each other.
"""
import time
from dataclasses import dataclass, field

import bridge


@dataclass
class Report:
    """One recording's finished report, and what it cost."""
    asset_id: str
    final_text: str = ""
    raw_transcript: str = ""
    corrected_transcript: str = ""
    transcripts: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return {"asset_id": self.asset_id, "final_text": self.final_text,
                "raw_transcript": self.raw_transcript,
                "corrected_transcript": self.corrected_transcript,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "error": self.error, **self.transcripts}


def reconcile_prompt(transcripts: dict[str, str]) -> tuple[str, str]:
    """`separate`: several engines' transcripts, no audio.

    Labelled by slot number rather than concatenated, because the prompt asks
    the model to weigh them against each other and it cannot do that if it
    cannot tell where one ends.
    """
    user_text = "\n\n".join(
        f"STT engine {key.removeprefix('transcript_')} transcript:\n{text}"
        for key, text in _in_slot_order(transcripts))
    return bridge.with_template(bridge.RECONCILE), user_text


def audio_prompt(transcripts: dict[str, str]) -> tuple[str, str | None]:
    """`multimodal` / `hybrid`: the model hears the recording.

    Any transcripts go in as cross-check material and are labelled as
    fallible, which is the wording the controller uses -- calling them
    reference material rather than ground truth changes what the model does
    with a disagreement.
    """
    user_text = None
    if transcripts:
        user_text = "\n\n".join(
            f"Reference transcript {position} (from a separate STT engine — "
            f"may contain errors):\n{text}"
            for position, (_, text) in enumerate(_in_slot_order(transcripts), start=1))
    return bridge.with_template(bridge.TRANSCRIBE_FROM_AUDIO), user_text


def _in_slot_order(transcripts: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(transcripts.items(),
                  key=lambda item: int(item[0].removeprefix("transcript_")))


def build_report(asset_id: str, transcripts: dict[str, str], model, pipeline: str) -> Report:
    """One LLM call, parsed into a report.

    A failure is recorded on the report rather than raised. One recording that
    the model refused is a data point; it must not cost the other eight.
    """
    report = Report(asset_id=asset_id, transcripts=dict(transcripts))
    started = time.perf_counter()
    try:
        if pipeline == "separate":
            if not transcripts:
                raise ValueError("separate needs at least one transcript to reconcile")
            system_prompt, user_text = reconcile_prompt(transcripts)
        else:
            system_prompt, user_text = audio_prompt(transcripts)

        reply = model.generate(system_prompt, user_text)
        parsed = bridge.extract_json(reply)
        report.final_text = parsed.get("final_text") or ""
        report.raw_transcript = parsed.get("raw_transcript") or ""
        report.corrected_transcript = parsed.get("corrected_transcript") or ""
        if not report.final_text:
            report.error = "the model returned no final_text"
    except Exception as error:  # noqa: BLE001 - recorded per recording, never fatal
        report.error = f"{type(error).__name__}: {error}"
    report.elapsed_seconds = time.perf_counter() - started
    return report


def build_reports(transcripts_by_asset: dict[str, dict[str, str]], model, pipeline: str,
                  on_progress=None) -> list[Report]:
    """The report stage over a whole batch, with the model loaded once.

    Loading dominates: a 30B at 4-bit takes minutes to place and seconds per
    report, so the batch is what the load is amortised over.
    """
    reports = []
    for asset_id, transcripts in transcripts_by_asset.items():
        report = build_report(asset_id, transcripts, model, pipeline)
        reports.append(report)
        if on_progress:
            on_progress(report)
    return reports
