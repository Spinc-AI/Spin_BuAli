"""Turning transcripts into a report, the way the controller does.

This is the stage that makes the benchmark measure the product rather than one
component of it. The reference labels are signed radiology reports, so the
thing to score is a report -- and a report is what comes out of here.

Every prompt is imported from `controller/prompts.py`. None is written here.
A benchmark that phrased the instruction its own way would be ranking models on
a system nobody ships, and the difference would be invisible in the results.

This benchmark runs two of the controller's three pipelines -- `hybrid` is
still a production pipeline in `controller/pipelines.py`, just not part of
this comparison:

    separate    transcripts, no audio    -> RECONCILE
    multimodal  audio, no transcripts    -> TRANSCRIBE_FROM_AUDIO

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


def reconcile_prompt(transcripts: dict[str, str], structure_guide: str | None = None) -> tuple[str, str]:
    """`separate`: several engines' transcripts, no audio.

    Labelled by slot number rather than concatenated, because the prompt asks
    the model to weigh them against each other and it cannot do that if it
    cannot tell where one ends.
    """
    user_text = "\n\n".join(
        f"STT engine {key.removeprefix('transcript_')} transcript:\n{text}"
        for key, text in _in_slot_order(transcripts))
    system = bridge.with_template(bridge.RECONCILE)
    if structure_guide:
        system = f"{system}\n\n{structure_guide}"
    return system, user_text


def audio_prompt(transcripts: dict[str, str], structure_guide: str | None = None) -> tuple[str, str | None]:
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
    system = bridge.with_template(bridge.TRANSCRIBE_FROM_AUDIO)
    if structure_guide:
        system = f"{system}\n\n{structure_guide}"
    return system, user_text


def _in_slot_order(transcripts: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(transcripts.items(),
                  key=lambda item: int(item[0].removeprefix("transcript_")))


def _snippet(reply: str, head: int = 200, tail: int = 200) -> str:
    """Head and tail of a failed reply, for the CSV's `transcription_error`
    column. A truncated-JSON failure looks different from a repetition-loop
    failure at the tail; a refusal looks different at the head. Neither is
    visible from the exception message alone."""
    reply = reply.replace("\n", " ").strip()
    if len(reply) <= head + tail:
        return repr(reply)
    return repr(f"{reply[:head]} ...[{len(reply) - head - tail} chars omitted]... {reply[-tail:]}")


def build_report(asset_id: str, transcripts: dict[str, str], model, pipeline: str,
                 structure_guide: str | None = None, audio_path=None) -> Report:
    """One LLM call, parsed into a report.

    `structure_guide` is a benchmark-only addendum -- never part of
    `controller/prompts.py` -- appended after the JSON template to ask for the
    section order this dataset's reference reports use. It exists because this
    benchmark grades the combined STT+LLM output against a fixed report
    template, not free-form text, so a model that says the same things in a
    different order should not be marked wrong for that alone. See
    `report_structure.py`.

    `audio_path` is the recording itself, only meaningful (and only ever
    passed) for `multimodal` -- `separate` reconciles transcripts and never
    touches the audio, so passing it there would be a caller bug, not
    something to silently accept.

    A failure is recorded on the report rather than raised. One recording that
    the model refused is a data point; it must not cost the other eight.
    """
    report = Report(asset_id=asset_id, transcripts=dict(transcripts))
    started = time.perf_counter()
    try:
        if pipeline == "separate":
            if not transcripts:
                raise ValueError("separate needs at least one transcript to reconcile")
            if audio_path:
                raise ValueError("separate reconciles transcripts, not audio -- "
                                 "audio_path should not be set for this pipeline")
            system_prompt, user_text = reconcile_prompt(transcripts, structure_guide)
        else:
            if not audio_path:
                raise ValueError(f"{pipeline} needs the recording; audio_path was not given")
            system_prompt, user_text = audio_prompt(transcripts, structure_guide)

        reply = model.generate(system_prompt, user_text, audio_path=audio_path)
        parsed = bridge.extract_json(reply)
        report.final_text = parsed.get("final_text") or ""
        report.raw_transcript = parsed.get("raw_transcript") or ""
        report.corrected_transcript = parsed.get("corrected_transcript") or ""
        if not report.final_text:
            report.error = f"the model returned no final_text | reply: {_snippet(reply)}"
    except Exception as error:  # noqa: BLE001 - recorded per recording, never fatal
        # `reply` may not exist if model.generate() itself raised before returning.
        tail = f" | reply: {_snippet(reply)}" if "reply" in locals() else ""
        report.error = f"{type(error).__name__}: {error}{tail}"
    report.elapsed_seconds = time.perf_counter() - started
    return report


def build_reports(transcripts_by_asset: dict[str, dict[str, str]], model, pipeline: str,
                  on_progress=None, structure_guide: str | None = None, items=None) -> list[Report]:
    """The report stage over a whole batch, with the model loaded once.

    Loading dominates: a 30B at 4-bit takes minutes to place and seconds per
    report, so the batch is what the load is amortised over.

    `items` is the dataset entries this batch covers -- `dataset.Item`, each
    with an `.audio` path -- needed only for `separate` != pipeline (i.e.
    `multimodal`), which sends the recording itself rather than a transcript.
    `separate` never reads `items`, so passing `None` there (the default) is
    fine; a non-`separate` pipeline without `items` fails per-recording in
    `build_report`, not silently.
    """
    audio_by_asset = {item.asset_id: item.audio for item in items} if items else {}
    reports = []
    for asset_id, transcripts in transcripts_by_asset.items():
        audio_path = audio_by_asset.get(asset_id) if pipeline != "separate" else None
        report = build_report(asset_id, transcripts, model, pipeline, structure_guide,
                              audio_path=audio_path)
        reports.append(report)
        if on_progress:
            on_progress(report)
    return reports
