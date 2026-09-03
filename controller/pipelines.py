"""The three pipelines that turn a recording into a radiology report.

All three end in the same place -- one LLM call that fills in
prompts.REPORT_TEMPLATE -- and differ only in what that call is given:

    separate    transcripts, no audio    -> prompts.RECONCILE
    multimodal  audio, no transcripts    -> prompts.TRANSCRIBE_FROM_AUDIO
    hybrid      audio AND transcripts    -> prompts.TRANSCRIBE_FROM_AUDIO

`multimodal` is therefore just `hybrid` with no STT slots configured, and the
two share one code path below rather than having one function each.
"""
from fastapi import HTTPException

import llm_client
import prompts
import providers
import stt_client
from schemas import LlmTarget, Pipeline, RuntimeSttSlotConfig


def run(pipeline: Pipeline, audio: bytes, filename: str,
        slots: list[RuntimeSttSlotConfig | None], language: str | None,
        llm: LlmTarget) -> dict:
    """Run `pipeline` over one recording and return the filled-in report.

    The result is the LLM's JSON output, with each STT slot's raw transcript
    merged in alongside it under `transcript_1`..`transcript_N`.
    """
    transcripts = (_transcribe_slots(audio, filename, slots, language)
                   if pipeline.uses_stt else {})
    if pipeline.uses_audio_llm:
        return _report_from_audio(audio, filename, transcripts, llm)
    return _reconcile(transcripts, llm)


def _transcribe_slots(audio: bytes, filename: str,
                      slots: list[RuntimeSttSlotConfig | None],
                      language: str | None) -> dict[str, str]:
    """Transcribe with every configured slot, skipping the unused ones.

    Keys are numbered by slot position, so `transcript_2` always means the
    second slot even when the first is left empty.
    """
    transcripts: dict[str, str] = {}
    for position, slot in enumerate(slots or [], start=1):
        if slot is None:
            continue
        try:
            transcripts[f"transcript_{position}"] = stt_client.transcribe(
                audio, slot, language, filename)
        except Exception as exc:
            raise HTTPException(502, f"STT slot {position} ('{slot.model}') failed: {exc}")
    return transcripts


def _reconcile(transcripts: dict[str, str], llm: LlmTarget) -> dict:
    """`separate`: fold the slots' transcripts into one report, no audio."""
    if not transcripts:
        raise HTTPException(400, "no STT slot produced a transcript to reconcile")
    user_text = "\n\n".join(
        f"STT engine {key.removeprefix('transcript_')} transcript:\n{text}"
        for key, text in _in_slot_order(transcripts)
    )
    return {**transcripts, **_complete(prompts.RECONCILE, user_text, llm)}


def _report_from_audio(audio: bytes, filename: str, transcripts: dict[str, str],
                       llm: LlmTarget) -> dict:
    """`multimodal` / `hybrid`: the LLM hears the recording itself, with any
    transcripts passed alongside as cross-check material."""
    try:
        audio_format = providers.audio_format(filename, llm.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    user_text = None
    if transcripts:
        user_text = "\n\n".join(
            f"Reference transcript {position} (from a separate STT engine — "
            f"may contain errors):\n{text}"
            for position, (_, text) in enumerate(_in_slot_order(transcripts), start=1)
        )

    result = _complete(prompts.TRANSCRIBE_FROM_AUDIO, user_text, llm,
                       audio=audio, audio_format=audio_format)
    return {**transcripts, **result}


def _complete(system_prompt: str, user_text: str | None, llm: LlmTarget,
              audio: bytes | None = None, audio_format: str | None = None) -> dict:
    """One LLM call, parsed into the report JSON."""
    try:
        reply = llm_client.complete(
            prompts.with_template(system_prompt), user_text, model=llm.model,
            api_key=llm.api_key, base_url=llm.base_url,
            audio=audio, audio_format=audio_format,
        )
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {exc}")
    try:
        return llm_client.extract_json(reply)
    except ValueError:
        raise HTTPException(502, f"LLM did not return valid JSON:\n{reply}")


def _in_slot_order(transcripts: dict[str, str]) -> list[tuple[str, str]]:
    """The transcripts sorted by slot number rather than by string key."""
    return sorted(transcripts.items(),
                  key=lambda item: int(item[0].removeprefix("transcript_")))
