"""BuAli's three pipelines, hardcoded (no JSON-instruction interpretation).

Ported from Orchestrator's 02_Radiology_Report_Assist_STT instruction
(steps "transcribe_slot_N" / "reconcile" / "transcribe_and_report_directly")
and its run_instruction() step-loop — same system prompts, same template,
same behavior, just three plain functions instead of a generic loop reading
run_when_stt_mode-tagged steps.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from fastapi import HTTPException

import stt_llm_client as client
from schemas import SttSlotConfig

REPORT_TEMPLATE = json.loads(
    (pathlib.Path(__file__).parent / "report_template.json").read_text(encoding="utf-8")
)

RECONCILE_SYSTEM_PROMPT = (
    "You are a radiology transcription QA assistant. You will be given up to three independent "
    "transcriptions of the SAME spoken radiology report, each produced by a different speech-to-text "
    "engine (local or cloud) -- not all three may be present. They may disagree in places due to "
    "transcription errors. Produce THREE outputs, each strictly grounded in what the transcripts "
    "actually contain -- never invent, assume, or add findings, measurements, sections, or "
    "typical/expected report boilerplate that isn't supported by the transcripts:\n"
    "1. raw_transcript: your best reconstruction of literally what was said, resolving disagreements "
    "between the engines by picking whichever wording is most plausible given radiology context -- do "
    "not clean up phrasing, do not reformat, do not restructure.\n"
    "2. corrected_transcript: the same content as raw_transcript with minimal fixes only -- correct "
    "obvious transcription/terminology mistakes and light punctuation, but keep the same order, "
    "content, and level of completeness. Do not restructure into report sections.\n"
    "3. final_text: an understanding/conclusion of corrected_transcript's content -- it may be "
    "organized or formatted differently, but must ONLY reflect what corrected_transcript actually "
    "says. If the transcript is incomplete, fragmentary, or doesn't cover a full radiology report, "
    "leave those gaps alone -- do NOT fill them in with typical/expected findings. It's fine for "
    "final_text to be short or incomplete if that's what the source supports.\n"
    "Also list the discrepancies you resolved between engines (discrepancies_found) and any relevant "
    "notes. Respond with JSON only, matching the template."
)

TRANSCRIBE_AND_REPORT_SYSTEM_PROMPT = (
    "You are a radiology transcription assistant. You are given an audio recording of a SPOKEN "
    "radiology report -- listen to it directly. You may ALSO be given one or more reference "
    "transcripts of the same audio, produced separately by other speech-to-text engines -- if so, "
    "treat them only as supporting evidence to cross-check against, NOT ground truth (they may "
    "themselves contain transcription errors); your own listening is the primary source of truth, and "
    "where they disagree with what you hear, trust what you hear unless the reference transcript "
    "resolves a genuine ambiguity. Produce THREE outputs, each strictly grounded in what is actually "
    "said in the audio -- never invent, assume, or add findings, measurements, sections, or "
    "typical/expected report boilerplate that isn't supported by the audio (or by the reference "
    "transcripts, if given):\n"
    "1. raw_transcript: a literal, verbatim transcription of the audio (disfluencies/false starts may "
    "be lightly smoothed only where clearly a slip of the tongue) -- no reformatting, no reorganizing.\n"
    "2. corrected_transcript: the same content as raw_transcript with minimal fixes only -- correct "
    "obvious mis-speaking, terminology, and light punctuation, but keep the same order, content, and "
    "level of completeness. Do not restructure into report sections.\n"
    "3. final_text: an understanding/conclusion of corrected_transcript's content -- it may be "
    "organized or formatted differently, but must ONLY reflect what corrected_transcript actually "
    "says. If the audio is incomplete, fragmentary, or doesn't cover a full radiology report, leave "
    "those gaps alone -- do NOT fill them in with typical/expected findings. It's fine for final_text "
    "to be short or incomplete if that's what the audio supports.\n"
    "List any relevant notes (notes; discrepancies_found should be an empty list here unless you're "
    "specifically noting a disagreement with a reference transcript). Respond with JSON only, matching "
    "the template."
)

TRANSCRIPT_LABELS = {
    "transcript_1": "STT engine 1 transcript",
    "transcript_2": "STT engine 2 transcript",
    "transcript_3": "STT engine 3 transcript",
}


def _run_stt_slots(audio: bytes, stt_slots: list[Optional[SttSlotConfig]],
                    language: Optional[str]) -> dict[str, str]:
    """Transcribe every configured slot (unconfigured slots are skipped). Returns
    {"transcript_1": ..., "transcript_2": ..., ...} for whichever slots ran."""
    outputs: dict[str, str] = {}
    for idx, slot in enumerate(stt_slots or []):
        if slot is None:
            continue
        outputs[f"transcript_{idx + 1}"] = client.transcribe_slot(audio, slot, language)
    return outputs


def _reconcile(outputs: dict[str, str], llm_model: str,
               llm_api_key: Optional[str], llm_base_url: Optional[str]) -> dict:
    """The "reconcile" step: fold transcript_N outputs into one LLM call."""
    present = [name for name in TRANSCRIPT_LABELS if name in outputs]
    if not present:
        raise HTTPException(400, "none of this step's inputs were produced: transcript_1..3")
    user_content = "\n\n".join(f"{TRANSCRIPT_LABELS[name]}:\n{outputs[name]}" for name in present)
    messages = [
        {"role": "system",
         "content": RECONCILE_SYSTEM_PROMPT
         + "\n\nJSON template to fill:\n"
         + json.dumps(REPORT_TEMPLATE, ensure_ascii=False, indent=2)},
        {"role": "user", "content": user_content},
    ]
    try:
        reply = client.chat(messages, model=llm_model, response_format={"type": "json_object"},
                            api_key=llm_api_key, base_url=llm_base_url)
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {exc}")
    try:
        result = client.extract_json(reply)
    except ValueError:
        raise HTTPException(502, f"LLM did not return valid JSON:\n{reply}")

    merged = {name: outputs[name] for name in present}
    merged.update(result)
    return merged


def run_separate(audio: bytes, stt_slots: list[Optional[SttSlotConfig]], language: Optional[str],
                  llm_model: str, llm_api_key: Optional[str], llm_base_url: Optional[str]) -> dict:
    """Up to 3 independent STT engines transcribe, then an LLM reconciles."""
    outputs = _run_stt_slots(audio, stt_slots, language)
    return _reconcile(outputs, llm_model, llm_api_key, llm_base_url)


def _transcribe_and_report_directly(audio: bytes, audio_filename: str, reference_outputs: dict[str, str],
                                     llm_model: str, llm_api_key: Optional[str],
                                     llm_base_url: Optional[str]) -> dict:
    audio_format = client.audio_format_from_filename(audio_filename, model=llm_model)
    system_prompt = (TRANSCRIBE_AND_REPORT_SYSTEM_PROMPT
                     + "\n\nJSON template to fill:\n"
                     + json.dumps(REPORT_TEMPLATE, ensure_ascii=False, indent=2))

    reference_transcripts = [reference_outputs[k] for k in sorted(reference_outputs)
                             if k.startswith("transcript_")]
    user_text = None
    if reference_transcripts:
        user_text = "\n\n".join(
            f"Reference transcript {i + 1} (from a separate STT engine — may contain errors):\n{t}"
            for i, t in enumerate(reference_transcripts)
        )

    try:
        reply = client.llm_chat_audio(audio, audio_format, system_prompt, user_text, llm_model,
                                      api_key=llm_api_key, base_url=llm_base_url)
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {exc}")
    try:
        result = client.extract_json(reply)
    except ValueError:
        raise HTTPException(502, f"LLM did not return valid JSON:\n{reply}")

    merged = dict(reference_outputs)
    merged.update(result)
    return merged


def run_multimodal(audio: bytes, audio_filename: str, llm_model: str,
                   llm_api_key: Optional[str], llm_base_url: Optional[str]) -> dict:
    """Audio goes straight to an audio-capable LLM — no STT step at all."""
    return _transcribe_and_report_directly(audio, audio_filename, {}, llm_model, llm_api_key, llm_base_url)


def run_hybrid(audio: bytes, audio_filename: str, stt_slots: list[Optional[SttSlotConfig]],
               language: Optional[str], llm_model: str,
               llm_api_key: Optional[str], llm_base_url: Optional[str]) -> dict:
    """STT slot(s) AND the audio itself both go to the LLM — transcripts as reference material."""
    outputs = _run_stt_slots(audio, stt_slots, language)
    return _transcribe_and_report_directly(audio, audio_filename, outputs, llm_model, llm_api_key, llm_base_url)
