"""Pipeline behaviour: which inputs reach the LLM, and how results merge."""
import pytest
from fastapi import HTTPException

import pipelines
import prompts
from schemas import LlmTarget, Pipeline, SttSlotConfig

LLM = LlmTarget(model="gemini:gemini-2.5-pro")
AUDIO = b"fake-audio-bytes"


def slots(*models):
    """Slot list from model names; None leaves that slot unconfigured."""
    return [SttSlotConfig(model=m) if m else None for m in models]


class TestSeparate:
    def test_reconciles_transcripts_without_audio(self, services_up):
        result = pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav",
                               slots("whisper", "openai:whisper-1"), "fa", LLM)

        call = services_up["complete"][0]
        assert call["audio"] is None, "separate must not send audio to the LLM"
        assert call["system_prompt"].startswith(prompts.RECONCILE)
        assert "STT engine 1 transcript" in call["user_text"]
        assert "STT engine 2 transcript" in call["user_text"]
        assert result["final_text"] == "report"

    def test_raw_transcripts_are_kept_alongside_the_report(self, services_up):
        result = pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots("whisper"), None, LLM)
        assert result["transcript_1"] == "transcript from whisper"
        assert result["final_text"] == "report"

    def test_slot_numbering_follows_position_not_order_run(self, services_up):
        result = pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav",
                               slots(None, "whisper"), None, LLM)
        assert "transcript_2" in result and "transcript_1" not in result
        assert "STT engine 2 transcript" in services_up["complete"][0]["user_text"]

    def test_no_transcripts_is_a_client_error(self, services_up):
        with pytest.raises(HTTPException) as exc:
            pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots(None), None, LLM)
        assert exc.value.status_code == 400

    def test_slot_failure_names_the_slot(self, services_up, monkeypatch):
        import stt_client

        def boom(audio, slot, language):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(stt_client, "transcribe", boom)
        with pytest.raises(HTTPException) as exc:
            pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots("whisper"), None, LLM)
        assert exc.value.status_code == 502
        assert "slot 1" in exc.value.detail and "whisper" in exc.value.detail


class TestMultimodal:
    def test_sends_audio_and_skips_stt(self, services_up):
        pipelines.run(Pipeline.MULTIMODAL, AUDIO, "report.wav", slots("whisper"), "fa", LLM)

        assert services_up["transcribe"] == [], "multimodal must not run STT"
        call = services_up["complete"][0]
        assert call["audio"] == AUDIO
        assert call["audio_format"] == "wav"
        assert call["user_text"] is None, "no reference transcripts to pass"
        assert call["system_prompt"].startswith(prompts.TRANSCRIBE_FROM_AUDIO)

    def test_rejects_a_container_the_provider_cannot_take(self, services_up):
        with pytest.raises(HTTPException) as exc:
            pipelines.run(Pipeline.MULTIMODAL, AUDIO, "report.flac", [], None,
                          LlmTarget(model="openai:gpt-4o-audio"))
        assert exc.value.status_code == 400


class TestHybrid:
    def test_sends_both_audio_and_transcripts(self, services_up):
        result = pipelines.run(Pipeline.HYBRID, AUDIO, "report.mp3",
                               slots("whisper"), "fa", LLM)

        call = services_up["complete"][0]
        assert call["audio"] == AUDIO and call["audio_format"] == "mp3"
        assert "Reference transcript 1" in call["user_text"]
        assert "may contain errors" in call["user_text"]
        assert result["transcript_1"] == "transcript from whisper"

    def test_is_multimodal_when_no_slots_are_configured(self, services_up):
        pipelines.run(Pipeline.HYBRID, AUDIO, "report.wav", [], None, LLM)
        assert services_up["complete"][0]["user_text"] is None

    def test_references_are_numbered_contiguously(self, services_up):
        """A gap in the slots must not leave a gap in the reference numbering."""
        pipelines.run(Pipeline.HYBRID, AUDIO, "report.wav",
                      slots(None, "whisper", "openai:whisper-1"), None, LLM)
        user_text = services_up["complete"][0]["user_text"]
        assert "Reference transcript 1" in user_text
        assert "Reference transcript 2" in user_text
        assert "Reference transcript 3" not in user_text


class TestLlmContract:
    def test_template_is_appended_to_every_prompt(self, services_up):
        pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots("whisper"), None, LLM)
        assert "JSON template to fill:" in services_up["complete"][0]["system_prompt"]
        assert "corrected_transcript" in services_up["complete"][0]["system_prompt"]

    def test_credentials_are_forwarded(self, services_up):
        target = LlmTarget(model="openai:gpt-4o", api_key="sk-x", base_url="https://proxy/v1")
        pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots("whisper"), None, target)
        call = services_up["complete"][0]
        assert (call["api_key"], call["base_url"]) == ("sk-x", "https://proxy/v1")

    def test_unparseable_reply_is_a_bad_gateway(self, services_up, monkeypatch):
        import llm_client

        monkeypatch.setattr(llm_client, "complete",
                            lambda *a, **k: "sorry, I can't do that")
        with pytest.raises(HTTPException) as exc:
            pipelines.run(Pipeline.SEPARATE, AUDIO, "r.wav", slots("whisper"), None, LLM)
        assert exc.value.status_code == 502
        assert "did not return valid JSON" in exc.value.detail
