"""`extract_json` against real, live-observed LLM replies.

Every case here except the control-character one is the shape any JSON
extractor needs to handle (fences, chatter). The control-character case is
the one worth a dedicated test: it is a real failure this repo hit more than
once, across more than one model, and the fix (`strict=False`) is a one-word
diff that a future "tidy up this function" pass could easily revert without
realizing what it re-breaks.
"""
import json

import pytest

import prompts


class TestExtractJSON:
    def test_a_bare_object(self):
        assert prompts.extract_json('{"final_text": "report"}') == {"final_text": "report"}

    def test_fenced_with_a_json_language_tag(self):
        reply = '```json\n{"final_text": "report"}\n```'
        assert prompts.extract_json(reply) == {"final_text": "report"}

    def test_fenced_without_a_language_tag(self):
        reply = '```\n{"final_text": "report"}\n```'
        assert prompts.extract_json(reply) == {"final_text": "report"}

    def test_the_model_explains_itself_first(self):
        reply = 'Here is the JSON response:\n\n```json\n{"final_text": "report"}\n```'
        assert prompts.extract_json(reply) == {"final_text": "report"}

    def test_no_object_in_the_reply_raises(self):
        with pytest.raises(ValueError, match="no JSON object found"):
            prompts.extract_json("I cannot produce a report from this audio.")

    def test_a_literal_tab_inside_a_string_value_does_not_raise(self):
        """The exact live failure: `JSONDecodeError: Invalid control
        character at: line 2 column 57`, from a reply whose raw_transcript
        field contained a literal tab byte instead of an escaped `\\t`."""
        reply = '{"raw_transcript": "abdomen pelvic retroperitoneal\t\tfor the liver"}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(reply)  # the strict default really does reject this
        assert prompts.extract_json(reply) == {
            "raw_transcript": "abdomen pelvic retroperitoneal\t\tfor the liver"}

    def test_a_literal_newline_inside_a_string_value_does_not_raise(self):
        reply = '{"final_text": "line one\nline two"}'
        assert prompts.extract_json(reply) == {"final_text": "line one\nline two"}
