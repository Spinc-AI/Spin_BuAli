"""`gemma/collator.py`'s pure-Python parts -- `build_target` needs neither
`torch` nor a real processor, so it gets a real test instead of a source-text
pin. `GemmaChatCollator` itself needs both and is exercised only as far as
`test_cli.py`'s import (proving the module loads) and by hand on real
hardware -- see `finetune/README.md`.
"""
import json

from gemma.collator import build_target


class TestBuildTarget:
    def test_every_report_template_key_is_present(self):
        target = build_target("some report text")
        assert set(target) == {
            "raw_transcript", "corrected_transcript", "final_text",
            "discrepancies_found", "notes"}

    def test_the_three_transcript_fields_all_echo_the_reference(self):
        target = build_target("Liver has normal size.")
        assert target["raw_transcript"] == "Liver has normal size."
        assert target["corrected_transcript"] == "Liver has normal size."
        assert target["final_text"] == "Liver has normal size."

    def test_discrepancies_found_is_empty_and_notes_is_null(self):
        target = build_target("anything")
        assert target["discrepancies_found"] == []
        assert target["notes"] is None

    def test_the_result_is_valid_json_serializable(self):
        target = build_target("Report with \"quotes\" and a newline\nhere.")
        # round-trips without raising -- the collator's next step is exactly
        # json.dumps(build_target(...))
        assert json.loads(json.dumps(target, ensure_ascii=False)) == target
