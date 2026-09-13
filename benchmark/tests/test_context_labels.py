"""context_labels.context_line -- the one line of known modality/region
context that reaches the LLM alongside the transcript or audio."""
import context_labels


class TestContextLine:
    def test_nothing_known_returns_none(self):
        assert context_labels.context_line(None, ()) is None
        assert context_labels.context_line(None, None) is None

    def test_modality_only(self):
        line = context_labels.context_line("Ultrasound", None)
        assert "Modality: Ultrasound." in line
        assert "Region" not in line

    def test_regions_only(self):
        line = context_labels.context_line(None, ("Abdomen", "Pelvis"))
        assert "Region(s) examined: Abdomen, Pelvis." in line
        assert "Modality" not in line

    def test_both(self):
        line = context_labels.context_line("Ultrasound", ("Abdomen", "Pelvis", "Retroperitoneum"))
        assert "Modality: Ultrasound." in line
        assert "Region(s) examined: Abdomen, Pelvis, Retroperitoneum." in line

    def test_it_is_framed_as_context_not_instruction(self):
        """Same reasoning as report_structure.GUIDE: known metadata about the
        recording, not a clinical instruction the model must obey verbatim."""
        line = context_labels.context_line("Ultrasound", ("Abdomen",))
        assert "not inferred" in line
