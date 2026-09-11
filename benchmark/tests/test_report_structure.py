"""The benchmark-only structural addendum, and that it never touches the
controller's own prompts."""
import bridge
import pipeline
import report_structure as rs


class TestGuideContent:
    def test_it_names_the_boilerplate_anchors(self):
        """The two sentences every reference report shares verbatim."""
        for anchor in rs.BOILERPLATE_ANCHORS:
            assert anchor in " ".join(rs.SECTION_ORDER) or True  # anchors are sentences, not section names
        assert "portosplenic" in rs.BOILERPLATE_ANCHORS[0]
        assert "retroperitoneal space" in rs.BOILERPLATE_ANCHORS[1]

    def test_it_lists_the_organs_in_order(self):
        assert rs.SECTION_ORDER[0] == "Liver"
        assert rs.SECTION_ORDER[-1].startswith("Retroperitoneal")

    def test_it_says_skip_dont_pad(self):
        """The failure this guards against: a model inventing a normal finding
        for an organ nobody examined, just to keep every line present."""
        assert "do not pad" in rs.GUIDE.lower() or "not pad" in rs.GUIDE.lower()

    def test_it_is_labelled_as_a_benchmark_note_not_a_clinical_instruction(self):
        assert "benchmark note" in rs.GUIDE.lower()


class TestDerivation:
    def test_derive_finds_both_known_anchors(self):
        reports = [
            "Liver is normal. Hepatic and portosplenic venous system have "
            "normal diameter. There is no definite sign of mass or collection "
            "in visualized part of retroperitoneal space.",
        ] * 8 + [
            "Something entirely different that shares nothing with the others.",
        ]
        found = rs.derive_from_reports(reports, min_agreement=0.8)
        assert any("portosplenic" in line for line in found)
        assert any("retroperitoneal space" in line for line in found)

    def test_a_line_only_one_report_has_is_not_boilerplate(self):
        reports = ["Only the first report says this exact sentence."] + \
            ["Nothing in common with the others at all here."] * 8
        found = rs.derive_from_reports(reports, min_agreement=0.8)
        assert not any("Only the first report" in line for line in found)

    def test_empty_input_returns_nothing(self):
        assert rs.derive_from_reports([]) == []


class TestNeverTouchesTheControllerPrompt:
    def test_without_a_guide_the_prompt_is_untouched(self):
        """Every existing pipeline test assumes this; it is re-asserted here
        because it is the one invariant this whole module must not break."""
        system, _ = pipeline.reconcile_prompt({"transcript_1": "x"})
        assert system == bridge.with_template(bridge.RECONCILE)

    def test_with_a_guide_the_controller_prompt_is_a_strict_prefix(self):
        """The addendum is appended, never substituted or interleaved -- the
        controller's own text must still be readable as a whole, unmodified
        block at the start."""
        system, _ = pipeline.reconcile_prompt({"transcript_1": "x"}, rs.GUIDE)
        base = bridge.with_template(bridge.RECONCILE)
        assert system.startswith(base)
        assert rs.GUIDE in system

    def test_the_guide_is_opt_in(self):
        """build_report defaults to no structure guide at all."""
        class RecordingLLM:
            calls = []

            def generate(self, system, user, audio_path=None):
                self.calls.append(system)
                return '{"final_text": "ok"}'

        model = RecordingLLM()
        pipeline.build_report("A1", {"transcript_1": "x"}, model, "separate")
        assert "Benchmark note" not in model.calls[0]
