"""The notebook is now a flat rewrite, not the repo's modules.

That is what makes it readable — every function is defined in a cell you can
edit and re-run — and it is also the risk: two implementations of the same
metrics can drift apart, and nothing about a green test suite would say so.

So the important test here does not check the notebook's structure. It runs
the notebook's own scoring cells in a clean process and checks that they give
the same numbers as `evaluation/`. If someone changes one and not the other,
this fails.
"""
import ast
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "benchmark" / "notebooks" / "kaggle_dual_t4.ipynb"
BUILDER = REPO / "benchmark" / "notebooks" / "build_notebook.py"

# Cells that need a GPU, the dataset, or the network, and so cannot run here.
UNRUNNABLE = ("!pip", "find_labels", "load_audio", "torch.cuda.get_device")

# Report pairs the two implementations must agree on. Chosen for the failure
# each one represents, not for coverage.
CASES = [
    ("identical",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis.",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("side flipped",
     "There is a 6 mm stone in the distal left ureter. No hydronephrosis.",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("negation dropped",
     "There is a 6 mm stone in the distal right ureter. Hydronephrosis is present.",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("number wrong",
     "There is a 7 mm stone in the distal right ureter. No hydronephrosis.",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("unit wrong",
     "There is a 6 cm stone in the distal right ureter. No hydronephrosis.",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("empty",
     "",
     "There is a 6 mm stone in the distal right ureter. No hydronephrosis."),
    ("looping",
     "The liver is normal. " * 40,
     "The liver is normal."),
]

# Fields both implementations compute, under the same name.
SHARED = ["wer", "cer", "medical_term_f1", "negation_errors", "laterality_errors",
          "number_errors", "unit_errors", "critical_omissions",
          "unsupported_additions", "requires_medical_review"]


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def strip_magics(source):
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith(("!", "%")))


def scoring_source(notebook):
    """Every cell up to and including the scorer, with the unrunnable ones out.

    This is the notebook's metric implementation, lifted whole -- so what the
    comparison below scores really is what a Kaggle session would run.
    """
    collected = []
    for cell in code_cells(notebook):
        source = "".join(cell["source"])
        if any(marker in source for marker in UNRUNNABLE):
            continue
        collected.append(strip_magics(source))
        if "def score_report" in source:
            break
    return "\n\n".join(collected)


HEADER = (
    "import gc, hashlib, io, json, os, re, time, unicodedata, warnings\n"
    "from collections import Counter\n"
    "from dataclasses import dataclass, field\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "warnings.filterwarnings('ignore')\n"
)


def _run(driver_source, name):
    """Run a generated script and read back the scores it wrote.

    Through a file, not stdout: the notebook's cells print demonstrations of
    their own output, which is right for a notebook and useless for a pipe.
    """
    workspace = pathlib.Path(tempfile.mkdtemp())
    output = workspace / "scores.json"
    script = workspace / name
    script.write_text(driver_source(output), encoding="utf-8")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, encoding="utf-8", env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(output.read_text(encoding="utf-8"))


def run_notebook_scorer(cases):
    """Score `cases` with the notebook's own code, in a clean process."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def driver(output):
        return (
            HEADER + scoring_source(notebook) + "\n\n"
            "import json as _json, pathlib as _pathlib\n"
            f"_cases = {cases!r}\n"
            "_out = [score_report(h, r) for _, h, r in _cases]\n"
            f"_pathlib.Path({str(output)!r}).write_text("
            "_json.dumps(_out), encoding='utf-8')\n"
        )

    return _run(driver, "notebook_cells.py")


def run_repo_scorer(cases):
    """The same cases through `evaluation/`."""
    def driver(output):
        return (
            "import json, pathlib, sys\n"
            f"sys.path.insert(0, {str(REPO / 'evaluation')!r})\n"
            "from extractors import ClinicalTerms\n"
            "from medical_metrics import evaluate\n"
            "terms = ClinicalTerms()\n"
            f"cases = {cases!r}\n"
            "out = []\n"
            "for _, hypothesis, reference in cases:\n"
            "    r = evaluate(hypothesis, reference, terms)\n"
            "    flat = {**r['general'], **r['clinical_counts'], **r['clinical_metrics']}\n"
            "    flat['requires_medical_review'] = r['requires_medical_review']\n"
            "    out.append(flat)\n"
            f"pathlib.Path({str(output)!r}).write_text("
            "json.dumps(out), encoding='utf-8')\n"
        )

    return _run(driver, "repo_scorer.py")


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def both_scorers():
    return run_notebook_scorer(CASES), run_repo_scorer(CASES)


class TestTheNotebookIsGenerated:
    def test_the_committed_notebook_matches_the_generator(self):
        """Run `python benchmark/notebooks/build_notebook.py` after editing it."""
        before = NOTEBOOK.read_bytes()
        subprocess.run([sys.executable, str(BUILDER)], check=True,
                       capture_output=True, cwd=str(BUILDER.parent))
        assert NOTEBOOK.read_bytes() == before, "notebook is stale; re-run build_notebook.py"

    def test_every_code_cell_is_valid_python(self, notebook):
        """None of these would fail until the cell ran on Kaggle, hours in."""
        for index, cell in enumerate(code_cells(notebook)):
            try:
                ast.parse(strip_magics("".join(cell["source"])))
            except SyntaxError as error:
                pytest.fail(f"code cell {index}: line {error.lineno}: {error.msg}")

    def test_it_defines_its_code_rather_than_writing_files(self, notebook):
        """The point of the rewrite: cells contain code, not strings of code."""
        source = "".join("".join(c["source"]) for c in code_cells(notebook))
        assert "write(\"benchmark/" not in source
        assert "def score_report" in source, "the scorer is defined in a cell"
        assert "class WhisperSTT" in source, "the models are defined in cells"

    def test_it_needs_nothing_from_this_repo(self, notebook):
        source = "".join("".join(c["source"]) for c in code_cells(notebook))
        for forbidden in ("import bridge", "import settings", "from schemas",
                          "import evaluate_results"):
            assert forbidden not in source, f"notebook still imports {forbidden}"


class TestItAgreesWithTheEvaluationService:
    """The guard that matters. Two implementations, one set of numbers."""

    def test_the_shared_metrics_match(self, both_scorers):
        notebook_scores, repo_scores = both_scorers
        mismatches = []
        for (label, _, _), mine, theirs in zip(CASES, notebook_scores, repo_scores):
            for field in SHARED:
                if field not in theirs:
                    continue
                a, b = mine.get(field), theirs.get(field)
                if isinstance(a, float) or isinstance(b, float):
                    if abs((a or 0) - (b or 0)) > 0.02:
                        mismatches.append(f"{label}.{field}: notebook={a} evaluation={b}")
                elif a != b:
                    mismatches.append(f"{label}.{field}: notebook={a} evaluation={b}")
        assert not mismatches, "the notebook and evaluation/ disagree:\n  " + \
            "\n  ".join(mismatches)

    def test_a_flipped_side_is_caught_by_both(self, both_scorers):
        """The error WER shrugs at. If either implementation stops catching it,
        the benchmark is ranking on the wrong thing."""
        notebook_scores, repo_scores = both_scorers
        index = [c[0] for c in CASES].index("side flipped")
        assert notebook_scores[index]["laterality_errors"] > 0
        assert repo_scores[index]["laterality_errors"] > 0
        assert notebook_scores[index]["wer"] < 0.15, "WER barely moves — that is the point"

    def test_a_dropped_negation_is_caught_by_both(self, both_scorers):
        notebook_scores, repo_scores = both_scorers
        index = [c[0] for c in CASES].index("negation dropped")
        assert notebook_scores[index]["negation_errors"] > 0
        assert repo_scores[index]["negation_errors"] > 0

    def test_a_looping_model_is_caught_by_both(self, both_scorers):
        """No clinical counter fires on a repeated plausible sentence."""
        notebook_scores, repo_scores = both_scorers
        index = [c[0] for c in CASES].index("looping")
        assert notebook_scores[index]["requires_medical_review"] is True
        assert repo_scores[index]["requires_medical_review"] is True

    def test_a_perfect_report_scores_perfectly_in_both(self, both_scorers):
        notebook_scores, repo_scores = both_scorers
        index = [c[0] for c in CASES].index("identical")
        assert notebook_scores[index]["wer"] == 0.0 == repo_scores[index]["wer"]
        assert notebook_scores[index]["medical_term_f1"] == 1.0


class TestTheNotebookRuns:
    def test_the_scoring_cells_execute_end_to_end(self):
        """Not just parseable — actually runnable, in a clean interpreter."""
        scores = run_notebook_scorer([CASES[0]])
        assert scores[0]["wer"] == 0.0

    def test_the_vocabulary_loads_in_the_notebook(self, notebook):
        source = "".join("".join(c["source"]) for c in code_cells(notebook))
        assert "CONCEPTS = [" in source and "TERM_INDEX" in source
