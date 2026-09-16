"""The generated Kaggle fine-tuning notebook -- same reasoning as
`benchmark/tests/test_notebook.py`: no logic of its own, every cell's code
is valid Python, and it is regenerated from `build_notebook.py`, not
hand-edited.
"""
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "finetune" / "notebooks" / "kaggle_finetune.ipynb"
BUILDER = REPO / "finetune" / "notebooks" / "build_notebook.py"


def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def source_of(cell):
    return "\n".join(line for line in "".join(cell["source"]).splitlines()
                     if not line.lstrip().startswith(("!", "%")))


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


class TestTheNotebookIsGenerated:
    def test_the_committed_notebook_matches_the_current_source(self):
        before = NOTEBOOK.read_bytes()
        subprocess.run([sys.executable, str(BUILDER)], check=True,
                       capture_output=True, cwd=str(BUILDER.parent))
        assert NOTEBOOK.read_bytes() == before, "notebook is stale; re-run build_notebook.py"

    def test_every_code_cell_is_valid_python(self, notebook):
        import ast

        for index, cell in enumerate(code_cells(notebook)):
            try:
                ast.parse(source_of(cell))
            except SyntaxError as error:
                pytest.fail(f"code cell {index}: {error}")


class TestItClonesTheRightBranch:
    def test_clones_audio_finetune_not_the_benchmark_branch(self, notebook):
        source = "".join(source_of(c) for c in code_cells(notebook))
        assert "audio-finetune" in source
        assert "benchmark-selfcontained" not in source


class TestItCallsBothScriptsAsSubprocesses:
    """Not imported -- called as `python -m gemma.train_lora` -- so the
    scripts' own `argparse` interface is the only contract, and the notebook
    cannot drift out of sync with an internal function signature the way an
    `import`-based notebook could.
    """

    def test_both_gemma_scripts_are_invoked(self, notebook):
        source = "".join(source_of(c) for c in code_cells(notebook))
        assert "gemma.train_lora" in source
        assert "gemma.evaluate" in source

    def test_dry_run_is_exercised_before_a_real_training_cell(self, notebook):
        sources = [source_of(c) for c in code_cells(notebook)]
        dry_run_index = next(i for i, s in enumerate(sources) if "--dry-run" in s)
        train_index = next(i for i, s in enumerate(sources)
                           if "train_lora" in s and "--dry-run" not in s)
        assert dry_run_index < train_index

    def test_the_hf_token_is_prompted_live_not_hardcoded(self, notebook):
        import re

        source = "".join(source_of(c) for c in code_cells(notebook))
        assert "getpass" in source
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", source), (
            "a literal-looking HF token is present in the generated notebook")
