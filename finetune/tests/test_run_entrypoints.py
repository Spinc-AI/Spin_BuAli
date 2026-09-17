"""`run.py` and `evaluate.py` -- the plain-Python entry points, no notebook
and no `python -m` module-path syntax required. Both are thin proxies to
`gemma/train_lora.py`'s and `gemma/evaluate.py`'s own `main()`; what this
file actually checks is that the proxy really is thin (no re-implemented
logic to drift out of sync) and that it works when invoked from a directory
other than `finetune/` itself, which is the whole point of a top-level
entry point a user might run from the repo root.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FINETUNE_DIR = REPO / "finetune"
RUN_PY = FINETUNE_DIR / "run.py"
EVALUATE_PY = FINETUNE_DIR / "evaluate.py"


class TestTheyAreThinProxiesNotReimplementations:
    def test_run_py_only_calls_gemma_train_loras_main(self):
        source = RUN_PY.read_text(encoding="utf-8")
        assert "from gemma.train_lora import main" in source
        assert "def main(" not in source  # no second implementation to drift

    def test_evaluate_py_only_calls_gemma_evaluates_main(self):
        source = EVALUATE_PY.read_text(encoding="utf-8")
        assert "from gemma.evaluate import main" in source
        assert "def main(" not in source


class TestTheyRunFromAnyDirectory:
    """The reason a plain top-level entry point is worth having at all --
    `python -m gemma.train_lora` only works from inside `finetune/`, and a
    user reaching for "the normal way to run this" from the repo root
    should not have to know that.
    """

    def test_run_py_reaches_the_same_argument_parser_from_the_repo_root(self):
        result = subprocess.run(
            [sys.executable, str(RUN_PY), "--help"],
            cwd=str(REPO), capture_output=True, text=True)
        assert result.returncode == 0
        assert "--labels-csv" in result.stdout
        assert "--dry-run" in result.stdout

    def test_evaluate_py_reaches_the_same_argument_parser_from_the_repo_root(self):
        result = subprocess.run(
            [sys.executable, str(EVALUATE_PY), "--help"],
            cwd=str(REPO), capture_output=True, text=True)
        assert result.returncode == 0
        assert "--labels-csv" in result.stdout
        assert "--adapter-dir" in result.stdout

    def test_run_py_without_labels_csv_fails_the_same_way_the_module_does(self):
        result = subprocess.run(
            [sys.executable, str(RUN_PY)], cwd=str(REPO), capture_output=True, text=True)
        assert result.returncode != 0
        assert "--labels-csv" in result.stderr
