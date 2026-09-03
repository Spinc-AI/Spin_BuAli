"""The notebook is a build artifact, and two things can go wrong with one: it
drifts from the source it was generated from, or it turns out not to be
self-contained after all.

The second is the one worth testing hardest. These reconstruct the module tree
from the notebook's own cells in a temporary directory, with this repo kept off
`sys.path`, and run a benchmark there -- which is what Kaggle does.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "benchmark" / "notebooks" / "kaggle_dual_t4.ipynb"
BUILDER = REPO / "benchmark" / "notebooks" / "build_notebook.py"

# Cells that cannot run off Kaggle: shell installs, and the GPU assertion.
SKIP_MARKERS = ("Accelerator to GPU T4",)

# Stop once every module is on disk. The cell after this one lists the model
# registry, which imports torch -- always present on Kaggle, not necessarily
# here, and not needed by the dry-run path these tests exercise.
STOP_MARKER = 'write("benchmark/run_benchmark.py"'

HAS_TORCH = importlib.util.find_spec("torch") is not None


def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def setup_source(notebook):
    """The writer cell plus every embedded module -- the whole of section 2."""
    collected = []
    for cell in code_cells(notebook):
        source = "".join(cell["source"])
        if any(marker in source for marker in SKIP_MARKERS):
            continue
        source = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith(("!", "%")))
        if not source.strip():
            continue
        collected.append(source)
        if STOP_MARKER in source:
            break
    return "\n\n".join(collected)


def run_driver(tmp_path, notebook, tail):
    """Execute the notebook's section 2, then `tail`, in an isolated process."""
    driver = tmp_path / "driver.py"
    driver.write_text(setup_source(notebook) + "\n\n" + tail, encoding="utf-8")
    # cwd is the temp dir and PYTHONPATH is stripped, so sys.path[0] is the temp
    # dir and nothing from this repo is importable.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(driver)], cwd=str(tmp_path),
                          capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


class TestItIsGeneratedFromTheSource:
    def test_the_committed_notebook_matches_the_current_source(self):
        """If this fails, a module changed and the notebook was not rebuilt --
        run `python benchmark/notebooks/build_notebook.py`. It is the whole
        safeguard against the notebook and the tested code drifting apart."""
        before = NOTEBOOK.read_bytes()
        subprocess.run([sys.executable, str(BUILDER)], check=True,
                       capture_output=True, cwd=str(BUILDER.parent))
        assert NOTEBOOK.read_bytes() == before, "notebook is stale; re-run build_notebook.py"

    def test_every_source_file_is_embedded(self, notebook):
        """The embedding is verbatim, so the code that runs on Kaggle is
        character-for-character the code the rest of this suite covers."""
        spec = importlib.util.spec_from_file_location("builder", BUILDER)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)

        embedded = "".join("".join(cell["source"]) for cell in code_cells(notebook))
        for source_path, destination, _ in builder.SOURCES:
            text = (REPO / source_path).read_text(encoding="utf-8")
            assert text in embedded, f"{source_path} is not embedded verbatim"
            assert f'write("{destination}"' in embedded

    def test_every_code_cell_is_valid_python(self, notebook):
        import ast

        for index, cell in enumerate(code_cells(notebook)):
            source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                               if not line.lstrip().startswith(("!", "%")))
            try:
                ast.parse(source)
            except SyntaxError as error:
                pytest.fail(f"code cell {index}: {error}")


class TestItIsActuallySelfContained:
    def test_it_reconstructs_and_runs_with_the_repo_off_the_path(self, notebook, tmp_path):
        """The claim the whole design rests on."""
        result = run_driver(tmp_path, notebook, RUN_A_BENCHMARK)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SELF-CONTAINED OK" in result.stdout, result.stdout

    def test_the_reconstructed_tree_mirrors_the_repo(self, notebook, tmp_path):
        """The layout is not arbitrary: each module keeps its own imports only
        because `evaluation/` and `stt/` land beside `benchmark/`."""
        result = run_driver(tmp_path, notebook, LIST_THE_TREE)
        assert result.returncode == 0, result.stdout + result.stderr
        for folder in ("benchmark", "evaluation", "stt"):
            assert folder in result.stdout, result.stdout

    @pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed here")
    def test_the_model_registry_loads_from_the_reconstructed_tree(self, notebook, tmp_path):
        """The stt half, which the dry-run path never touches."""
        result = run_driver(tmp_path, notebook, LOAD_THE_REGISTRY)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "REGISTRY OK" in result.stdout


RUN_A_BENCHMARK = '''
import dataset, leaderboard, run_benchmark, transcribe
import numpy as np, soundfile as sf

audio_dir, truth_dir = SRC.parent / "audio", SRC.parent / "truth"
audio_dir.mkdir(parents=True, exist_ok=True)
truth_dir.mkdir(parents=True, exist_ok=True)
for name, seconds in (("A1", 4.0), ("A2", 70.0)):
    sf.write(str(audio_dir / (name + ".wav")),
             np.zeros(int(seconds * 16000), np.float32), 16000)
    (truth_dir / (name + ".txt")).write_text(
        "There is a 6 mm stone in the distal right ureter. No hydronephrosis.",
        encoding="utf-8")

items = dataset.from_directory(audio_dir, truth_dir)
runs, results, summary = run_benchmark.run(
    items, models=["dry-run"], model_factory=transcribe.dry_run_factory)

bucket = summary["models"][0]
assert len(results) == 2, results
assert "corpus_wer" in bucket["corpus"]
assert bucket["by_duration"], "duration buckets missing"
assert "script_contamination_mean" in bucket["text"]

out = SRC.parent / "out"
leaderboard.write(out, results, summary, runs)
written = sorted(p.name for p in out.iterdir())
assert "leaderboard.csv" in written and "per_report.csv" in written, written

print("windows for the 70s clip:",
      max(t.windows for run in runs for t in run.transcripts))
print("files written:", written)
print("SELF-CONTAINED OK")
'''

LIST_THE_TREE = '''
print(sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.py")))
'''

LOAD_THE_REGISTRY = '''
import bridge

registry = bridge.model_registry()
assert registry, "registry is empty"
print("REGISTRY OK", len(registry))
'''
