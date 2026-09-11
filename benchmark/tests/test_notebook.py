"""The generated notebook: thin, dependent on a git clone, one cell per run.

This is not the self-contained design from before. The notebook now has no
logic of its own -- every function it calls lives in a `.py` file here, and a
bug fix means fixing that file and re-running the clone cell, not
regenerating and re-uploading this one. So what these tests protect is
different too:

* the notebook is regenerated from the current source (no silent drift)
* every cell is syntactically valid
* the module-collision bug this design is exposed to -- three sibling
  directories each with their own `config.py` -- cannot come back silently
* one cell per (STT, LLM, pipeline) run, not a loop, so a run is one cell you
  can skip on resume
* it is actually connected: run against a live checkout of this repo with a
  simulated Kaggle mount, it produces one CSV per run and a merged master
"""
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "benchmark" / "notebooks" / "kaggle_dual_t4.ipynb"
BUILDER = REPO / "benchmark" / "notebooks" / "build_notebook.py"

# Cells the harness cannot run standalone: a real git clone, a real pip
# install, and Hugging Face auth against the live API.
NOT_RUNNABLE_OFFLINE = ("subprocess.run", "pip install", "resolve_hf_token", "check_access(")


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
        """Fails when a module changed and the notebook was not rebuilt.
        Run `python benchmark/notebooks/build_notebook.py`."""
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

    def test_the_notebook_has_no_embedded_source_files(self, notebook):
        """The previous design wrote modules to disk as raw strings and
        imported them back. That is gone -- everything here is `import x`."""
        source = "".join(source_of(c) for c in code_cells(notebook))
        assert 'write("' not in source
        assert "r'''" not in source


class TestItDependsOnACleanClone:
    def test_it_clones_the_named_branch(self, notebook):
        source = "".join(source_of(c) for c in code_cells(notebook))
        assert "git" in source and "clone" in source
        assert "benchmark-selfcontained" in source

    def test_only_benchmark_goes_on_sys_path_directly(self, notebook):
        """The bug this pins: evaluation/, controller/ and stt/app/ each carry
        their own config.py. Putting all of them on sys.path in the clone
        cell -- rather than letting bridge.py reach into them lazily -- makes
        whichever one insert() processed last shadow the others, and
        `evaluate_results.py`'s `import config` silently resolves to the
        wrong file."""
        clone_cell = next(c for c in code_cells(notebook) if "git" in source_of(c))
        source = source_of(clone_cell)
        assert 'sys.path.insert(0, str(REPO_DIR / "benchmark"))' in source
        for other in ("evaluation", "controller", "stt/app", "stt"):
            assert f'REPO_DIR / "{other}"' not in source, (
                f"{other} must not be inserted directly; bridge.py reaches it")

    def test_a_second_run_pulls_instead_of_recloning(self, notebook):
        clone_cell = next(c for c in code_cells(notebook) if "git" in source_of(c))
        source = source_of(clone_cell)
        assert "pull" in source or ("fetch" in source and "reset" in source)


class TestOneCellPerRun:
    def test_run_cells_outnumber_the_pipelines_run_calls_by_exactly_that_many(self, notebook):
        run_cells = [c for c in code_cells(notebook) if "runner.run_one(" in source_of(c)]
        assert len(run_cells) >= 3, "one cell per top-3 STT engine, at minimum"

    def test_no_cell_loops_over_multiple_runs(self, notebook):
        """The whole point: a `for` loop calling run_one several times would
        put several runs behind one cell, and stopping the session mid-loop
        would lose whichever run was in flight. Each run gets its own cell."""
        for cell in code_cells(notebook):
            source = source_of(cell)
            if "runner.run_one(" in source:
                assert source.count("runner.run_one(") == 1

    def test_each_run_cell_names_a_different_stt_engine(self, notebook):
        run_cells = [source_of(c) for c in code_cells(notebook) if "runner.run_one(" in source_of(c)]
        first_args = []
        for source in run_cells:
            start = source.index("runner.run_one(") + len("runner.run_one(")
            first_args.append(source[start:source.index(",", start)].strip())
        assert len(first_args) == len(set(first_args)), first_args


class TestItIsActuallyConnected:
    """Runs the generated notebook's own cells against a live checkout of this
    repo and a simulated Kaggle mount. Dry-run STT and LLM factories stand in
    for real weights -- this proves the wiring, not the models."""

    @staticmethod
    def _runnable_cells(notebook):
        return [c for c in code_cells(notebook)
               if not any(marker in source_of(c) for marker in NOT_RUNNABLE_OFFLINE)]

    def test_it_runs_end_to_end_and_writes_a_csv_per_run_plus_a_master(
            self, notebook, tmp_path, monkeypatch):
        mount = tmp_path / "kaggle_input" / "spin-buali-dataset" / "Spin_BuAli_DataSet"
        mount.mkdir(parents=True)
        _write_demo_dataset(mount)

        sys.path.insert(0, str(REPO / "benchmark"))
        import kaggle_dataset
        import llm as llm_module
        import transcribe as transcribe_module
        monkeypatch.setattr(kaggle_dataset, "DEFAULT_ROOTS", (tmp_path / "kaggle_input",))

        namespace = {"__name__": "__main__"}
        exec(compile(
            "import pathlib, sys\n"
            f"REPO_DIR = pathlib.Path(r'{REPO}')\n"
            "sys.path.insert(0, str(REPO_DIR / 'benchmark'))\n",
            "<setup>", "exec"), namespace)
        namespace["_dry_stt"] = transcribe_module.dry_run_factory
        namespace["_dry_llm"] = llm_module.dry_run_factory

        for cell in self._runnable_cells(notebook):
            source = source_of(cell)
            if "runner.run_one(" in source:
                source = source.replace(
                    "df_", "__kw = dict(model_factory=_dry_stt, llm_factory=_dry_llm)\ndf_", 1)
                source = source.replace("results_dir=RESULTS_DIR,",
                                        "results_dir=RESULTS_DIR, **__kw,")
            exec(compile(source, "<cell>", "exec"), namespace)
            if "RESULTS_DIR = pathlib.Path(" in source:
                # The config cell hardcodes /kaggle/working/results, which on
                # this machine resolves to a real, persistent path outside the
                # test sandbox. Redirect it the moment it is set, before any
                # run cell can write through it.
                namespace["RESULTS_DIR"] = tmp_path / "results"

        results_dir = pathlib.Path(str(namespace["RESULTS_DIR"]))
        per_run_csvs = sorted(results_dir.glob("results__*.csv"))
        assert len(per_run_csvs) == 3, "one CSV per top-3 STT engine"
        assert (results_dir / "results_master.csv").is_file()
        assert namespace["master"] is not None and len(namespace["master"]) == 3

    def test_the_dataset_is_found_at_three_levels_of_nesting(self, notebook, tmp_path, monkeypatch):
        """The layout a zipped Kaggle Dataset actually produces: an extra
        folder inside the archive that a two-level-only search would miss."""
        mount = tmp_path / "kaggle_input" / "spin-buali-dataset" / "Spin_BuAli_DataSet"
        mount.mkdir(parents=True)
        _write_demo_dataset(mount)

        sys.path.insert(0, str(REPO / "benchmark"))
        import kaggle_dataset
        monkeypatch.setattr(kaggle_dataset, "DEFAULT_ROOTS", (tmp_path / "kaggle_input",))
        resolved = kaggle_dataset.resolve()
        assert resolved == mount / "labels.csv"


def _write_demo_dataset(folder: pathlib.Path):
    """Two tiny recordings and a labels.csv, the shape `dataset.from_csv`
    expects. Real audio, so the STT stage's decode step has something to
    decode even though the model itself is a dry-run stub."""
    import csv

    import numpy as np
    import soundfile as sf

    rows = []
    for name, seconds, reference in (
        ("A1", 2.0, "There is a 6 mm stone in the right kidney."),
        ("A2", 3.0, "The urinary bladder is normal."),
    ):
        sf.write(str(folder / f"{name}.wav"), np.zeros(int(seconds * 16000), np.float32), 16000)
        rows.append({"asset_id": name, "audio": f"{name}.wav", "image": "",
                     "report": reference})
    with (folder / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "audio", "image", "report"])
        writer.writeheader()
        writer.writerows(rows)
