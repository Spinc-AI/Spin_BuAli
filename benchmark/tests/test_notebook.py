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
# install, Hugging Face auth against the live API, and the version check
# (it imports torch, which this suite deliberately runs without).
NOT_RUNNABLE_OFFLINE = ("subprocess.run", "pip install", "resolve_hf_token", "check_access(",
                        "import numpy, torch, transformers")


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
    def test_run_cells_are_exactly_the_multimodal_roster(self, notebook):
        """One cell per audio-capable LLM, and nothing else.

        `separate` is deliberately not in this notebook: on this dataset the
        STT stage was the entire result (seamless returned unrelated English
        sentences for a kidney ultrasound), so every LLM behind it scored near
        WER 1.0 for the STT engine's failure rather than its own. The pipeline
        is still in pipeline.py and still tested -- it just has no cell here.
        """
        import sys

        sys.path.insert(0, str(REPO / "benchmark"))
        import runner

        run_cells = [source_of(c) for c in code_cells(notebook)
                     if "runner.run_one(" in source_of(c)]
        assert len(run_cells) == len(runner.MULTIMODAL_LLM)
        assert all('"multimodal"' in s for s in run_cells)
        assert not any('"separate"' in s for s in run_cells)

    def test_the_roster_comes_from_runner_not_a_second_copy(self, notebook):
        """The drift this pins: build_notebook.py restated the rosters as
        literals, with nothing tying them to runner.py -- so a roster edit
        there would silently not reach the generated notebook. Every model
        runner names must appear in a run cell, and no run cell may name a
        model runner doesn't."""
        import sys

        sys.path.insert(0, str(REPO / "benchmark"))
        import runner

        run_cells = [source_of(c) for c in code_cells(notebook)
                     if "runner.run_one(" in source_of(c)]
        named = {key for source in run_cells
                 for key in runner.MULTIMODAL_LLM if f'"{key}"' in source}
        expected = set(runner.MULTIMODAL_LLM)
        assert named == expected, f"notebook and runner.py disagree: {named ^ expected}"

        # No STT engine may appear either -- a leftover `separate` cell would
        # name one, and would otherwise pass every other check here.
        for stt_key in runner.TOP3_STT:
            assert not any(f'"{stt_key}"' in s for s in run_cells), (
                f"{stt_key} still appears in a run cell; this notebook is multimodal only")

    def test_no_cell_loops_over_multiple_runs(self, notebook):
        """The whole point: a `for` loop calling run_one several times would
        put several runs behind one cell, and stopping the session mid-loop
        would lose whichever run was in flight. Each run gets its own cell."""
        for cell in code_cells(notebook):
            source = source_of(cell)
            if "runner.run_one(" in source:
                assert source.count("runner.run_one(") == 1

    def test_every_run_cell_writes_a_distinct_label(self, notebook):
        """Several cells legitimately share an stt_key (3 LLMs per STT engine)
        or an llm_key (3 STT engines, or the multimodal roster) -- what must
        never collide is the CSV each cell writes, which the `label=` builds."""
        import re

        run_cells = [source_of(c) for c in code_cells(notebook) if "runner.run_one(" in source_of(c)]
        labels = []
        for source in run_cells:
            match = re.search(r'label="([^"]+)"', source)
            assert match, "run cell has no static label= prefix to check"
            labels.append(match.group(1))
        assert len(labels) == len(set(labels)), labels

    def test_every_run_cell_passes_no_stt_engine(self, notebook):
        """The check that makes "multimodal is one step" true in the notebook
        and not just in the docstrings: `run_one`'s STT stage is gated on its
        first positional argument, so a cell that passed an engine there would
        transcribe first and feed the text in -- a different experiment under
        the same name."""
        run_cells = [source_of(c) for c in code_cells(notebook)
                     if "runner.run_one(" in source_of(c)]
        multimodal = [s for s in run_cells if '"multimodal"' in s]
        assert multimodal and len(multimodal) == len(run_cells)
        for source in multimodal:
            start = source.index("runner.run_one(") + len("runner.run_one(")
            first_arg = source[start:source.index(",", start)].strip()
            assert first_arg == "None"


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
        sys.path.insert(0, str(REPO / "benchmark"))
        import runner

        expected_runs = len(runner.MULTIMODAL_LLM)
        per_run_csvs = sorted(results_dir.glob("results__*.csv"))
        assert len(per_run_csvs) == expected_runs, "one CSV per multimodal run"
        assert (results_dir / "results_master.csv").is_file()
        assert namespace["master"] is not None and len(namespace["master"]) == expected_runs

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
