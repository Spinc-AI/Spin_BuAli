"""What has been run, what has not, and how a stopped session picks up again.

A Kaggle session ends when Kaggle decides it ends. So the design assumes it
will be killed mid-run, and makes that cost one run rather than the batch:

**The output file is the ledger.** A run is finished when its CSV exists;
there is no separate state file to fall out of step with the results it claims
to describe. Deleting a result deletes the record of it, which is the behaviour
anyone would expect.

**Every write is atomic.** Results go to a temporary name and are renamed into
place, because `os.replace` is atomic on every filesystem we run on. A session
killed mid-write leaves a stray `.tmp`, never a truncated CSV that looks
complete and quietly poisons the leaderboard.

**Runs are addressed by a hash of their configuration.** The same settings
always produce the same id, so resuming is a matter of asking which ids already
have a file, and re-planning after adding a model does not renumber anything.
"""
import csv
import hashlib
import json
import os
import pathlib
import time

RESULTS = "runs"        # <out>/runs/<run_id>.csv    per-recording rows
SUMMARIES = "summaries"  # <out>/summaries/<run_id>.json  the aggregate + config
TRANSCRIPTS = "transcripts"  # <out>/transcripts/<key>.json  the STT cache


def run_id(spec: dict) -> str:
    """A stable short id for one run's configuration.

    Canonical JSON, so key order in the plan never changes the id. Twelve hex
    characters: enough that a collision is not a practical concern, short
    enough to read in a filename.
    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def write_atomic(path: pathlib.Path, write, encoding: str = "utf-8") -> pathlib.Path:
    """Write through a temporary file and rename it into place.

    `write(handle)` does the writing. Nothing observes a partial file: until
    the rename, the destination either does not exist or holds the previous
    complete version.

    Plain utf-8 by default. CSV asks for `utf-8-sig` because Excel needs the
    BOM to read Persian, but a BOM in front of JSON breaks every parser that
    reads it back -- so it is opt-in per file type, not the default.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding=encoding, newline="") as handle:
        write(handle)
    os.replace(temporary, path)
    return path


def write_csv(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    """One CSV, atomically. Columns are the union of the rows' keys, in the
    order they were first seen, so a row missing a field leaves a blank rather
    than shifting the table."""
    columns = list(dict.fromkeys(key for row in rows for key in row))

    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return write_atomic(path, write, encoding="utf-8-sig")


def write_json(path: pathlib.Path, payload) -> pathlib.Path:
    return write_atomic(path, lambda handle: json.dump(
        payload, handle, ensure_ascii=False, indent=2))


class Ledger:
    """The results directory, read as a record of what is already done."""

    def __init__(self, out_dir):
        self.root = pathlib.Path(out_dir)
        for name in (RESULTS, SUMMARIES, TRANSCRIPTS):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    # --- what exists --------------------------------------------------------
    def result_path(self, identifier: str) -> pathlib.Path:
        return self.root / RESULTS / f"{identifier}.csv"

    def summary_path(self, identifier: str) -> pathlib.Path:
        return self.root / SUMMARIES / f"{identifier}.json"

    def is_done(self, identifier: str) -> bool:
        """A run counts as done when its CSV is in place. Because writes are
        atomic, the file existing means the run finished."""
        return self.result_path(identifier).is_file()

    def completed(self) -> set[str]:
        return {path.stem for path in (self.root / RESULTS).glob("*.csv")}

    def pending(self, plan: list[dict]) -> list[dict]:
        """The planned runs with no result yet, in plan order."""
        done = self.completed()
        return [run for run in plan
                if run["run_id"] not in done and run.get("tier") != "C"]

    # --- recording a run ----------------------------------------------------
    def record(self, run: dict, rows: list[dict], summary: dict) -> pathlib.Path:
        """Persist one finished run.

        The summary is written first and the CSV last, so the CSV -- which is
        what `is_done` checks -- is never the thing that exists without the
        rest beside it.
        """
        self.summary_path(run["run_id"]).parent.mkdir(parents=True, exist_ok=True)
        write_json(self.summary_path(run["run_id"]), {"run": run, "summary": summary})
        return write_csv(self.result_path(run["run_id"]), rows)

    # --- the STT cache ------------------------------------------------------
    def transcript_path(self, key: str) -> pathlib.Path:
        return self.root / TRANSCRIPTS / f"{key}.json"

    def cached_transcripts(self, key: str) -> dict | None:
        """Transcripts from an earlier run, if this exact (preprocessing, STT)
        pair has already been done.

        This is what makes re-running the LLM half cheap: speech recognition is
        the expensive stage and does not change when the prompt or the language
        model does.
        """
        path = self.transcript_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt cache entry is a reason to redo it, not to fail

    def cache_transcripts(self, key: str, payload: dict) -> pathlib.Path:
        return write_json(self.transcript_path(key), payload)

    # --- reporting ----------------------------------------------------------
    def status(self, plan: list[dict]) -> dict:
        done = self.completed()
        by_tier: dict[str, dict[str, int]] = {}
        for run in plan:
            counts = by_tier.setdefault(run["tier"], {"done": 0, "pending": 0, "deferred": 0})
            if run["tier"] == "C":
                counts["deferred"] += 1
            elif run["run_id"] in done:
                counts["done"] += 1
            else:
                counts["pending"] += 1
        return {
            "planned": len(plan),
            "completed": sum(1 for run in plan if run["run_id"] in done),
            "by_tier": dict(sorted(by_tier.items())),
        }

    def all_rows(self) -> list[dict]:
        """Every per-recording row from every completed run, for the combined
        leaderboard. Rebuilt from the files, so it always reflects what is
        actually on disk."""
        rows = []
        for path in sorted((self.root / RESULTS).glob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        return rows

    def all_summaries(self) -> list[dict]:
        summaries = []
        for path in sorted((self.root / SUMMARIES).glob("*.json")):
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return summaries


class Budget:
    """A wall-clock limit for one session.

    Kaggle kills a session at twelve hours with no warning and no results. This
    stops before that, between runs, so the work already done is written out
    and the next session picks up from there.
    """

    def __init__(self, minutes: float | None = None, max_runs: int | None = None):
        self.deadline = time.monotonic() + minutes * 60 if minutes else None
        self.max_runs = max_runs
        self.completed = 0

    def remaining_minutes(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, (self.deadline - time.monotonic()) / 60)

    def allows_another(self, expected_minutes: float = 0.0) -> tuple[bool, str]:
        """Whether to start another run. Refuses one that is not expected to
        finish inside the budget, rather than starting it and being killed."""
        if self.max_runs is not None and self.completed >= self.max_runs:
            return False, f"reached the {self.max_runs}-run limit for this session"
        remaining = self.remaining_minutes()
        if remaining is None:
            return True, ""
        if remaining <= 0:
            return False, "session time budget spent"
        if expected_minutes and expected_minutes > remaining:
            return False, (f"the next run needs about {expected_minutes:.0f} min and only "
                           f"{remaining:.0f} min are left in the budget")
        return True, ""

    def record_run(self) -> None:
        self.completed += 1
