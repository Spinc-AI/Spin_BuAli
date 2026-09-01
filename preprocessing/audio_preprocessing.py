'''Audio preprocessing pipeline — runs the six stages, all of them or a subset.

Each stage lives in its own module and can be imported or run on its own; this
file is the one that puts them together:

    1. validation       — decodable, non-empty, finite, plausible length
    2. standardization  — WAV / PCM 16-bit / mono / 16 kHz
    3. vad              — where the speech is, padded (experimental, non-destructive)
    4. chunking         — overlapping chunks, count and cut points chosen per file
    5. timestamps       — segments.json plus the acceptance checks
    6. report           — preprocessing_report.csv

The original file is only ever read — never rewritten, gain-adjusted, denoised or
silence-stripped. Anything the pipeline *does* change in the derived copy is
recorded rather than applied silently. Every knob lives in
``preprocessing_config.json``; nothing is hard-coded here.

    python audio_preprocessing.py recording.mp3 -o output
    python audio_preprocessing.py recordings/ -o output --skip vad
    python audio_preprocessing.py recording.wav -o output --only validation,standardization
    python audio_preprocessing.py --list-stages

Skipping a stage leaves it out of the metadata entirely — ``--skip vad`` produces
a ``segments.json`` with no ``vad`` key at all, not an empty one.

Output layout, one directory per input file (named by its content hash):

    output/8af1c23d19ab70e4/
    ├── standardized.wav
    ├── chunks/
    │   ├── chunk_000.wav
    │   └── chunk_001.wav
    └── segments.json
'''

import argparse
import json
import logging
import os
import shutil
import sys

import chunking
import report as report_stage
import standardization
import timestamps
import vad as vad_stage
import validation
from common import (
    AUDIO_EXTENSIONS,
    PreprocessingError,
    file_id,
    file_sha256,
    load_config,
    log,
)

# The six stages, in the order they run. Each module exposes NAME, REQUIRES and
# run(context); adding a stage means writing one module and listing it here.
STAGES = [validation, standardization, vad_stage, chunking, timestamps, report_stage]
STAGE_NAMES = [stage.NAME for stage in STAGES]

# Stages that only make sense as part of a run, rather than as something to skip:
# without validation there is no audio, and without timestamps nothing is written.
ALWAYS = {"validation"}


class Context:
    '''Everything the stages hand to each other for one file.'''

    def __init__(self, path, output_dir, config, metadata=None):
        self.path = os.path.abspath(path)
        self.output_dir = output_dir
        self.config = config
        self.metadata = metadata or {}

        self.sha256 = file_sha256(self.path)
        self.file_id = file_id(self.sha256, config["output"]["id_strategy"])

        self.audio = None
        self.sample_rate = None
        self.standardized = None
        self.regions = None
        self.chunk_decision = None
        self.results = {}
        self.stages_run = []


# ============================================================
# Stage selection
# ============================================================

def select_stages(only=None, skip=None):
    '''Resolve --only / --skip into the ordered list of stages to run.

    Dependencies are checked rather than silently satisfied: asking for chunking
    without standardization is a mistake worth hearing about, not something to
    paper over by running a stage that was not requested.
    '''
    if only and skip:
        raise ValueError("use --only or --skip, not both")

    if only:
        wanted = {name.strip() for name in only if name.strip()}
    else:
        wanted = set(STAGE_NAMES)
    wanted |= ALWAYS

    for name in wanted:
        if name not in STAGE_NAMES:
            raise ValueError(f"unknown stage '{name}' — choose from {', '.join(STAGE_NAMES)}")

    if skip:
        removed = {name.strip() for name in skip if name.strip()}
        for name in removed:
            if name not in STAGE_NAMES:
                raise ValueError(f"unknown stage '{name}' — choose from {', '.join(STAGE_NAMES)}")
            if name in ALWAYS:
                raise ValueError(f"stage '{name}' cannot be skipped — nothing runs without it")
        wanted -= removed

    selected = [stage for stage in STAGES if stage.NAME in wanted]
    for stage in selected:
        missing = [name for name in stage.REQUIRES if name not in wanted]
        if missing:
            raise ValueError(
                f"stage '{stage.NAME}' needs {', '.join(missing)}, which this selection leaves out"
            )
    return selected


# ============================================================
# Pipeline
# ============================================================

def process_file(path, output_root, config, metadata=None, stages=None):
    '''Run one file through the selected stages and return its metadata record.

    Raises ``PreprocessingError`` if the input fails validation or the written
    artefacts fail the acceptance checks; the caller decides whether one bad file
    stops the batch.
    '''
    stages = stages if stages is not None else select_stages()
    names = [stage.NAME for stage in stages]

    context = Context(path, None, config, metadata)
    context.stages_run = names
    output_dir = os.path.join(output_root, context.file_id) if output_root else None

    if output_dir and _already_done(output_dir, config, path, context.file_id):
        with open(os.path.join(output_dir, config["output"]["segments_name"]), encoding="utf-8") as handle:
            return {**json.load(handle), "output_dir": output_dir}

    if output_dir and _writes_files(names):
        # Clear the directory rather than writing over it: a previous run with a
        # longer chunk list would otherwise leave orphaned chunk_0NN.wav files.
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        context.output_dir = output_dir

    for stage in stages:
        if stage.NAME == "report":
            continue
        log.debug("%s: %s", os.path.basename(path), stage.NAME)
        outcome = stage.run(context)
        if stage.NAME == "timestamps":
            result = {**outcome, "output_dir": output_dir}
        else:
            context.results[stage.NAME] = outcome
            _wire(stage.NAME, context)

    if "timestamps" not in names:
        result = {**timestamps.assemble(context), "output_dir": output_dir}

    if result.get("checks", {}).get("problems"):
        raise PreprocessingError("; ".join(result["checks"]["problems"]))
    return result


def _wire(name, context):
    '''Pass what one stage learned to the stage that needs it.'''
    if name != "vad":
        return

    # Chunking over VAD regions is opt-in, and only when VAD actually found
    # something: a backend that returns nothing would otherwise produce a
    # directory with no chunks at all, quietly losing the recording.
    if context.config["chunking"]["source"] != "vad_segments":
        return
    report = context.results["vad"]
    if report["available"] and report["segments"]:
        context.regions = [(s["start_sec"], s["end_sec"]) for s in report["segments"]]
    else:
        log.warning(
            "%s: chunking.source is 'vad_segments' but VAD found no speech; "
            "chunking the whole file instead",
            os.path.basename(context.path),
        )
        report["chunking_fell_back_to_whole_file"] = True


def _writes_files(names):
    '''True when the selection produces artefacts worth clearing a directory for.'''
    return bool({"standardization", "chunking", "timestamps"} & set(names))


def _already_done(output_dir, config, path, identifier):
    '''True when a previous run's segments.json is there and overwrite is off.'''
    if config["output"]["overwrite"]:
        return False
    if not os.path.isfile(os.path.join(output_dir, config["output"]["segments_name"])):
        return False
    log.info("%s already processed as %s, skipping", os.path.basename(path), identifier)
    return True


def process_all(paths, output_root, config, metadata=None, stages=None, stop_on_error=False):
    '''Process every input path, collecting one row per file for the report.'''
    stages = stages if stages is not None else select_stages()
    names = [stage.NAME for stage in stages]
    metadata = metadata or {}

    rows, results = [], []
    for path in paths:
        entry = metadata if _is_flat(metadata) else metadata.get(os.path.basename(path), {})
        try:
            result = process_file(path, output_root, config, entry, stages)
            results.append(result)
            rows.append(report_stage.row(result))
            log.info("%s -> %s", os.path.basename(path), _summary(result))
        except Exception as exc:
            if stop_on_error:
                raise
            log.error("%s -> failed: %s", os.path.basename(path), exc)
            rows.append(report_stage.failure_row(path, exc, names))
    return results, rows


def _summary(result):
    '''One-line description of what a file turned into, for the log.'''
    parts = [result.get("file_id", "")]
    if "standardized" in result:
        parts.append(f"{result['standardized']['duration_sec']:.1f}s")
    if "chunks" in result:
        decision = result.get("chunking", {}).get("decision", {})
        parts.append(f"{len(result['chunks'])} chunk(s) [{decision.get('strategy', '?')}]")
    if "vad" in result:
        parts.append(f"vad={result['vad']['backend']}")
    return "  ".join(part for part in parts if part)


def _is_flat(metadata):
    '''True when the metadata file is one record for every input, not a per-file map.'''
    return bool(metadata) and not all(isinstance(value, dict) for value in metadata.values())


# ============================================================
# CLI
# ============================================================

def collect_inputs(paths):
    '''Expand directories into the audio files they contain, keeping order stable.'''
    collected = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                        collected.append(os.path.join(root, name))
        elif os.path.isfile(path):
            collected.append(path)
        else:
            raise FileNotFoundError(path)
    return collected


def parse_args(argv=None):
    '''Build the command line described in this module's docstring.'''
    parser = argparse.ArgumentParser(description="Standardize, segment and chunk audio for STT.")
    parser.add_argument("inputs", nargs="*", help="audio files, or directories of them")
    parser.add_argument("-o", "--output-dir", default=None, help="output root (default: config)")
    parser.add_argument("-c", "--config", default=None, help="path to preprocessing_config.json")
    parser.add_argument("-m", "--metadata", default=None, help="JSON metadata, flat or keyed by filename")
    parser.add_argument("--report", default=None, help="path to preprocessing_report.csv")
    parser.add_argument("--only", default=None, help=f"run only these stages ({', '.join(STAGE_NAMES)})")
    parser.add_argument("--skip", default=None, help="run every stage except these")
    parser.add_argument("--list-stages", action="store_true", help="print the stages and exit")
    parser.add_argument("--strategy", default=None, choices=("adaptive", "uniform", "fixed"),
                        help="override chunking.strategy for this run")
    parser.add_argument("--no-append", action="store_true", help="overwrite the report instead of appending")
    parser.add_argument("--stop-on-error", action="store_true", help="abort the batch on the first failure")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every stage")
    return parser.parse_args(argv)


def main(argv=None):
    '''Entry point: process the inputs, write the report, report failures.'''
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.list_stages:
        for stage in STAGES:
            needs = f" (needs {', '.join(stage.REQUIRES)})" if stage.REQUIRES else ""
            fixed = " [always runs]" if stage.NAME in ALWAYS else ""
            print(f"{stage.NAME}{needs}{fixed}")
        return 0

    if not args.inputs:
        log.error("no inputs given")
        return 2

    config = load_config(args.config)
    if args.strategy:
        config["chunking"]["strategy"] = args.strategy

    try:
        stages = select_stages(
            args.only.split(",") if args.only else None,
            args.skip.split(",") if args.skip else None,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    log.info("stages: %s", " -> ".join(stage.NAME for stage in stages))

    output_root = args.output_dir or config["output"]["root"]
    metadata = {}
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as handle:
            metadata = json.load(handle)

    inputs = collect_inputs(args.inputs)
    if not inputs:
        log.error("no audio files found in %s", ", ".join(args.inputs))
        return 2

    _, rows = process_all(inputs, output_root, config, metadata, stages, args.stop_on_error)

    failed = sum(1 for row in rows if row["status"] != "ok")
    if any(stage.NAME == "report" for stage in stages):
        path = args.report or os.path.join(output_root, config["output"]["report_name"])
        report_stage.write_report(path, rows, append=not args.no_append)
        log.info("%d/%d file(s) processed, report at %s", len(rows) - failed, len(rows), path)
    else:
        log.info("%d/%d file(s) processed, no report written", len(rows) - failed, len(rows))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
