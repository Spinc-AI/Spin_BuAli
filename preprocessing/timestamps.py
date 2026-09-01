'''Stage 5 — assembling and writing segments.json.

Collects what the earlier stages produced into the one metadata file, and checks
the written artefacts against the task's acceptance criteria before declaring
them good. Only stages that actually ran contribute a key: skip VAD and the
``vad`` key is absent from the file rather than present and empty.
'''

import json
import os
from datetime import datetime, timezone

from common import file_sha256, probe

NAME = "timestamps"
REQUIRES = ()


def run(context):
    '''Write segments.json from whatever the run produced, then verify it.'''
    result = assemble(context)
    result["checks"] = {"passed": None, "problems": []}

    if context.output_dir:
        problems = verify(result, context.config, context.output_dir)
        result["checks"] = {"passed": not problems, "problems": problems}
        write_segments(result, context.output_dir, context.config)
    return result


def assemble(context):
    '''Build the metadata record, including only the stages that ran.'''
    result = {
        "file_id": context.file_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages_run": list(context.stages_run),
        "source": {
            "path": context.path,
            "filename": os.path.basename(context.path),
            "sha256": context.sha256,
            "size_bytes": os.path.getsize(context.path),
            **context.results.get("validation", {}),
        },
        "metadata": context.metadata or {},
        "policy": dict(context.config["policy"]),
    }

    if "standardization" in context.results:
        result["standardized"] = context.results["standardization"]
    if "vad" in context.results:
        result["vad"] = context.results["vad"]
    if "chunking" in context.results:
        result["chunking"] = {**context.config["chunking"], "decision": context.chunk_decision}
        result["chunks"] = context.results["chunking"]
    return result


def write_segments(result, output_dir, config):
    '''Persist segments.json next to the audio it describes.'''
    path = os.path.join(output_dir, config["output"]["segments_name"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return path


# ============================================================
# Acceptance checks
# ============================================================

def verify(result, config, output_dir):
    '''Re-check the written artefacts against the acceptance criteria.

    Runs on the files on disk, not on the in-memory arrays, so a wrong subtype or
    a truncated write is caught here rather than downstream in the STT service.
    Each check applies only if the stage that would have produced it ran.
    '''
    audio_cfg, chunk_cfg = config["audio"], config["chunking"]
    # Timestamps are stored in milliseconds, so a comparison against the audio
    # itself has to allow that rounding plus the sample the boundary lands on.
    tolerance = 1e-3 + 1.0 / audio_cfg["target_sample_rate"]
    problems = []

    if "standardized" in result:
        written = probe(os.path.join(output_dir, config["output"]["standardized_name"]))
        if (written["sample_rate"], written["channels"], written["subtype"], written["format"]) != (
            audio_cfg["target_sample_rate"],
            audio_cfg["target_channels"],
            audio_cfg["target_subtype"],
            audio_cfg["target_format"],
        ):
            problems.append(f"standardized.wav is {written} not the configured target")

    duration = result.get("standardized", {}).get("duration_sec")

    for chunk in result.get("chunks", []):
        path = os.path.join(output_dir, chunk["file"].replace("/", os.sep))
        info = probe(path)
        if (info["sample_rate"], info["channels"], info["subtype"]) != (
            audio_cfg["target_sample_rate"],
            audio_cfg["target_channels"],
            audio_cfg["target_subtype"],
        ):
            problems.append(f"{chunk['file']} is not WAV/PCM_16/mono/16kHz")
        if info["duration_sec"] > chunk_cfg["max_duration_sec"] + tolerance:
            problems.append(
                f"{chunk['file']} is {info['duration_sec']:.3f}s, over the "
                f"{chunk_cfg['max_duration_sec']}s maximum"
            )
        if abs(info["duration_sec"] - chunk["duration_sec"]) > tolerance:
            problems.append(f"{chunk['file']} duration disagrees with its timestamp")
        if duration is not None and (chunk["start_sec"] < 0 or chunk["end_sec"] > duration + tolerance):
            problems.append(f"{chunk['file']} timestamps fall outside the file")

    for segment in result.get("vad", {}).get("segments", []):
        if duration is not None and (segment["start_sec"] < 0 or segment["end_sec"] > duration + tolerance):
            problems.append(f"VAD segment {segment['index']} falls outside the file")

    if result["source"]["sha256"] != file_sha256(result["source"]["path"]):
        problems.append("the original file changed during processing")

    return problems
