'''Stage 4 — cutting the standardized audio into overlapping chunks.

Three strategies, chosen by ``chunking.strategy``:

``adaptive`` (default)
    Picks the chunk count *and* the cut points per recording. It considers every
    chunk count whose chunks would land inside the configured length window, and
    for each one nudges the boundaries onto the quietest moment nearby, then keeps
    whichever count cut through the least speech. A recording that pauses every
    twenty seconds and one that runs on without a breath get different layouts.
``uniform``
    The same even layout with no listening — fixed count from the duration alone.
``fixed``
    Strict target-length windows, leaving a short final piece.

Whatever the strategy, consecutive chunks share ``overlap_sec`` seconds so a word
on a boundary survives whole in one of them, and no chunk exceeds
``max_duration_sec``. Chunks are cut from the standardized audio with silence
left in, so nothing can be lost to a VAD decision.

Standalone:

    python chunking.py recording.wav
'''

import math
import os

import numpy as np

from common import frame_rms, write_wav

NAME = "chunking"
REQUIRES = ("standardization",)


def run(context):
    '''Plan and write the chunks, recording how the layout was chosen.'''
    config = context.config
    sr = config["audio"]["target_sample_rate"]
    duration = len(context.standardized) / sr

    regions = context.regions
    plan, decision = plan_chunks(duration, config, regions, context.standardized, sr)

    context.chunk_decision = decision
    if not context.output_dir:
        return [{"start_sec": round(s, 3), "end_sec": round(e, 3)} for s, e in plan]
    return cut_chunks(
        context.standardized, sr, plan, os.path.join(context.output_dir, config["output"]["chunks_dir"]), config
    )


# ============================================================
# Planning
# ============================================================

def plan_chunks(duration, config, regions=None, audio=None, sr=None):
    '''Lay out overlapping (start, end) windows in seconds over ``regions``.

    Returns the windows and a record of how the layout was decided. ``audio`` is
    what the adaptive strategy listens to; without it adaptive falls back to the
    uniform layout, which is the same thing minus the boundary snapping.
    '''
    settings = config["chunking"]
    regions = regions if regions is not None else [(0.0, duration)]
    strategy = settings["strategy"]

    quietness = None
    if strategy == "adaptive" and audio is not None:
        quietness = _quietness(audio, sr, settings["adaptive"]["frame_ms"])
    elif strategy == "adaptive":
        strategy = "uniform"

    chunks, decisions = [], []
    for start, end in regions:
        start, end = max(0.0, start), min(duration, end)
        if end - start <= 0:
            continue
        windows, decision = _plan_region(start, end, settings, strategy, quietness)
        chunks.extend(windows)
        decisions.append(decision)

    return chunks, {
        "strategy": strategy,
        "requested_strategy": settings["strategy"],
        "listened_to_audio": quietness is not None,
        "regions": decisions,
    }


def _plan_region(start, end, settings, strategy, quietness):
    '''Window a single region with the chosen strategy.'''
    duration = end - start
    if duration <= settings["max_duration_sec"]:
        return [(start, end)], {"chunk_count": 1, "reason": "region fits in one chunk"}

    if strategy == "fixed":
        target = settings["target_duration_sec"]
        windows = _plan_fixed(start, end, target, target - settings["overlap_sec"])
        return windows, {"chunk_count": len(windows), "reason": "fixed target length"}

    if strategy == "uniform":
        count = _base_count(duration, settings)
        return _windows(start, end, count, settings["overlap_sec"]), {
            "chunk_count": count,
            "reason": "even split, audio not consulted",
        }

    return _plan_adaptive(start, end, settings, quietness)


def _plan_adaptive(start, end, settings, quietness):
    '''Try every workable chunk count and keep the one that cuts through least speech.'''
    overlap = settings["overlap_sec"]
    tuning = settings["adaptive"]
    candidates = _candidate_counts(end - start, settings)

    scored = []
    for count in candidates:
        boundaries = _snap_boundaries(start, end, count, settings, quietness)
        windows = _windows_from(boundaries, start, end, overlap)
        quiet = _boundary_quietness(boundaries[1:-1], quietness)
        penalty = tuning["extra_chunk_penalty"] * (count - candidates[0])
        scored.append((quiet - penalty, quiet, count, windows))

    scored.sort(key=lambda entry: (-entry[0], entry[2]))
    _, quiet, count, windows = scored[0]
    return windows, {
        "chunk_count": count,
        "counts_considered": candidates,
        "boundary_quietness": round(quiet, 4),
        "rejected": [
            {"chunk_count": entry[2], "boundary_quietness": round(entry[1], 4)}
            for entry in scored[1:]
        ],
        "reason": "quietest boundaries among the workable chunk counts",
    }


def _candidate_counts(duration, settings):
    '''Every chunk count whose chunks would stay inside the configured length window.

    A chunk carries ``overlap`` seconds of its neighbours (half at each edge), so
    the written length of an interior chunk is ``duration / count + overlap``. The
    bounds below are that expression solved against the min and max.
    '''
    overlap = settings["overlap_sec"]
    target, minimum, maximum = (
        settings["target_duration_sec"],
        settings["min_duration_sec"],
        settings["max_duration_sec"],
    )
    extra = settings["adaptive"]["max_extra_chunks"]

    base = max(2, math.ceil(duration / (target - overlap)))
    lowest = max(2, math.ceil(duration / (maximum - overlap)))
    highest = max(lowest, math.floor(duration / max(minimum - overlap / 2, 1e-6)))

    counts = [n for n in range(base - extra, base + extra + 1) if lowest <= n <= highest]
    return counts or [max(base, lowest)]


def _base_count(duration, settings):
    '''The chunk count an even split would use, ignoring the audio itself.'''
    return max(2, math.ceil(duration / (settings["target_duration_sec"] - settings["overlap_sec"])))


def _windows(start, end, count, overlap):
    '''Evenly spaced content boundaries turned into overlapping windows.'''
    step = (end - start) / count
    boundaries = [start + index * step for index in range(count)] + [end]
    return _windows_from(boundaries, start, end, overlap)


def _windows_from(boundaries, start, end, overlap):
    '''Widen each content span by half the overlap on each side, clamped to the region.

    Consecutive windows then share exactly ``overlap`` seconds, and the first and
    last keep the region's own edges.
    '''
    half = overlap / 2
    return [
        (max(start, left - half), min(end, right + half))
        for left, right in zip(boundaries, boundaries[1:])
    ]


def _snap_boundaries(start, end, count, settings, quietness):
    '''Move each interior boundary onto the quietest moment it is allowed to reach.

    Each boundary is kept within ``snap_window_sec`` of where an even split would
    have put it, and within the range that keeps the chunk behind it and every
    chunk still to come inside the configured length window — so listening to the
    audio can improve where a cut lands but can never produce an illegal chunk.
    '''
    overlap = settings["overlap_sec"]
    window = settings["adaptive"]["snap_window_sec"]
    shortest = settings["min_duration_sec"] - overlap / 2
    longest = settings["max_duration_sec"] - overlap

    step = (end - start) / count
    ideal = [start + index * step for index in range(count)] + [end]

    snapped = [start]
    for index in range(1, count):
        previous = snapped[index - 1]
        # What is left after this cut has to divide into the remaining chunks, so
        # the bound is the whole remainder rather than just the next boundary —
        # constraining against the *ideal* next cut would pin every boundary to
        # within a fraction of a second of where it started.
        remaining = count - index
        low = max(ideal[index] - window, previous + shortest, end - remaining * longest)
        high = min(ideal[index] + window, previous + longest, end - remaining * shortest)
        snapped.append(_quietest(quietness, ideal[index], low, high))
    snapped.append(end)
    return snapped


def _quietness(audio, sr, frame_ms):
    '''A 0..1 quietness curve over the recording — 1 where it is most silent.'''
    energy, step = frame_rms(audio, sr, frame_ms)
    loudest = energy.max()
    return (1.0 - energy / loudest if loudest > 0 else np.ones_like(energy)), step


def _quietest(quietness, ideal, low, high):
    '''The quietest time in [low, high], or ``ideal`` when the range is unusable.'''
    if quietness is None or low >= high:
        return ideal
    curve, step = quietness
    first = max(0, int(math.ceil(low / step)))
    last = min(len(curve) - 1, int(math.floor(high / step)))
    if first > last:
        return ideal
    return float((first + int(np.argmax(curve[first : last + 1]))) * step)


def _boundary_quietness(boundaries, quietness):
    '''Mean quietness at the cut points — the score adaptive chunking maximises.'''
    if quietness is None or not boundaries:
        return 0.0
    curve, step = quietness
    values = [curve[min(len(curve) - 1, int(round(time / step)))] for time in boundaries]
    return float(np.mean(values))


def _plan_fixed(start, end, target, step):
    '''Window a region into exact-length chunks, leaving a short final piece.'''
    windows, cursor = [], start
    while cursor < end:
        windows.append((cursor, min(cursor + target, end)))
        if windows[-1][1] >= end:
            break
        cursor += step
    return windows


# ============================================================
# Writing
# ============================================================

def cut_chunks(audio, sr, plan, chunk_dir, config):
    '''Write each planned window to its own WAV and describe it for segments.json.'''
    os.makedirs(chunk_dir, exist_ok=True)
    name_format = config["output"]["chunk_name_format"]
    chunks_dir_name = config["output"]["chunks_dir"]

    records = []
    for index, (start, end) in enumerate(plan):
        first = max(0, int(round(start * sr)))
        last = min(len(audio), int(round(end * sr)))
        name = name_format.format(index=index)
        write_wav(os.path.join(chunk_dir, name), audio[first:last], config)

        previous = plan[index - 1] if index else None
        following = plan[index + 1] if index + 1 < len(plan) else None
        records.append(
            {
                "index": index,
                "file": f"{chunks_dir_name}/{name}",
                "start_sec": round(first / sr, 3),
                "end_sec": round(last / sr, 3),
                "duration_sec": round((last - first) / sr, 3),
                "start_sample": first,
                "end_sample": last,
                "overlap_prev_sec": round(max(0.0, previous[1] - start), 3) if previous else 0.0,
                "overlap_next_sec": round(max(0.0, end - following[0]), 3) if following else 0.0,
            }
        )
    return records


if __name__ == "__main__":
    import json
    import sys

    from common import decode, load_config

    settings = load_config()
    samples, rate, _, _ = decode(sys.argv[1])
    windows, how = plan_chunks(len(samples) / rate, settings, None, samples, rate)
    print(json.dumps({"decision": how, "chunks": [list(w) for w in windows]}, indent=2))
