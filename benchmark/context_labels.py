"""Modality/region context, fed to the LLM alongside the transcript or audio.

A radiologist dictating a report already knows what study they are looking
at -- the order says "abdominal ultrasound," not "guess from what you hear."
An LLM given only the transcript has no such prior: nothing stops it from
describing lung findings on an abdominal study, or missing that "kidney"
in a mangled transcript almost certainly means a real kidney line is coming,
not a mistranscription. `dataset.Item.modality`/`.regions` -- read from
`labels.csv`'s optional `modality`/`region` columns -- carry exactly the
context a real order would, and this module turns that into one prompt line.

Benchmark-only, same reasoning as `report_structure.py`: this is metadata
about a specific labelled dataset, not something `controller/prompts.py`
can assume every caller has. A caller with no modality/region info (most
callers, today) gets no line at all -- this never invents context, and
never blocks a run that has none.
"""


def context_line(modality: str | None, regions: tuple[str, ...] | list[str] | None) -> str | None:
    """One line of known context for a recording, or `None` if nothing is known.

    `modality`/`regions` are exactly `dataset.Item.modality`/`.regions` --
    pass them straight through, no lookup against `docs/taxonomy/` required.
    A caller with only one of the two still gets a line naming just that one.
    """
    parts = []
    if modality:
        parts.append(f"Modality: {modality}.")
    if regions:
        parts.append(f"Region(s) examined: {', '.join(regions)}.")
    if not parts:
        return None
    return (
        "Known context for this recording (provided by the ordering system, "
        "not inferred -- use it to set expectations, but still report only "
        "what the recording actually supports): " + " ".join(parts)
    )
