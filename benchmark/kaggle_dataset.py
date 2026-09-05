"""Finding the attached Kaggle Dataset, wherever it actually mounted.

This used to be written directly into the notebook as a string cell. That was
the wrong place for it: a bug in it could only be fixed by regenerating and
re-uploading the whole notebook. It is a normal module now, so a fix here is a
`git pull` away.
"""
import pathlib

DEFAULT_ROOTS = (
    pathlib.Path("/kaggle/input"),
    pathlib.Path("/kaggle/working"),
)


def find_labels(name: str = "labels.csv", roots=None) -> list[pathlib.Path]:
    """Every `labels.csv` under the search roots, shallowest first.

    Searched at any depth because how deep it sits depends on how the dataset
    was zipped -- a top-level folder inside the archive adds a level, and
    Kaggle keeps whatever was in there. Shallowest first because a dataset
    that also ships an example or a backup copy will have the real one
    nearest the top.
    """
    found = []
    for root in (roots or DEFAULT_ROOTS):
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() == name.lower():
                found.append(path)
    return sorted(set(found), key=lambda p: (len(p.parts), str(p)))


def describe_tree(root=pathlib.Path("/kaggle/input"), max_depth: int = 3,
                  max_lines: int = 60) -> str:
    """The mounted tree as text, so a failed search says what IS there
    instead of leaving the path to be guessed at."""
    root = pathlib.Path(root)
    if not root.exists():
        return f"  {root} does not exist"
    base = len(root.parts)
    lines, shown = [], 0
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - base
        if depth > max_depth:
            continue
        if shown >= max_lines:
            lines.append("  ...")
            break
        lines.append(f"  {'  ' * (depth - 1)}{path.name}{'/' if path.is_dir() else ''}")
        shown += 1
    return "\n".join(lines)


def resolve(name: str = "labels.csv", roots=None, override: str | None = None) -> pathlib.Path:
    """The one `labels.csv` to use, or a clear error naming what was found.

    `override` takes priority so a caller can always short-circuit the search
    with a path copied from `describe_tree`'s output.
    """
    if override:
        path = pathlib.Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"override path does not exist: {path}")
        return path

    candidates = find_labels(name, roots)
    if not candidates:
        tree = describe_tree(roots[0] if roots else DEFAULT_ROOTS[0])
        raise FileNotFoundError(
            f"no {name} found under {roots or DEFAULT_ROOTS}.\n"
            f"This is what is actually mounted:\n{tree}\n"
            f"Attach the dataset with Add Input, or pass override=<path you see above>.")
    return candidates[0]


def count_audio(data_dir, extensions=(".mp3", ".wav", ".flac", ".m4a")) -> int:
    """Audio files in a folder, case-insensitively and without double-counting
    on a case-insensitive filesystem where *.mp3 and *.MP3 match the same file."""
    data_dir = pathlib.Path(data_dir)
    seen = {p.resolve() for p in data_dir.iterdir()
           if p.is_file() and p.suffix.lower() in extensions}
    return len(seen)
