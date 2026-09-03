"""Generate the Kaggle notebook.

The notebook is written the way a notebook should be: every class and function
is defined in a cell you can read, edit and re-run. Nothing is written to disk
and imported back, and nothing is imported from this repo -- attach the audio
dataset, hit Run All, and it works.

That is a deliberate trade against the rest of the project. The repo's modules
exist to be services; the notebook is a flat, self-contained rewrite of the
parts a benchmark needs. It can drift from `evaluation/`, so
`tests/test_notebook.py` scores the same reports both ways and fails if the
numbers stop agreeing.

    python build_notebook.py
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "kaggle_dual_t4.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


def clinical_terms_literal():
    """The vocabulary as a Python literal, so the cell reads as data not a blob."""
    data = json.loads((REPO / "evaluation" / "clinical_terms.json").read_text(encoding="utf-8"))
    lines = [f'CLINICAL_TERMS_VERSION = "{data.get("version", "unknown")}"', "", "CONCEPTS = ["]
    for concept in data["concepts"]:
        variants = ", ".join(json.dumps(v, ensure_ascii=False) for v in concept["variants"])
        lines.append(f'    ("{concept["id"]}", "{concept["category"]}", [{variants}]),')
    lines.append("]")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
md("""
# Spin BuAli — radiology pipeline benchmark (Kaggle, dual T4)

Runs Persian radiology dictations through STT → LLM and scores the **report**
that comes out against the radiologist's signed report.

**Before running:** Settings → Accelerator **GPU T4 ×2**, Internet **On**, and
attach the `spin-buali-dataset` dataset under *Add Input*.

Every function is defined in a cell below — read it, change it, re-run it.

---

**What gets scored, and why not word error rate.** These reports are templated:
two lines are identical across every case. Measured on this data, a *different
patient's* report scores WER 0.39–0.48 while a correctly reworded one scores
0.40 — they overlap. And flipping left for right, the error that sends a surgeon
to the wrong kidney, moves WER by **0.009**.

So WER is reported but not ranked on. The ranking is the clinical columns:
negation, laterality, number, unit, critical omissions, term F1.
""")

# ── 1. Setup ─────────────────────────────────────────────────────────────
md("## 1 — Setup")

code("""
!pip install -q jiwer sacrebleu bitsandbytes accelerate
""")

code("""
# ════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════
import gc, hashlib, io, json, os, re, time, unicodedata, warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CARDS = torch.cuda.device_count()

print(f"torch {torch.__version__}   device={DEVICE}   GPUs={CARDS}")
for i in range(CARDS):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.1f} GB")

# Usable VRAM per card, after the CUDA context and allocator fragmentation.
USABLE_GB = (min(torch.cuda.get_device_properties(i).total_memory
                 for i in range(CARDS)) / 1024**3 - 1.0) if CARDS else 0.0
print(f"\\nusable ~{USABLE_GB:.1f} GB per card, ~{USABLE_GB * max(CARDS,1):.1f} GB sharded")
""")

code("""
# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════
RESULTS_DIR = Path("/kaggle/working/results")     # the only dir Kaggle keeps
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000        # every model here wants 16 kHz mono
WINDOW_SEC = 28.0        # Whisper's encoder is fixed at 30 s
OVERLAP_SEC = 3.0        # so a word on a cut survives whole in one window
MAX_NEW_TOKENS = 1536

# A report worse than this counts as catastrophic in the summary.
CATASTROPHIC_WER = 1.0
""")

# ── 2. Dataset ───────────────────────────────────────────────────────────
md("""
## 2 — Dataset

The audio and its labels come from the attached Kaggle Dataset. One row per
recording: `asset_id`, `audio`, `image`, `report`. The `report` column is the
signed radiology report — that is the answer key.
""")

code('''
# ════════════════════════════════════════════════════════════════
#  LOCATE AND LOAD
# ════════════════════════════════════════════════════════════════
# Searched at any depth, because how deep the labels sit depends on how the
# dataset was zipped: a top-level folder inside the archive adds a level, and
# Kaggle keeps whatever was in there.
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]

# Set this if the search picks the wrong one, or you have several datasets.
LABELS_OVERRIDE = None    # e.g. "/kaggle/input/spin-buali-dataset/Small_Demo/labels.csv"


def find_labels(name="labels.csv", roots=None):
    """Every labels.csv under the search roots, shallowest first.

    Shallowest first because a dataset that also ships an example or a backup
    copy will have the real one nearest the top.
    """
    found = []
    for root in (roots or SEARCH_ROOTS):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() == name.lower():
                found.append(path)
    return sorted(set(found), key=lambda p: (len(p.parts), str(p)))


def show_what_is_there(root=Path("/kaggle/input"), max_depth=3, max_lines=60):
    """Print the mounted tree, so a failure says what IS there rather than
    leaving you to guess at the path."""
    if not root.exists():
        print(f"  {root} does not exist — is this running on Kaggle?")
        return
    base, shown = len(root.parts), 0
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - base
        if depth > max_depth:
            continue
        if shown >= max_lines:
            print("  ...")
            break
        print(f"  {'  ' * (depth - 1)}{path.name}{'/' if path.is_dir() else ''}")
        shown += 1


if LABELS_OVERRIDE:
    LABELS = Path(LABELS_OVERRIDE)
    assert LABELS.is_file(), f"LABELS_OVERRIDE does not exist: {LABELS}"
else:
    candidates = find_labels()
    if not candidates:
        print("No labels.csv found. This is what is actually mounted:\\n")
        show_what_is_there()
        print("\\nAttach the dataset with Add Input, or set LABELS_OVERRIDE above")
        print("to the full path of the labels.csv you can see in the tree.")
        raise SystemExit("dataset not found")
    if len(candidates) > 1:
        print(f"{len(candidates)} labels.csv found — using the first:")
        for c in candidates:
            print(f"   {c}")
        print()
    LABELS = candidates[0]

DATA_DIR = LABELS.parent
labels = pd.read_csv(LABELS)

print(f"labels : {LABELS}")
print(f"folder : {DATA_DIR}")
# A set, because a case-insensitive filesystem matches both patterns with the
# same files and would report double.
_audio = {p.resolve() for pattern in ("*.mp3", "*.MP3") for p in DATA_DIR.glob(pattern)}
print(f"audio  : {len(_audio)} mp3 file(s)")
print(f"rows   : {len(labels)}")
labels[["asset_id", "audio", "report"]].head()
''')

code("""
# ════════════════════════════════════════════════════════════════
#  DECODE THE AUDIO ONCE
# ════════════════════════════════════════════════════════════════
def load_audio(path, target_sr=TARGET_SR):
    \"\"\"Mono float32 at target_sr. Decoded once here rather than per model, so
    the comparison stays about inference and not about decoding.\"\"\"
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, target_sr).squeeze(0).numpy()
    return np.asarray(audio, np.float32), target_sr


clips = []
for _, row in labels.iterrows():
    audio, sr = load_audio(DATA_DIR / row["audio"])
    clips.append({
        "asset_id": row["asset_id"],
        "audio": audio,
        "sample_rate": sr,
        "duration_sec": len(audio) / sr,
        "reference": str(row["report"]),
    })

total = sum(c["duration_sec"] for c in clips)
print(f"{len(clips)} clips, {total/60:.1f} minutes of audio")
for c in clips:
    print(f"  {c['asset_id']:12} {c['duration_sec']:6.1f}s  "
          f"{len(c['reference'].split()):4} reference words")
""")

CLINICAL_TERMS_CELL = clinical_terms_literal() + '''


# Longest phrase wins, so "fatty liver" is not also counted as "liver".
TERM_INDEX = {}
for _cid, _cat, _variants in CONCEPTS:
    for _v in _variants:
        _phrase = tuple(normalize(_v).split())
        if _phrase:
            TERM_INDEX[_phrase] = (_cid, _cat)

MAX_PHRASE = max(len(p) for p in TERM_INDEX)
print(f"{len(CONCEPTS)} concepts, {len(TERM_INDEX)} phrases, "
      f"longest {MAX_PHRASE} words   (vocabulary {CLINICAL_TERMS_VERSION})")
'''

# ── 3. Metrics ───────────────────────────────────────────────────────────
md("""
## 3 — Scoring

Everything below is text processing: no model, no GPU. It is the same set of
metrics the project's evaluation service computes, written flat.

Two things drive the design:

**Normalise before comparing.** Persian digits, Arabic letter forms, ZWNJ and
spelled-out numbers all have to be folded together first, or `۶` and `6` count
as an error.

**Align before counting.** `6 mm stone` against `7 mm calculus` is *one* number
error — not one addition plus one omission. Counting raw text would inflate two
metrics while the one that should have caught it reads zero.
""")

md("### 3.1 — Persian / English text normalisation")

code(r'''
# ════════════════════════════════════════════════════════════════
#  NORMALISATION
# ════════════════════════════════════════════════════════════════
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Arabic letter forms that mean the same letter in Persian.
LETTER_FOLD = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا",
                             "إ": "ا", "آ": "ا", "ؤ": "و", "ئ": "ی"})

ZWNJ = "‌"          # نیم‌فاصله — a joining control, not a word boundary
DIACRITICS = re.compile(r"[ً-ْٰ]")
PUNCTUATION_RE = re.compile(r"""[.,;:?!()\-"'«»،؛؟]""")
STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")

SPELLED = {
    "یک": "1", "دو": "2", "سه": "3", "چهار": "4", "پنج": "5", "شش": "6",
    "هفت": "7", "هشت": "8", "نه": "9", "ده": "10",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize(text):
    """Fold everything that is a difference in spelling rather than meaning."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(PERSIAN_DIGITS).translate(LETTER_FOLD)
    text = DIACRITICS.sub("", text).replace(ZWNJ, "")
    # Two passes so a decimal point survives: strip punctuation, then any dot
    # that is not sitting between digits.
    text = PUNCTUATION_RE.sub(" ", text)
    text = STRAY_DOT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    """Normalised words, with spelled-out numbers turned into digits."""
    return [SPELLED.get(word, word) for word in normalize(text).split()]


print(tokenize("سنگ ۶ میلی‌متری در کلیه راست"))
print(tokenize("There is a six mm stone."))
''')

md("""
### 3.2 — The clinical vocabulary

One entry per concept, with every way it gets said. The Persian and English
variants share an id — which is what lets a Persian dictation and an English
report be compared concept by concept, even though word error rate cannot.

It is a seed: around forty concepts, scoped to abdominal and pelvic ultrasound
and built from real dictations. Real coverage needs RadLex plus frequency
mining over your own corpus.
""")

code(CLINICAL_TERMS_CELL)

md("""
### 3.3 — Extractors: measurements, negation, laterality

Three judgement calls worth knowing about:

**A bare number is not a measurement.** In Persian *یک* is both "one" and the
indefinite article, so counting every digit would invent measurements. A number
counts when it carries a unit, or belongs to a dimension group like `107 در 44`.

**Negation reads in both directions.** English puts the cue before the finding
("no stone"), Persian after it ("سنگ دیده نشد"). Checking one direction misses
half of them.

**Negation applies to findings, not anatomy.** A report says "no stone", never
"no kidney". Scoring anatomy for negation would just measure how far the cue
window happened to reach.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  MEASUREMENTS
# ════════════════════════════════════════════════════════════════
TO_MM = {"mm": 1.0, "میلیمتر": 1.0, "میلی": 1.0,
         "cm": 10.0, "سانتیمتر": 10.0, "سانتی": 10.0}
VOLUME_UNITS = {"cc", "ml", "سیسی"}
DIMENSION_JOINERS = {"x", "در", "*", "by"}


@dataclass(frozen=True)
class Measurement:
    value: float
    unit: str
    dimension: str          # length | volume

    @property
    def canonical(self):
        """In millimetres, so 10 mm and 1 cm compare equal."""
        return self.value * TO_MM.get(self.unit, 1.0)

    def same_quantity(self, other):
        return (self.dimension == other.dimension
                and abs(self.canonical - other.canonical) < 1e-6)

    def __str__(self):
        n = int(self.value) if float(self.value).is_integer() else self.value
        return f"{n} {self.unit}".strip()


def extract_measurements(text):
    """Every number that carries a unit, or sits in a dimension group."""
    tokens = tokenize(text)
    found = []
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        value = float(token)
        unit = next((t for t in tokens[index + 1:index + 3]
                     if t in TO_MM or t in VOLUME_UNITS), None)
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        before = tokens[index - 1] if index else ""
        grouped = after in DIMENSION_JOINERS or before in DIMENSION_JOINERS

        if unit in VOLUME_UNITS:
            found.append(Measurement(value, unit, "volume"))
        elif unit in TO_MM:
            found.append(Measurement(value, unit, "length"))
        elif grouped:
            found.append(Measurement(value, "", "length"))
    return found


# ════════════════════════════════════════════════════════════════
#  NEGATION AND LATERALITY
# ════════════════════════════════════════════════════════════════
PRE_NEGATION = {"no", "not", "without", "absent", "free", "بدون", "فاقد"}
POST_NEGATION = {"نشد", "نمیشود", "ندارد", "نیست", "نشده", "منفی"}
STOP_NEGATION = {"but", "however", "اما", "ولی"}
NEGATION_WINDOW = 6

LATERALITY = {"right": "right", "rt": "right", "راست": "right",
              "left": "left", "lt": "left", "چپ": "left",
              "bilateral": "bilateral", "both": "bilateral", "دوطرفه": "bilateral"}
LATERALITY_WINDOW = 4

# A report says "no stone", never "no kidney".
NEGATABLE = {"finding", "device"}


def is_negated(tokens, position):
    """A cue before (English) or after (Persian) the concept, inside a window
    that a contrast word closes."""
    for offset in range(1, NEGATION_WINDOW + 1):
        index = position - offset
        if index >= 0:
            if tokens[index] in STOP_NEGATION:
                break
            if tokens[index] in PRE_NEGATION:
                return True
    for offset in range(1, NEGATION_WINDOW + 1):
        index = position + offset
        if index < len(tokens):
            if tokens[index] in STOP_NEGATION:
                break
            if tokens[index] in POST_NEGATION:
                return True
    return False


def laterality_at(tokens, position):
    """The nearest side word on either flank, or None."""
    for offset in range(1, LATERALITY_WINDOW + 1):
        for index in (position - offset, position + offset):
            if 0 <= index < len(tokens) and tokens[index] in LATERALITY:
                return LATERALITY[tokens[index]]
    return None


@dataclass(frozen=True)
class Mention:
    concept_id: str
    category: str
    negated: bool
    laterality: object


def find_concepts(text):
    """Every concept in the vocabulary, with its negation and side.

    Longest phrase wins, so "fatty liver" is not also counted as "liver".
    """
    tokens = tokenize(text)
    mentions, position = [], 0
    while position < len(tokens):
        for length in range(min(MAX_PHRASE, len(tokens) - position), 0, -1):
            hit = TERM_INDEX.get(tuple(tokens[position:position + length]))
            if hit is None:
                continue
            concept_id, category = hit
            mentions.append(Mention(concept_id, category,
                                    is_negated(tokens, position),
                                    laterality_at(tokens, position)))
            position += length
            break
        else:
            position += 1
    return mentions


demo = "There is a 6 mm stone in the right kidney. No hydronephrosis."
print("measurements:", [str(m) for m in extract_measurements(demo)])
for m in find_concepts(demo):
    print(f"  {m.concept_id:22} {m.category:9} negated={m.negated!s:5} side={m.laterality}")
''')

md("### 3.4 — General metrics: WER, CER, chrF, and the failure detectors")

code(r'''
# ════════════════════════════════════════════════════════════════
#  EDIT DISTANCE, WITH THE OPERATION BREAKDOWN
# ════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int
    reference_length: int

    @property
    def errors(self):
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self):
        return self.errors / self.reference_length if self.reference_length else 0.0


def edit_counts(reference, hypothesis):
    """Levenshtein, keeping which operation each error was.

    Insertions are tracked separately because in a medical transcript they are
    the fabrication signal, not just noise.
    """
    rows, columns = len(reference), len(hypothesis)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distance[row][0] = row
    for column in range(columns + 1):
        distance[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if reference[row - 1] == hypothesis[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
            else:
                distance[row][column] = 1 + min(distance[row - 1][column - 1],
                                                distance[row][column - 1],
                                                distance[row - 1][column])
    subs = ins = dels = 0
    row, column = rows, columns
    while row > 0 or column > 0:
        if (row and column and reference[row - 1] == hypothesis[column - 1]
                and distance[row][column] == distance[row - 1][column - 1]):
            row, column = row - 1, column - 1
        elif row and column and distance[row][column] == distance[row - 1][column - 1] + 1:
            subs += 1
            row, column = row - 1, column - 1
        elif column and distance[row][column] == distance[row][column - 1] + 1:
            ins += 1
            column -= 1
        else:
            dels += 1
            row -= 1
    return EditCounts(subs, ins, dels, rows)


def word_errors(reference, hypothesis):
    return edit_counts(tokenize(reference), tokenize(hypothesis))


def char_errors(reference, hypothesis):
    strip = lambda t: [c for c in normalize(t) if not c.isspace()]
    return edit_counts(strip(reference), strip(hypothesis))


# ════════════════════════════════════════════════════════════════
#  chrF — character n-gram F-score, recall-weighted (sacreBLEU default)
# ════════════════════════════════════════════════════════════════
def chrf(reference, hypothesis, max_order=6, beta=2.0):
    """Partial credit where WER gives none: a morphological variant shares most
    of its characters, and WER calls it wholly wrong."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0

    precisions, recalls = [], []
    for order in range(1, max_order + 1):
        ref_grams = Counter(ref[i:i + order] for i in range(len(ref) - order + 1))
        hyp_grams = Counter(hyp[i:i + order] for i in range(len(hyp) - order + 1))
        overlap = sum((ref_grams & hyp_grams).values())
        if sum(hyp_grams.values()):
            precisions.append(overlap / sum(hyp_grams.values()))
        if sum(ref_grams.values()):
            recalls.append(overlap / sum(ref_grams.values()))
    if not precisions or not recalls:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


# ════════════════════════════════════════════════════════════════
#  THE FAILURE DETECTORS
# ════════════════════════════════════════════════════════════════
def repetition_score(text, n=4):
    """Share of repeated n-grams — the loop detector.

    This is the one that catches a model emitting one plausible sentence
    forever. Every clinical metric stays quiet on that, because the repeated
    content is individually correct.
    """
    words = tokenize(text)
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(v - 1 for v in counts.values() if v > 1) / len(grams)


def length_ratio(reference, hypothesis):
    """Output length over reference length. Catches fabrication built from
    words the clinical vocabulary has never heard of."""
    ref = len(tokenize(reference))
    return len(tokenize(hypothesis)) / ref if ref else 0.0


PUNCTUATION_SET = set(""".,;:?!()-"'«»،؛؟""")


def punctuation_f1(reference, hypothesis):
    """Computed on raw text, before normalisation strips it. Models that
    punctuate (Whisper) and models that emit bare text (Wav2Vec2, MMS) are
    otherwise not comparable on it at all."""
    ref = Counter(c for c in reference if c in PUNCTUATION_SET)
    hyp = Counter(c for c in hypothesis if c in PUNCTUATION_SET)
    if not ref and not hyp:
        return 1.0            # neither punctuates: agreement, not failure
    matched = sum((ref & hyp).values())
    precision = matched / sum(hyp.values()) if sum(hyp.values()) else 0.0
    recall = matched / sum(ref.values()) if sum(ref.values()) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


ARABIC_RANGES = [("؀", "ۿ"), ("ݐ", "ݿ"), ("ࡰ", "࢟"), ("ࢠ", "ࣿ"), ("ﭐ", "﷿"), ("ﹰ", "﻿")]


def script_contamination(text):
    """Share of letters that are not Arabic-script.

    Read this against the reference's own figure, never alone. These
    radiologists dictate English terms on purpose, so a perfectly correct
    transcript scores as heavily contaminated. The gap between the two is the
    signal; the number by itself is not.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for c in letters
                  if not any(lo <= c <= hi for lo, hi in ARABIC_RANGES))
    return foreign / len(letters)
''')

md("""
### 3.5 — The scorer

`score_report(hypothesis, reference)` is the whole public surface: two texts in,
one dict of numbers out.

Order matters. Nothing is counted until the entities are aligned, because
comparing raw text turns a single substitution into one addition *plus* one
omission — inflating two metrics while the one that should have caught it reads
zero.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  ALIGNMENT — pair things up before judging them
# ════════════════════════════════════════════════════════════════
def align_measurements(reference, hypothesis):
    """Exact physical matches first, so a correct-but-reordered measurement is
    never mistaken for an error. Whatever is left pairs up within its own
    dimension; leftovers are omissions and additions."""
    remaining = list(hypothesis)
    pairs, unmatched = [], []

    for measurement in reference:
        match = next((c for c in remaining if measurement.same_quantity(c)), None)
        if match is not None:
            remaining.remove(match)
            pairs.append((measurement, match))
        else:
            unmatched.append(measurement)

    still_missing = []
    for measurement in unmatched:
        match = next((c for c in remaining if c.dimension == measurement.dimension), None)
        if match is not None:
            remaining.remove(match)
            pairs.append((measurement, match))
        else:
            still_missing.append(measurement)
    return pairs, still_missing, remaining


def classify_mismatch(reference, hypothesis):
    """Same number, different unit is a misheard unit. Same unit, different
    number is a misheard value. Comparing physical quantities rather than unit
    strings means 10 mm and 1 cm never count as either."""
    if reference.same_quantity(hypothesis):
        return set()
    kinds = set()
    if reference.unit != hypothesis.unit:
        kinds.add("unit")
    if reference.value != hypothesis.value:
        kinds.add("number")
    return kinds or {"number"}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


# ════════════════════════════════════════════════════════════════
#  THE SCORER
# ════════════════════════════════════════════════════════════════
MAX_REPETITION = 0.5        # above this the model is looping
MAX_LENGTH_RATIO = 2.0      # twice the reference is padding, not transcription
MIN_LENGTH_RATIO = 0.5      # half of it is truncation


def score_report(hypothesis, reference):
    """Score one generated report against the signed one."""
    ref_concepts, hyp_concepts = {}, {}
    for mention in find_concepts(reference):
        ref_concepts.setdefault(mention.concept_id, mention)
    for mention in find_concepts(hypothesis):
        hyp_concepts.setdefault(mention.concept_id, mention)

    matched = set(ref_concepts) & set(hyp_concepts)
    only_reference = set(ref_concepts) - set(hyp_concepts)
    only_hypothesis = set(hyp_concepts) - set(ref_concepts)

    negation_errors = negation_scored = 0
    laterality_errors = laterality_scored = 0
    critical = []
    for concept_id in sorted(matched):
        expected, produced = ref_concepts[concept_id], hyp_concepts[concept_id]
        if expected.category in NEGATABLE:
            negation_scored += 1
            if expected.negated != produced.negated:
                negation_errors += 1
                critical.append({"type": "negation_flip", "concept": concept_id,
                                 "reference": "negative" if expected.negated else "positive",
                                 "prediction": "negative" if produced.negated else "positive"})
        if expected.laterality is not None:
            laterality_scored += 1
            if expected.laterality != produced.laterality:
                laterality_errors += 1
                critical.append({"type": "laterality_flip", "concept": concept_id,
                                 "reference": expected.laterality,
                                 "prediction": produced.laterality})

    ref_measures = extract_measurements(reference)
    hyp_measures = extract_measurements(hypothesis)
    pairs, missing, extra = align_measurements(ref_measures, hyp_measures)

    number_errors = unit_errors = 0
    for expected, produced in pairs:
        kinds = classify_mismatch(expected, produced)
        number_errors += "number" in kinds
        unit_errors += "unit" in kinds
        if kinds:
            critical.append({"type": "measurement_mismatch", "concept": "measurement",
                             "reference": str(expected), "prediction": str(produced)})
    for expected in missing:
        critical.append({"type": "measurement_omission", "concept": "measurement",
                         "reference": str(expected), "prediction": None})

    # An omission is critical when what went missing was a measurement, or a
    # concept the reference stated with a negation or a side. A structural rule,
    # not a hand-maintained list of important terms.
    dropped = [c for c in sorted(only_reference)
               if ref_concepts[c].negated or ref_concepts[c].laterality is not None]
    critical_omissions = len(missing) + len(dropped)
    unsupported_additions = len(extra) + len(only_hypothesis)

    true_positive = len(matched)
    false_positive = len(only_hypothesis)
    false_negative = len(only_reference)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    wer = word_errors(reference, hypothesis)
    cer = char_errors(reference, hypothesis)
    repetition = repetition_score(hypothesis)
    ratio = length_ratio(reference, hypothesis)

    # A degenerate output needs a human look even when every clinical counter
    # reads zero: a model repeating one plausible sentence forever produces no
    # negation, laterality or number error at all.
    reasons = [name for name, tripped in (
        ("repetition", repetition >= MAX_REPETITION),
        ("length_ratio", ratio >= MAX_LENGTH_RATIO
         or (reference.strip() and ratio <= MIN_LENGTH_RATIO)),
    ) if tripped]
    reasons += [name for name, count in (
        ("negation_errors", negation_errors),
        ("laterality_errors", laterality_errors),
        ("number_errors", number_errors),
        ("unit_errors", unit_errors),
        ("critical_omissions", critical_omissions),
        ("unsupported_additions", unsupported_additions),
    ) if count]

    return {
        # text
        "wer": round(wer.rate, 4),
        "cer": round(cer.rate, 4),
        "chrf": round(chrf(reference, hypothesis), 4),
        "substitutions": wer.substitutions,
        "insertions": wer.insertions,
        "deletions": wer.deletions,
        "reference_words": wer.reference_length,
        "hypothesis_words": len(tokenize(hypothesis)),
        "word_errors": wer.errors,
        "char_errors": cer.errors,
        "reference_chars": cer.reference_length,
        "repetition_score": round(repetition, 4),
        "length_ratio": round(ratio, 4),
        "punctuation_f1": round(punctuation_f1(reference, hypothesis), 4),
        "script_contamination": round(script_contamination(hypothesis), 4),
        "reference_script_contamination": round(script_contamination(reference), 4),
        # clinical counts, and the denominators each rate needs
        "reference_entities": len(ref_concepts),
        "true_positive_terms": true_positive,
        "false_positive_terms": false_positive,
        "false_negative_terms": false_negative,
        "reference_measurements": len(ref_measures),
        "negation_errors": negation_errors,
        "negation_scored": negation_scored,
        "laterality_errors": laterality_errors,
        "laterality_scored": laterality_scored,
        "number_errors": number_errors,
        "unit_errors": unit_errors,
        "critical_omissions": critical_omissions,
        "critical_omission_scored": len(ref_measures) + len(dropped),
        "unsupported_additions": unsupported_additions,
        "unsupported_addition_scored": len(hyp_measures) + len(hyp_concepts),
        # clinical rates, per report
        "medical_term_precision": round(precision, 4),
        "medical_term_recall": round(recall, 4),
        "medical_term_f1": round(f1, 4),
        "negation_error_rate": round(_ratio(negation_errors, negation_scored), 4),
        "laterality_error_rate": round(_ratio(laterality_errors, laterality_scored), 4),
        "number_error_rate": round(_ratio(number_errors, len(ref_measures)), 4),
        "unit_error_rate": round(_ratio(unit_errors, len(ref_measures)), 4),
        # verdict
        "requires_medical_review": bool(reasons),
        "review_reasons": ";".join(reasons),
        "critical_errors": len(critical),
        "critical_detail": json.dumps(critical, ensure_ascii=False),
    }
''')

md("""
Check it on a case where the answer is known. The middle row is the one that
matters: a flipped side barely moves WER, and the laterality counter catches it.
""")

code(r'''
_ref = "There is a 6 mm stone in the distal right ureter. No hydronephrosis."
for _label, _hyp in [
    ("identical", _ref),
    ("left/right flipped", _ref.replace("right", "left")),
    ("negation dropped", _ref.replace("No hydronephrosis", "Hydronephrosis is present")),
    ("6 mm -> 7 mm", _ref.replace("6 mm", "7 mm")),
    ("empty", ""),
]:
    s = score_report(_hyp, _ref)
    print(f"  {_label:20} WER={s['wer']:.3f}  termF1={s['medical_term_f1']:.2f}  "
          f"lat={s['laterality_errors']}  neg={s['negation_errors']}  "
          f"num={s['number_errors']}  review={s['requires_medical_review']}")
''')

# ── 4. Models ────────────────────────────────────────────────────────────
md("""
## 4 — Models

### 4.1 — Speech recognition

One class per architecture, `load` / `transcribe` / `unload`. Add a model by
adding a row to `STT_REGISTRY`.

**Windowing is not optional.** Whisper's encoder takes a fixed 30-second
window. Hand it a four-minute dictation and it transcribes the first thirty
seconds and stops — which reads as a catastrophic word error rate caused by the
model, when it was caused by the harness. Every model gets the same 28-second
windows with 3 seconds of overlap, and the overlap is de-duplicated at the seam.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  WINDOWING AND STITCHING
# ════════════════════════════════════════════════════════════════
def plan_windows(duration, window=WINDOW_SEC, overlap=OVERLAP_SEC):
    """Overlapping (start, end) windows covering the whole recording.

    The overlap is what makes stitching possible: a word landing on a cut is
    spoken fully inside one of the two neighbours.
    """
    if duration <= window:
        return [(0.0, duration)]
    step, windows, start = window - overlap, [], 0.0
    while start < duration:
        end = min(start + window, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def stitch(parts, max_overlap_words=40):
    """Join the windows, dropping what the overlap said twice.

    Where one window ends with the same words the next begins with, that is the
    shared audio transcribed once each. No match means the two disagreed about
    the overlap — keeping both is the safer error, because a duplication is
    visible to a reader and an omission is not.
    """
    merged = []
    for part in parts:
        words = part.split()
        if not words:
            continue
        if not merged:
            merged = words
            continue
        seam = 0
        for length in range(min(max_overlap_words, len(merged), len(words)), 0, -1):
            if merged[-length:] == words[:length]:
                seam = length
                break
        merged.extend(words[seam:])
    return " ".join(merged)


# ════════════════════════════════════════════════════════════════
#  STT MODEL CLASSES
# ════════════════════════════════════════════════════════════════
WHISPER_LANGUAGE = {"fa": "persian", "en": "english"}
SEAMLESS_LANGUAGE = {"fa": "pes", "en": "eng"}


class WhisperSTT:
    def __init__(self, model_id, **_):
        self.model_id = model_id

    def load(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = (WhisperForConditionalGeneration
                      .from_pretrained(self.model_id, torch_dtype=dtype)
                      .to(DEVICE).eval())
        self.model.generation_config.forced_decoder_ids = None
        return self

    def transcribe(self, audio, sr, language="fa"):
        features = self.processor(audio, sampling_rate=sr,
                                  return_tensors="pt").input_features.to(DEVICE)
        if DEVICE == "cuda":
            features = features.half()
        forced = (self.processor.get_decoder_prompt_ids(
            language=WHISPER_LANGUAGE.get(language, language), task="transcribe")
            if language else None)
        with torch.no_grad():
            ids = self.model.generate(features, forced_decoder_ids=forced)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def unload(self):
        self.model = self.processor = None
        free_gpu()


class SeamlessSTT:
    def __init__(self, model_id, version=2, target_lang="pes", **_):
        self.model_id, self.version, self.target_lang = model_id, version, target_lang

    def load(self):
        from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model
        cls = SeamlessM4Tv2Model if self.version == 2 else SeamlessM4TModel
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = cls.from_pretrained(self.model_id).to(DEVICE).eval()
        return self

    def transcribe(self, audio, sr, language="fa"):
        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        target = SEAMLESS_LANGUAGE.get(language, self.target_lang)
        with torch.no_grad():
            out = self.model.generate(**inputs, tgt_lang=target, generate_speech=False)
        if self.version == 2:
            sequences = out.sequences if hasattr(out, "sequences") else out
            return self.processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        tokens = out[0] if isinstance(out, (list, tuple)) else out
        return self.processor.decode(tokens.squeeze().tolist(), skip_special_tokens=True)

    def unload(self):
        self.model = self.processor = None
        free_gpu()


class CTCSTT:
    """wav2vec2 and MMS. MMS ships one backbone with per-language adapters, so
    it needs the adapter loaded before use; wav2vec2 does not."""

    def __init__(self, model_id, adapter=None, **_):
        self.model_id, self.adapter = model_id, adapter

    def load(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        kwargs = {"target_lang": self.adapter} if self.adapter else {}
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, **kwargs)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, **kwargs).to(DEVICE).eval()
        if self.adapter:
            self.model.load_adapter(self.adapter)
        return self

    def transcribe(self, audio, sr, language="fa"):
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(DEVICE)).logits
        return self.processor.batch_decode(torch.argmax(logits, dim=-1))[0]

    def unload(self):
        self.model = self.processor = None
        free_gpu()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


STT_REGISTRY = {
    "whisper-persian-v4":     (WhisperSTT,  {"model_id": "nezamisafa/whisper-persian-v4"}),
    "whisper-large-v3":       (WhisperSTT,  {"model_id": "openai/whisper-large-v3"}),
    "whisper-large-v3-turbo": (WhisperSTT,  {"model_id": "openai/whisper-large-v3-turbo"}),
    "whisper-halakoo":        (WhisperSTT,  {"model_id": "MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch"}),
    "whisper-vhdm":           (WhisperSTT,  {"model_id": "vhdm/whisper-large-fa-v1"}),
    "seamless-v2-large":      (SeamlessSTT, {"model_id": "facebook/seamless-m4t-v2-large", "version": 2}),
    "seamless-medium":        (SeamlessSTT, {"model_id": "facebook/hf-seamless-m4t-medium", "version": 1}),
    "mms-1b-all":             (CTCSTT,      {"model_id": "facebook/mms-1b-all", "adapter": "fas"}),
    "mms-1b-fl102":           (CTCSTT,      {"model_id": "facebook/mms-1b-fl102", "adapter": "fas"}),
    "wav2vec2-xlsr53":        (CTCSTT,      {"model_id": "jonatasgrosman/wav2vec2-large-xlsr-53-persian"}),
}


def transcribe_clips(model_name, clips, language="fa"):
    """Load one STT model, transcribe every clip in windows, unload it."""
    cls, kwargs = STT_REGISTRY[model_name]
    model = cls(**kwargs)
    started = time.perf_counter()
    model.load()
    load_seconds = time.perf_counter() - started

    out = {}
    for clip in clips:
        began = time.perf_counter()
        try:
            windows = plan_windows(clip["duration_sec"])
            sr = clip["sample_rate"]
            parts = [model.transcribe(clip["audio"][int(s * sr):int(e * sr)], sr, language)
                     for s, e in windows]
            text, error = stitch(parts), None
        except Exception as exc:
            text, error = "", f"{type(exc).__name__}: {exc}"
        out[clip["asset_id"]] = {
            "text": text, "error": error, "windows": len(plan_windows(clip["duration_sec"])),
            "seconds": round(time.perf_counter() - began, 2),
        }
        print(f"    {clip['asset_id']:12} {out[clip['asset_id']]['seconds']:6.1f}s"
              f"{'  ' + error if error else ''}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    model.unload()
    return out, {"load_seconds": round(load_seconds, 1), "peak_vram_gb": round(peak, 2)}
''')

md("""
### 4.2 — Language models, and what this hardware can hold

Each model is placed in the **highest precision the cards can hold** — fp16 if
it fits, then int8, then 4-bit. Precision is given up reluctantly, not by
default.

That produces three tiers:

| | | |
|---|---|---|
| **A** | native | fits unquantized — the number means what it says |
| **B** | quantized | the score includes whatever the compression cost |
| **C** | deferred | does not fit here at any useful precision |

Tier C is not a list of failures. It is the **same models as tier B at full
precision**, kept in the table so the gap is visible: run one on hardware that
can hold it and the difference against tier B *is* the quantization penalty.
Without it, a quantized 32B and a native 8B differ two ways at once and the
leaderboard cannot say which one moved the score.

**One caution.** Turing has no bf16, and Gemma and Qwen are bf16-native. They
are loaded fp16 here, which can overflow. If one of them produces nonsense,
that is the first thing to suspect.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  LLM REGISTRY AND PLACEMENT
# ════════════════════════════════════════════════════════════════
# params in billions, and whether the model can hear audio directly
LLM_REGISTRY = {
    "aya-expanse-8b":  ("CohereLabs/aya-expanse-8b",            8.03, False),
    "aya-expanse-32b": ("CohereLabs/aya-expanse-32b",          32.3,  False),
    "gemma-4-31b":     ("google/gemma-4-31B-it",               31.0,  False),
    "gemma-4-e4b":     ("google/gemma-4-E4B-it",                7.85, True),
    "gemma-4-12b":     ("google/gemma-4-12B-it",               12.0,  True),
    "qwen3-omni-30b":  ("Qwen/Qwen3-Omni-30B-A3B-Instruct",    30.5,  True),
}

BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "nf4": 0.5}
RUNTIME_OVERHEAD = 1.15     # activations, KV cache, processor buffers

# Tried in order: keep precision where possible, and prefer one card over two
# (it leaves the other free, and sharding pays a PCIe cost per forward pass).
LADDER = [("fp16", 1), ("fp16", 2), ("int8", 1), ("int8", 2), ("nf4", 1), ("nf4", 2)]


def estimate_gb(params_b, precision):
    return params_b * BYTES_PER_PARAM[precision] * RUNTIME_OVERHEAD


def place(name, usable_gb=None, cards=None):
    """The highest-precision placement that fits, or tier C."""
    usable_gb = USABLE_GB if usable_gb is None else usable_gb
    cards = CARDS if cards is None else cards
    _, params, _ = LLM_REGISTRY[name]
    native = estimate_gb(params, "fp16")
    for precision, needed in LADDER:
        if needed > max(cards, 1):
            continue
        size = estimate_gb(params, precision)
        if size <= usable_gb * needed:
            return {"model": name, "tier": "A" if precision == "fp16" else "B",
                    "precision": precision, "cards": needed,
                    "vram_gb": round(size, 1), "native_gb": round(native, 1)}
    return {"model": name, "tier": "C", "precision": None, "cards": 0,
            "vram_gb": 0.0, "native_gb": round(native, 1)}


PLACEMENTS = {name: place(name) for name in LLM_REGISTRY}

print(f"{'model':18} {'tier':5} {'placement':22} {'needs native'}")
for name, p in PLACEMENTS.items():
    where = (f"{p['precision']}, {p['cards']} card{'s' if p['cards'] > 1 else ''}"
             if p["tier"] != "C" else "does not fit here")
    print(f"  {name:18} {p['tier']:5} {where:22} {p['native_gb']:5.0f} GB")
''')

code(r'''
# ════════════════════════════════════════════════════════════════
#  LOADING ONE LANGUAGE MODEL
# ════════════════════════════════════════════════════════════════
class LLM:
    """Loaded at the precision its tier decided. Mirrors the STT classes:
    load / generate / unload."""

    def __init__(self, name):
        self.name = name
        self.model_id, self.params_b, self.audio_capable = LLM_REGISTRY[name]
        placement = PLACEMENTS[name]
        self.precision, self.cards = placement["precision"], placement["cards"]
        self.load_seconds = 0.0

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        started = time.perf_counter()

        options = {"device_map": "auto" if self.cards > 1 else 0}
        if self.precision == "fp16":
            options["dtype"] = torch.float16
        else:
            from transformers import BitsAndBytesConfig
            if self.precision == "int8":
                options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    # Saves ~0.4 bits/weight — the difference between fitting
                    # and not, for the 30B tier.
                    bnb_4bit_use_double_quant=True,
                    # Has to be fp16: Turing has no bf16.
                    bnb_4bit_compute_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **options).eval()
        self.load_seconds = time.perf_counter() - started
        return self

    def generate(self, system_prompt, user_text):
        """Only the newly generated tokens are decoded — several of these
        models echo the prompt, and a report that opens with its own
        instructions is not a report."""
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text or ""}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False)
        fresh = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(fresh, skip_special_tokens=True)

    def unload(self):
        self.model = self.tokenizer = None
        free_gpu()
''')

# ── 5. Pipeline ──────────────────────────────────────────────────────────
md("""
## 5 — The pipeline: transcripts → report

Three pipelines, differing only in what the single LLM call is given:

| | STT transcripts | audio to the LLM |
|---|---|---|
| `separate` | ✅ up to 3 engines, reconciled | ❌ |
| `multimodal` | ❌ | ✅ |
| `hybrid` | ✅ as cross-check material | ✅ |

These prompts are the ones the production controller uses. That is not
incidental — **the prompt is the experiment.** A benchmark that phrased the
instruction its own way would be ranking a system nobody ships, and nothing in
the results would say so.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  PROMPTS  (verbatim from controller/prompts.py)
# ════════════════════════════════════════════════════════════════
REPORT_TEMPLATE = {
    "raw_transcript": "<the transcript exactly as heard, uncorrected>",
    "corrected_transcript": "<the same content with transcription errors fixed>",
    "final_text": "<the finished radiology report>",
    "discrepancies_found": ["<each disagreement between sources, or omit>"],
    "notes": "<anything a radiologist should know, or null>",
}

RECONCILE = (
    "You are a radiology transcription QA assistant. You will be given up to three independent "
    "transcriptions of the SAME spoken radiology report, each produced by a different speech-to-text "
    "engine (local or cloud) -- not all three may be present. They may disagree in places due to "
    "transcription errors. Produce THREE outputs, each strictly grounded in what the transcripts "
    "actually say: do not invent findings, measurements, or laterality that no transcript supports."
)

TRANSCRIBE_FROM_AUDIO = (
    "You are a radiology transcription assistant. You are given an audio recording of a SPOKEN "
    "radiology report -- listen to it directly. You may ALSO be given one or more reference "
    "transcripts of the same audio, produced separately by other speech-to-text engines -- if so, "
    "treat them only as supporting evidence to cross-check against, NOT ground truth (they may "
    "contain errors). Do not invent findings, measurements, or laterality the audio does not support."
)


def with_template(prompt):
    """Append the JSON template the model is asked to fill in."""
    return (prompt + "\n\nJSON template to fill:\n"
            + json.dumps(REPORT_TEMPLATE, ensure_ascii=False, indent=2))


def extract_json(reply):
    """Pull the report object out of the reply, tolerating code fences and a
    model that explains itself before answering. They all do."""
    text = (reply or "").strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the LLM reply")
    return json.loads(text[start:end + 1])


# ════════════════════════════════════════════════════════════════
#  BUILDING ONE REPORT
# ════════════════════════════════════════════════════════════════
def build_report(transcripts, llm, pipeline):
    """One LLM call, parsed into a report. Returns (final_text, error)."""
    ordered = sorted(transcripts.items(),
                     key=lambda kv: int(kv[0].removeprefix("transcript_")))
    try:
        if pipeline == "separate":
            if not ordered:
                raise ValueError("separate needs at least one transcript to reconcile")
            system = with_template(RECONCILE)
            user = "\n\n".join(
                f"STT engine {key.removeprefix('transcript_')} transcript:\n{text}"
                for key, text in ordered)
        else:
            system = with_template(TRANSCRIBE_FROM_AUDIO)
            user = None
            if ordered:
                user = "\n\n".join(
                    f"Reference transcript {i} (from a separate STT engine — "
                    f"may contain errors):\n{text}"
                    for i, (_, text) in enumerate(ordered, start=1))

        parsed = extract_json(llm.generate(system, user))
        final = parsed.get("final_text") or ""
        return final, (None if final else "the model returned no final_text")
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
''')

# ── 6. The runner ────────────────────────────────────────────────────────
md("""
## 6 — The runner

`run_configuration(...)` is the equivalent of the previous notebook's
`evaluate_model`: it does one whole configuration, writes its own CSV, and
returns the DataFrame.

Two things it does that matter for a Kaggle session:

**Each run writes its CSV before the next begins.** Stop the session whenever
you like — nothing is in flight, so nothing is lost. Re-run the cell later and
finished runs are skipped, because *a run is finished when its CSV exists*.

**Transcripts are cached.** Speech recognition is the expensive stage and does
not change when the prompt or the language model does. Trying three LLMs over
the same engine costs three LLM passes and one transcription.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  THE RUNNER
# ════════════════════════════════════════════════════════════════
TRANSCRIPT_CACHE = RESULTS_DIR / "transcripts"
TRANSCRIPT_CACHE.mkdir(exist_ok=True)


def run_id(config):
    """A stable id for one configuration, so resuming is a matter of asking
    which ids already have a file."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def get_transcripts(stt_models, clips, language="fa"):
    """Transcribe with each engine, reusing anything already on disk."""
    per_clip = {clip["asset_id"]: {} for clip in clips}
    speed = {}
    for position, model_name in enumerate(stt_models, start=1):
        cache = TRANSCRIPT_CACHE / f"{model_name}.json"
        if cache.exists():
            print(f"  [cached] {model_name}")
            stored = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"  [stt] {model_name}")
            stored, speed[model_name] = transcribe_clips(model_name, clips, language)
            cache.write_text(json.dumps(stored, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        for asset_id, entry in stored.items():
            if asset_id in per_clip:
                per_clip[asset_id][f"transcript_{position}"] = entry["text"]
    return per_clip, speed


def run_configuration(stt_models, llm_name, pipeline, clips, language="fa", notes=""):
    """One configuration end to end: transcribe, generate, score, save."""
    config = {"stt_models": list(stt_models), "llm": llm_name, "pipeline": pipeline}
    identifier = run_id(config)
    csv_path = RESULTS_DIR / f"run__{identifier}.csv"

    if csv_path.exists():
        print(f"[skip] {identifier}  already done -> {csv_path.name}")
        return pd.read_csv(csv_path)

    placement = PLACEMENTS[llm_name]
    label = "+".join(stt_models) or "(none)"
    print(f"\n[run] {identifier}  {pipeline}  stt={label}  llm={llm_name} "
          f"({placement['tier']}, {placement['precision']})")

    transcripts, stt_speed = get_transcripts(stt_models, clips, language)

    llm = LLM(llm_name)
    try:
        llm.load()
    except Exception as exc:
        print(f"  [FAILED] could not load {llm_name}: {exc}")
        row = {**config, "run_id": identifier, "tier": placement["tier"],
               "precision": placement["precision"], "status": "llm_load_failed",
               "error": str(exc)[:400]}
        pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return pd.DataFrame([row])

    rows = []
    for clip in clips:
        began = time.perf_counter()
        final_text, error = build_report(transcripts[clip["asset_id"]], llm, pipeline)
        scored = score_report(final_text, clip["reference"])
        rows.append({
            "run_id": identifier,
            "asset_id": clip["asset_id"],
            "stt_models": label,
            "llm": llm_name,
            "pipeline": pipeline,
            "tier": placement["tier"],
            "precision": placement["precision"],
            "duration_sec": round(clip["duration_sec"], 2),
            "llm_seconds": round(time.perf_counter() - began, 2),
            "status": "error" if error else "ok",
            "error": error or "",
            **scored,
            "reference": clip["reference"],
            "hypothesis": final_text,
        })
        print(f"  {clip['asset_id']:12} WER={scored['wer']:.3f} "
              f"F1={scored['medical_term_f1']:.2f} "
              f"lat={scored['laterality_errors']} neg={scored['negation_errors']}"
              f"{'  ' + error if error else ''}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    llm.unload()

    frame = pd.DataFrame(rows)
    frame["llm_load_seconds"] = round(llm.load_seconds, 1)
    frame["peak_vram_gb"] = round(peak, 2)
    frame["notes"] = notes
    # utf-8-sig so Excel reads the Persian; written last, because its existence
    # is what marks the run finished.
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  saved -> {csv_path.name}   corpus WER "
          f"{frame['word_errors'].sum() / max(frame['reference_words'].sum(), 1):.3f}")
    return frame
''')

# ── 7. Runs ──────────────────────────────────────────────────────────────
md("""
## 7 — Tier A: models that fit unquantized

Start small. **Run one configuration first** and confirm the model loads and
writes a CSV before widening — a failure here costs two minutes, the same
failure inside a twenty-run loop costs the session.
""")

code(r'''
# One configuration, to prove the whole path works.
first = run_configuration(
    stt_models=["whisper-persian-v4"],
    llm_name="aya-expanse-8b",
    pipeline="separate",
    clips=clips,
)
first[["asset_id", "wer", "medical_term_f1", "laterality_errors",
       "negation_errors", "number_errors", "requires_medical_review"]]
''')

md("""
Now the rest of tier A. Add or remove lines freely — anything already run is
skipped, so re-running this cell after a stopped session continues where it
left off.
""")

code(r'''
TIER_A = [
    # (stt engines, llm, pipeline)
    (["whisper-persian-v4"],       "aya-expanse-8b", "separate"),
    (["whisper-large-v3"],         "aya-expanse-8b", "separate"),
    (["whisper-large-v3-turbo"],   "aya-expanse-8b", "separate"),
    (["seamless-v2-large"],        "aya-expanse-8b", "separate"),
    (["wav2vec2-xlsr53"],          "aya-expanse-8b", "separate"),
    # two engines reconciled against each other
    (["whisper-persian-v4", "seamless-v2-large"], "aya-expanse-8b", "separate"),
    # no STT at all — the model hears the recording itself
    ([],                           "gemma-4-e4b",    "multimodal"),
]

SESSION_MINUTES = 300      # stop starting runs after this long
started_at = time.time()

for stt_models, llm_name, pipeline in TIER_A:
    if PLACEMENTS[llm_name]["tier"] != "A":
        print(f"[skip] {llm_name} is tier {PLACEMENTS[llm_name]['tier']}, not A")
        continue
    elapsed_min = (time.time() - started_at) / 60
    if elapsed_min > SESSION_MINUTES:
        print(f"\nstopping: {elapsed_min:.0f} min spent, budget was {SESSION_MINUTES}")
        print("everything finished so far is saved — re-run this cell next session")
        break
    run_configuration(stt_models, llm_name, pipeline, clips)
''')

md("""
## 8 — Tier B: quantized to fit

Same runs, models that only fit compressed. Their scores include whatever the
quantization cost — which is the thing tier C exists to measure.
""")

code(r'''
TIER_B = [
    (["whisper-persian-v4"], "aya-expanse-32b", "separate"),
    (["whisper-persian-v4"], "gemma-4-31b",     "separate"),
    ([],                     "qwen3-omni-30b",  "multimodal"),
]

for stt_models, llm_name, pipeline in TIER_B:
    if PLACEMENTS[llm_name]["tier"] != "B":
        print(f"[skip] {llm_name} is tier {PLACEMENTS[llm_name]['tier']}, not B")
        continue
    run_configuration(stt_models, llm_name, pipeline, clips)
''')

md("""
## 9 — Tier C: what this hardware cannot hold

Not run here. Listed so the gap is visible in the results rather than being an
absence nobody notices — and so the same configuration can be run on a bigger
card and compared directly against its tier-B twin.
""")

code(r'''
deferred = [(name, p) for name, p in PLACEMENTS.items() if p["tier"] == "C"]
if not deferred:
    print("nothing deferred — every model fits on this hardware")
for name, p in deferred:
    print(f"  {name:18} needs ~{p['native_gb']:.0f} GB unquantized "
          f"(an A100 80GB or an H100)")
    print(f"                     compare against its tier-B run at "
          f"{PLACEMENTS[name]['precision'] or 'nf4'} to price the quantization")
''')

# ── 10. Results ──────────────────────────────────────────────────────────
md("""
## 10 — Master results

Rebuilt from the CSVs on disk, so it always reflects the runs that actually
finished — across however many sessions that took.

**Rates come from summed counts, never from averaging per-report rates.** One
negation error in a two-sentence report is a 100% rate; averaged in, it would
drown out a hundred correct ones.
""")

code(r'''
# ════════════════════════════════════════════════════════════════
#  MERGE EVERY RUN
# ════════════════════════════════════════════════════════════════
frames = [pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("run__*.csv"))]
reports = pd.concat([f for f in frames if "asset_id" in f.columns], ignore_index=True)
reports.to_csv(RESULTS_DIR / "all_reports.csv", index=False, encoding="utf-8-sig")
print(f"{len(reports)} scored report(s) from {reports['run_id'].nunique()} run(s)")


def summarize(group):
    """One row per configuration, with every rate rebuilt from summed counts."""
    total = lambda c: group[c].sum()
    ratio = lambda a, b: round(total(a) / total(b), 4) if total(b) else 0.0
    wers = sorted(group["wer"])
    percentile = lambda f: round(wers[min(int(f * len(wers)), len(wers) - 1)], 4)
    tp = total("true_positive_terms")
    fp, fn = total("false_positive_terms"), total("false_negative_terms")
    precision = round(tp / (tp + fp), 4) if tp + fp else 0.0
    recall = round(tp / (tp + fn), 4) if tp + fn else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0

    return pd.Series({
        "reports": len(group),
        "corpus_wer": ratio("word_errors", "reference_words"),
        "corpus_cer": ratio("char_errors", "reference_chars"),
        "wer_p50": percentile(0.50),
        "wer_p90": percentile(0.90),
        "chrf": round(group["chrf"].mean(), 4),
        "medical_term_f1": f1,
        "negation_error_rate": ratio("negation_errors", "negation_scored"),
        "laterality_error_rate": ratio("laterality_errors", "laterality_scored"),
        "number_error_rate": ratio("number_errors", "reference_measurements"),
        "unit_error_rate": ratio("unit_errors", "reference_measurements"),
        "critical_omission_rate": ratio("critical_omissions", "critical_omission_scored"),
        "unsupported_addition_rate": ratio("unsupported_additions", "unsupported_addition_scored"),
        "repetition_p95": round(group["repetition_score"].quantile(0.95), 4),
        "latin_share": round(group["script_contamination"].mean(), 4),
        "latin_share_reference": round(group["reference_script_contamination"].mean(), 4),
        "review_rate": round(group["requires_medical_review"].mean(), 4),
        "pct_catastrophic": round((group["wer"] >= CATASTROPHIC_WER).mean(), 4),
        "failures": int((group["status"] != "ok").sum()),
        "llm_seconds_total": round(group["llm_seconds"].sum(), 1),
        "peak_vram_gb": group["peak_vram_gb"].max() if "peak_vram_gb" in group else 0.0,
    })


board = (reports.groupby(["tier", "pipeline", "stt_models", "llm", "precision"])
         .apply(summarize).reset_index())
board = board.sort_values(["laterality_error_rate", "negation_error_rate", "corpus_wer"])
board.to_csv(RESULTS_DIR / "leaderboard.csv", index=False, encoding="utf-8-sig")
board
''')

md("""
### Read it in this order

1. **Laterality and negation errors** — a flipped side or a dropped "no" changes
   what the report *means*. These are the rows that would matter clinically.
2. **Number and unit errors** — a 6 mm stone reported as 6 cm is a tenfold error.
3. **Critical omissions** — a measurement or a sided finding that vanished.
4. **`latin_share` against `latin_share_reference`** — read the *gap*. These
   radiologists dictate English terms deliberately, so a correct report is
   "contaminated" too; a model drifting into English shows as the two diverging.
5. **`corpus_wer`** last, and only as a rough "how much rewriting is left".
""")

md("""
## 11 — What went wrong

An aggregate says a configuration is 12% wrong; it never says *which* 12%.
These are the reports to read before trusting anything above.
""")

code(r'''
worst = reports.sort_values("wer", ascending=False).head(15)
worst[["asset_id", "llm", "stt_models", "pipeline", "wer", "medical_term_f1",
       "laterality_errors", "negation_errors", "review_reasons", "error"]]
''')

code(r'''
flagged = reports[reports["requires_medical_review"] == True]
print(f"{len(flagged)} of {len(reports)} reports need a human look\n")
Counter(r for reasons in flagged["review_reasons"].fillna("")
        for r in str(reasons).split(";") if r).most_common()
''')

md("""
## 12 — One recording, side by side

Critical errors are the ones that change meaning: a flipped negation, a wrong
side, a wrong number.
""")

code(r'''
ASSET = reports.iloc[0]["asset_id"]
rows = reports[reports["asset_id"] == ASSET]

print("REFERENCE")
print(rows.iloc[0]["reference"])
print()
for _, row in rows.iterrows():
    print(f"--- {row['llm']} / {row['stt_models']} / {row['pipeline']}   "
          f"WER={row['wer']:.3f}  F1={row['medical_term_f1']:.2f}")
    print(row["hypothesis"])
    for issue in json.loads(row.get("critical_detail") or "[]"):
        print(f"    ! {issue['type']}: {issue['reference']} -> {issue['prediction']}")
    print()
''')

md("""
## 13 — Save

Everything is already on disk — each run wrote its CSV as it finished. Download
the output folder before the session ends.

| File | |
|---|---|
| `run__<id>.csv` | one configuration's per-report scores |
| `all_reports.csv` | every scored report from every run |
| `leaderboard.csv` | one row per configuration |
| `transcripts/*.json` | the STT cache — **keep it**, and the next session skips transcription entirely |
""")

code(r'''
for path in sorted(RESULTS_DIR.rglob("*")):
    if path.is_file():
        print(f"  {str(path.relative_to(RESULTS_DIR)):40} {path.stat().st_size/1024:8.1f} KB")
''')


def main():
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  {len(cells)} cells ({code_cells} code), {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
