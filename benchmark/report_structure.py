"""The section order the reference reports follow, as a prompt addendum.

Deliberately separate from `controller/prompts.py`. This benchmark scores the
generated report against a *specific* labelled dataset whose reports all follow
one house template; the production controller has no such dataset and must
stay generic. Mixing the two would mean either the benchmark quietly diverges
from what ships, or production starts assuming a template that only this one
customer's reports happen to use. So the guide lives here, is passed in
explicitly by `runner.py`, and is never silently on by default.

The order below was read off the nine labelled reports in
`Spin_BuAli_DataSet/Small_Demo/labels.csv` (see `derive_from_reports` for how),
not invented: every one of them states the organs in this sequence, and two
lines -- the portosplenic vein sentence and the retroperitoneal closing
sentence -- appear verbatim in all nine.
"""
import re

# The order every one of the nine reference reports states its findings in.
# Not every section appears in every report (Uterus/ovaries only when no
# prostate is described, and vice versa), so the guide says "in this order,
# skip what does not apply" rather than demanding every line.
SECTION_ORDER = [
    "Liver",
    "Hepatic and portosplenic venous system",
    "Intra and extra hepatic biliary ducts",
    "Gall bladder",
    "Splenic size and echotexture",
    "Pancreas",
    "Para-aortic region / lymph nodes",
    "Both kidneys (size, contour, echotexture)",
    "Kidney findings (stones, hydronephrosis, masses, cysts) -- one line per kidney",
    "Urinary bladder",
    "Ascites / free fluid",
    "Uterus and ovaries, OR Prostate and seminal vesicles (whichever applies)",
    "Retroperitoneal space -- closing statement",
]

# The two sentences that appear, close to verbatim, in every reference report.
# Naming them explicitly is cheaper than describing the pattern abstractly.
BOILERPLATE_ANCHORS = [
    "Hepatic and portosplenic venous system have normal diameter.",
    "There is no definite sign of mass or collection in visualized part of retroperitoneal space.",
]

GUIDE = (
    "Benchmark note (not a clinical instruction): the reports this system is "
    "compared against all follow one house template, stating findings in this "
    "order --\n"
    + "\n".join(f"  {i}. {section}" for i, section in enumerate(SECTION_ORDER, 1))
    + "\n\nFollow this order in `final_text` when the recording covers these "
      "organs. Skip a line only if that organ was not examined -- do not pad "
      "or invent a finding to keep every line present."
)


def derive_from_reports(report_texts: list[str], min_agreement: float = 0.8) -> list[str]:
    """Recompute the boilerplate anchors from a set of reports.

    Not called by anything above -- `BOILERPLATE_ANCHORS` was produced by
    running this once and reading the result. Kept so the guide can be
    regenerated if the labelled dataset grows or changes: a line that stops
    appearing in most reports should stop being cited as boilerplate.
    """
    if not report_texts:
        return []
    lines_per_report = [
        # Splitting on "." also breaks abbreviations like "S.O.L." into
        # single letters; a minimum length keeps those out of the count.
        {line.strip() for line in re.split(r"[.\n]", text) if len(line.strip()) > 3}
        for text in report_texts
    ]
    from collections import Counter

    counts = Counter(line for lines in lines_per_report for line in lines)
    threshold = min_agreement * len(report_texts)
    return sorted((line for line, count in counts.items() if count >= threshold),
                 key=lambda line: -counts[line])
