"""Image-pair content-audit heuristic for visual-parity triage.

Replaces the historical "is `non_white_pct < 3%`?" filter that produced
recurring false-positive "degenerate row" tickets on sparse-but-real plots
(trajectory lines, dot scatters, axis-off schematics with text + arrows).

The fundamental observation: a *truly* degenerate render — a blank white
panel, a failed savefig, an axis-only frame with no data — has BOTH

  - a tiny non-white pixel fraction AND
  - a tiny distinct-intensity count AND
  - very few dark pixels (no text, no axes labels, no plotted glyphs).

A sparse real plot fails at least one of those three tests: schematics
have hundreds of distinct intensities from font anti-aliasing, trajectory
plots have anti-aliased line edges, and dot scatters have visible markers.

Public surface
--------------

:func:`content_score(image_path)` returns a :class:`ContentScore`
namedtuple with the four diagnostic signals and a single
:attr:`ContentScore.is_degenerate` boolean.

:func:`audit_pair(matlab_png, python_png)` returns per-side scores and a
verdict string ('match', 'matlab-only-degenerate', etc.) suitable for the
upstream-watch reconciliation flow.

CLI
---

    python tools/parity/image_content_audit.py PATH [PATH ...]
        Print the per-image content score and degenerate flag.

    python tools/parity/image_content_audit.py --pair MATLAB.png PYTHON.png
        Print the per-side scores and the pair verdict.

    python tools/parity/image_content_audit.py --self-test
        Scan a known-schematic and a synthetic blank PNG; assert that the
        schematic registers as non-degenerate and the blank registers as
        degenerate. Exit non-zero on regression.

Documented in CLAUDE.md under "MATLAB capture-script conventions ->
image-pair validation heuristic".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Thresholds — empirically tuned on the iter 69/70 false-positive sample.
# A blank/degenerate panel must clear ALL three lower bounds.  Real plots
# (even sparse ones) clear at least one of them.
# --------------------------------------------------------------------------- #
NON_WHITE_PCT_FLOOR = 1.5    # fraction percent of non-near-white pixels
DISTINCT_INTENSITY_FLOOR = 20  # distinct grayscale levels (text/aa raises this)
DARK_PIXEL_FLOOR = 500       # count of intensity<100 pixels (text+lines)

NEAR_WHITE_THRESHOLD = 245   # grayscale >= this counts as "white"
DARK_THRESHOLD = 100         # grayscale < this counts as "dark"


@dataclass(frozen=True)
class ContentScore:
    """Per-image content diagnostics.

    Attributes
    ----------
    path:
        The source PNG path (for reporting).
    non_white_pct:
        Percent of pixels with grayscale value below
        ``NEAR_WHITE_THRESHOLD``.  Was the historical sole filter.
    distinct_intensity_count:
        Number of distinct grayscale levels (0-255) present in the image.
        Text and anti-aliased line edges push this above 50; a flat
        white panel registers 1-3.
    dark_pixel_count:
        Number of pixels with grayscale value below ``DARK_THRESHOLD``.
        A panel with axis spines + labels typically has > 1000 dark
        pixels even when the data is sparse.
    is_degenerate:
        ``True`` iff ALL three signals fall below their respective
        floors — i.e. there is no meaningful content (no plotted glyphs,
        no text, no axis frame).
    """

    path: Path
    non_white_pct: float
    distinct_intensity_count: int
    dark_pixel_count: int
    is_degenerate: bool

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "non_white_pct": round(self.non_white_pct, 3),
            "distinct_intensity_count": self.distinct_intensity_count,
            "dark_pixel_count": self.dark_pixel_count,
            "is_degenerate": self.is_degenerate,
        }


def _load_grayscale(image_path: Path) -> np.ndarray:
    """Return the image as a uint8 grayscale ``np.ndarray``.

    Uses PIL — already a transitive dependency via matplotlib.  We
    flatten alpha by compositing against white so transparent PNG
    backgrounds don't show as "dark" (alpha=0 pixels are visually white
    on docs sites).
    """
    from PIL import Image

    with Image.open(image_path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        gray = im.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def content_score(image_path: Path | str) -> ContentScore:
    """Compute the content-score for ``image_path``.

    Loads the PNG, flattens to grayscale (compositing alpha against
    white), and returns a :class:`ContentScore` with the three
    diagnostic signals plus the degenerate verdict.

    Raises
    ------
    FileNotFoundError
        If ``image_path`` does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    gray = _load_grayscale(path)
    total = gray.size
    non_white_pct = float((gray < NEAR_WHITE_THRESHOLD).sum()) / total * 100.0
    distinct = int(np.unique(gray).size)
    dark_count = int((gray < DARK_THRESHOLD).sum())
    is_degenerate = (
        non_white_pct < NON_WHITE_PCT_FLOOR
        and distinct < DISTINCT_INTENSITY_FLOOR
        and dark_count < DARK_PIXEL_FLOOR
    )
    return ContentScore(
        path=path,
        non_white_pct=non_white_pct,
        distinct_intensity_count=distinct,
        dark_pixel_count=dark_count,
        is_degenerate=is_degenerate,
    )


# --------------------------------------------------------------------------- #
# Pair verdict — used by reconciliation flows that compare a MATLAB PNG
# against the corresponding Python gallery PNG.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairVerdict:
    matlab: ContentScore
    python: ContentScore
    verdict: str  # 'match', 'both-degenerate', 'matlab-only-degenerate',
    #              'python-only-degenerate'


def audit_pair(matlab_png: Path | str, python_png: Path | str) -> PairVerdict:
    """Run the content-score on both sides and label the pair.

    Verdicts:

    - ``'match'`` — both sides are non-degenerate (typical case).
    - ``'both-degenerate'`` — both sides are blank/empty; the figure
      slot is probably unused on both sides, no reconciliation needed.
    - ``'matlab-only-degenerate'`` — MATLAB side is blank, Python side
      has content; likely a Python-only figure that has no MATLAB
      counterpart.
    - ``'python-only-degenerate'`` — Python side is blank; likely a
      real parity gap or a savefig regression on the Python side.
    """
    m = content_score(matlab_png)
    p = content_score(python_png)
    if m.is_degenerate and p.is_degenerate:
        verdict = "both-degenerate"
    elif m.is_degenerate:
        verdict = "matlab-only-degenerate"
    elif p.is_degenerate:
        verdict = "python-only-degenerate"
    else:
        verdict = "match"
    return PairVerdict(matlab=m, python=p, verdict=verdict)


# --------------------------------------------------------------------------- #
# Self-test — exercised by the CLI ``--self-test`` flag and the smoke test.
# Generates a synthetic blank PNG via PIL so it does not need a fixture
# checked into git.
# --------------------------------------------------------------------------- #
def _make_blank_png(tmpdir: Path) -> Path:
    """Write a 600x400 all-white PNG and return its path."""
    from PIL import Image

    blank = Image.new("RGB", (600, 400), (255, 255, 255))
    path = tmpdir / "synthetic_blank.png"
    blank.save(path)
    return path


def run_self_test() -> int:
    """Assert that a schematic registers as non-degenerate and a blank
    PNG registers as degenerate.

    The schematic case uses the upstream MATLAB connectivity diagram
    (NetworkTutorial_06.png) if a local MATLAB checkout is available;
    otherwise it falls back to the Python-side regenerated equivalent
    so the test runs without the MATLAB repo.
    """
    import os
    import tempfile

    matlab_root = Path(os.environ.get("NSTAT_MATLAB_PATH", "/Users/iahncajigas/projects/nstat"))
    schematic_candidates = [
        matlab_root / "helpfiles" / "NetworkTutorial_06.png",
        Path("docs/notebook_galleries/NetworkTutorial/fig_006.png"),
    ]
    schematic = next((p for p in schematic_candidates if p.exists()), None)
    if schematic is None:
        print("self-test SKIP: no NetworkTutorial_06 schematic available", file=sys.stderr)
        return 0

    sch_score = content_score(schematic)
    if sch_score.is_degenerate:
        print(
            "self-test FAIL: schematic flagged as degenerate:\n  "
            f"{sch_score.as_dict()}",
            file=sys.stderr,
        )
        return 1
    print(f"self-test ok: schematic non-degenerate -- {sch_score.as_dict()}")

    with tempfile.TemporaryDirectory() as td:
        blank_path = _make_blank_png(Path(td))
        blank_score = content_score(blank_path)
    if not blank_score.is_degenerate:
        print(
            "self-test FAIL: synthetic blank not flagged degenerate:\n  "
            f"{blank_score.as_dict()}",
            file=sys.stderr,
        )
        return 1
    print(f"self-test ok: synthetic blank flagged degenerate -- {blank_score.as_dict()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="PNG path(s) to score individually.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("MATLAB_PNG", "PYTHON_PNG"),
        type=Path,
        help="Compare a MATLAB/Python PNG pair and print the verdict.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the schematic + synthetic-blank smoke check.",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.pair:
        verdict = audit_pair(*args.pair)
        print(f"MATLAB: {verdict.matlab.as_dict()}")
        print(f"PYTHON: {verdict.python.as_dict()}")
        print(f"VERDICT: {verdict.verdict}")
        return 0

    if not args.paths:
        parser.error("provide PATH(s), --pair, or --self-test")
    for p in args.paths:
        score = content_score(p)
        flag = "DEGENERATE" if score.is_degenerate else "ok"
        print(
            f"[{flag}] {p}\n"
            f"    non_white_pct={score.non_white_pct:.2f}  "
            f"distinct={score.distinct_intensity_count}  "
            f"dark={score.dark_pixel_count}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
