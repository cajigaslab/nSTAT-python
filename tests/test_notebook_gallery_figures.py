"""Guard: every published notebook-gallery figure must contain real data.

The galleries under ``docs/notebook_galleries/`` were once full of blank
placeholder figures (a title + a MATLAB call label, no plot) from
tracker-stub notebooks and over-declared ``expected_figures`` counts. After
porting every notebook to faithful executable figures, this test locks that
in: no committed gallery PNG may be a near-blank canvas.

The threshold mirrors ``tools/notebook_build/build_notebook_galleries.py``
``has_real_content`` (non-white pixel fraction > 0.008): empty placeholders
sit near 0.004, the sparsest real plots (event rasters, coarse step signal
reps) are comfortably above 0.01.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY_ROOT = REPO_ROOT / "docs" / "notebook_galleries"
MIN_NON_WHITE = 0.008


def _non_white_fraction(png_path: Path) -> float:
    arr = np.asarray(Image.open(png_path).convert("RGB"))
    return 1.0 - float(np.all(arr >= 245, axis=-1).mean())


@pytest.mark.parametrize(
    "png_path",
    sorted(GALLERY_ROOT.glob("*/fig_*.png")),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_gallery_figure_has_data(png_path: Path) -> None:
    frac = _non_white_fraction(png_path)
    assert frac > MIN_NON_WHITE, (
        f"{png_path.parent.name}/{png_path.name} looks blank "
        f"(non-white fraction {frac:.4f} <= {MIN_NON_WHITE}). Every gallery "
        "figure must plot real data — port the notebook cell or fix the "
        "notebook's expected_figures count."
    )


def test_gallery_is_non_empty() -> None:
    """Sanity: the gallery exists and has figures (so the parametrized guard
    above isn't vacuously passing on an empty glob)."""
    figs = list(GALLERY_ROOT.glob("*/fig_*.png"))
    assert len(figs) > 100, f"expected the full notebook gallery, found {len(figs)} figures"
