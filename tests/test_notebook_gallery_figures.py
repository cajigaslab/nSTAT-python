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


def _embedded_image_count(nb) -> int:
    return sum(
        1
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if any(key.startswith("image/") for key in out.get("data", {}))
    )


def test_notebooks_embed_their_figures() -> None:
    """Every notebook that produces gallery figures must also EMBED those
    figures as cell outputs, so they render in the committed ``.ipynb`` (and on
    GitHub), not only in the gallery PNGs.

    ``FigureTracker`` displays each figure inline as it is saved; this guard
    fails if a notebook is committed output-stripped, or with fewer embedded
    figures than it produces.
    """
    import nbformat

    nb_dir = REPO_ROOT / "notebooks"
    failures = []
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        gallery = GALLERY_ROOT / nb_path.stem
        n_gallery = len(list(gallery.glob("fig_*.png"))) if gallery.exists() else 0
        if n_gallery == 0:
            continue  # notebooks with no figures (e.g. API-reference stubs)
        nb = nbformat.read(nb_path, as_version=4)
        n_embedded = _embedded_image_count(nb)
        if n_embedded < n_gallery:
            failures.append(
                f"{nb_path.name}: {n_embedded} embedded figure(s) < {n_gallery} "
                "gallery figure(s)"
            )
    assert not failures, (
        "Notebooks must embed every figure they produce (run "
        "`python tools/notebook_build/embed_figures.py --all`):\n  "
        + "\n  ".join(failures)
    )
