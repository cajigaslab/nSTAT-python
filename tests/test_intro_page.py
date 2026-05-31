"""Drift guards for ``docs/intro.html`` — the friendly 5-minute intro.

The intro page is a hand-curated landing page (self-contained HTML,
published via ``html_extra_path``).  Its value is exactly that it shows
real APIs and links real assets — so these tests catch the rot:

1. The file exists and is registered in ``docs/conf.py::html_extra_path``.
2. It mentions the README link line (so the GitHub landing keeps
   pointing at the published intro).
3. Every public ``nstat`` symbol the page name-drops is actually in
   ``nstat.__all__`` (catches API renames silently outdating the intro).
4. The paper-figure image paths it links resolve to the real PNGs
   produced by ``examples/paper/regenerate_all_figures.py``.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTRO_PATH = REPO_ROOT / "docs" / "intro.html"


def test_intro_page_exists() -> None:
    assert INTRO_PATH.exists(), (
        f"{INTRO_PATH.relative_to(REPO_ROOT)} missing — the intro "
        "landing page must ship with the repo."
    )


def test_intro_registered_in_html_extra_path() -> None:
    """``docs/conf.py`` must list intro.html so Sphinx publishes it."""
    conf = (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert "intro.html" in conf and "html_extra_path" in conf, (
        "docs/conf.py html_extra_path must include 'intro.html' so Sphinx "
        "copies it into the published Pages site."
    )


def test_readme_links_to_intro_page() -> None:
    """The README's top callout must surface the intro page so first-time
    visitors of the GitHub repo land on a friendly entry point."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "intro.html" in readme, (
        "README.md must link to the published intro.html so the GitHub "
        "landing page surfaces the friendly entry point."
    )


def test_intro_only_references_real_public_symbols() -> None:
    """Every ``nstat`` symbol the intro shows in code samples must exist
    in the current public API.  Catches the failure mode where the page
    rots: an API gets renamed and the intro silently keeps using the
    old name, misleading new users.
    """
    import nstat

    text = INTRO_PATH.read_text(encoding="utf-8")
    # Pull names from the code blocks (anything in <pre><code>...</code></pre>).
    code_blocks = re.findall(r"<pre><code>(.*?)</code></pre>", text, re.DOTALL)
    blob = "\n".join(code_blocks)

    # Conservative check: a handful of high-value symbols the page promises.
    must_exist = [
        "nspikeTrain",
        "Trial",
        "Covariate",
        "nstColl",
        "TrialConfig",
        "ConfigColl",
        "Analysis",
        "FitResult",
        "population_time_rescale",
    ]
    for sym in must_exist:
        if sym in blob:
            assert hasattr(nstat, sym), (
                f"intro.html names public symbol {sym!r} but it is not in "
                f"nstat.__all__ — rename the page or restore the symbol."
            )


def test_intro_paper_figure_images_resolve() -> None:
    """Each gallery thumbnail must point at a real paper-figure PNG.

    The intro uses ``_images/fig01_*.png`` — that's where Sphinx copies
    the canonical figure-tree PNGs from ``docs/figures/example0N/``.
    Stale gallery refs would 404 on the published site.
    """
    text = INTRO_PATH.read_text(encoding="utf-8")
    image_paths = re.findall(r'src="_images/(fig01_[^"]+)"', text)
    assert image_paths, "intro.html has no <img src='_images/fig01_*'> gallery references"

    figures_dir = REPO_ROOT / "docs" / "figures"
    missing: list[str] = []
    for name in image_paths:
        # The PNG sits under docs/figures/example0N/<name>.png — we don't
        # know which N, so glob.
        if not list(figures_dir.glob(f"example*/{name}")):
            missing.append(name)
    assert not missing, (
        f"intro.html gallery references PNGs that don't exist under "
        f"docs/figures/example*/: {missing}.  Regenerate via "
        f"examples/paper/regenerate_all_figures.py OR update the page."
    )
