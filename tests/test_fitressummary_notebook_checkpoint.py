from __future__ import annotations

from pathlib import Path

import nbformat


def test_fitressummary_checkpoint_is_not_brittle_to_model_count() -> None:
    path = Path("notebooks") / "FitResSummaryExamples.ipynb"
    nb = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    # Regression guard: avoid fixed-size assumptions that can differ across
    # numerical environments and optimization backends.
    assert "assert diff_aic.size == diff_bic.size and diff_aic.size > 0" in code
    assert "assert diff_aic.size == 3 and diff_bic.size == 3" not in code

    # Regression guard: IC deltas can be positive or negative depending on
    # stochastic simulation and fit ordering; keep bounds symmetric.
    assert '"best_aic_diff": (-10.0, 10.0)' in code
    assert '"best_bic_diff": (-10.0, 10.0)' in code
