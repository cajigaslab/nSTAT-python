from __future__ import annotations

import json
from pathlib import Path


def test_generated_notebook_contains_figure_contract_cells(project_root: Path) -> None:
    nb_path = project_root / "notebooks" / "helpfiles" / "SignalObjExamples.ipynb"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in data.get("cells", []) if cell.get("cell_type") == "code")

    assert "render_figures=True" in code
    assert "figure_dir = repo_root / 'reports' / 'figures' / 'notebooks'" in code
    assert "figure_contract_expected" in code
    assert "figure_count" in code
