from __future__ import annotations

from pathlib import Path

import nbformat
import yaml


MANIFEST_PATH = Path("tools/notebooks/notebook_manifest.yml")
FORBIDDEN_PREFIXES = ("%", "!")


def _manifest_rows() -> list[dict[str, str]]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    return [dict(row) for row in payload.get("notebooks", [])]


def test_notebooks_are_code_only_and_have_no_magics() -> None:
    for row in _manifest_rows():
        nb_path = Path(str(row["file"]))
        nb = nbformat.read(nb_path, as_version=4)

        assert nb.cells, f"{row['topic']}: notebook has no cells"
        assert all(cell.cell_type == "code" for cell in nb.cells), f"{row['topic']}: notebook contains non-code cells"

        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        for line in code.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith(FORBIDDEN_PREFIXES), (
                f"{row['topic']}: forbidden notebook magic/shell line: {line!r}"
            )

        assert 'matplotlib.use("Agg")' in code, f"{row['topic']}: missing Agg backend configuration"
        assert "np.random.seed(" in code, f"{row['topic']}: missing deterministic NumPy seed"

