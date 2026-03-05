from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import nbformat

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "reports" / "generate_validation_pdf.py"
_SPEC = importlib.util.spec_from_file_location("generate_validation_pdf", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gvp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gvp
_SPEC.loader.exec_module(gvp)


def test_execute_notebook_capture_discovers_saved_tracker_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    nb_dir = repo_root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    topic = "DiscoverySmoke"
    nb_path = nb_dir / f"{topic}.ipynb"

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "import matplotlib",
                    "matplotlib.use('Agg')",
                    "import matplotlib.pyplot as plt",
                    "from pathlib import Path",
                    f"topic = {topic!r}",
                    "out = Path.cwd().resolve().parent / 'output' / 'notebook_images' / topic",
                    "out.mkdir(parents=True, exist_ok=True)",
                    "fig = plt.figure(figsize=(4, 2))",
                    "ax = fig.add_subplot(1,1,1)",
                    "ax.plot([0, 1], [0, 1], color='k')",
                    "fig.tight_layout()",
                    "fig.savefig(out / 'fig_001.png', dpi=120)",
                    "plt.close(fig)",
                ]
            )
        )
    ]
    nbformat.write(nb, nb_path)

    monkeypatch.setattr(gvp, "REPO_ROOT", repo_root)

    report = gvp.execute_notebook_capture(
        target=gvp.NotebookTarget(topic=topic, file=nb_path, run_group="smoke"),
        tmp_dir=repo_root / "tmp" / "pdfs" / "validation_report",
        timeout=120,
        matlab_help_root=None,
        parity_threshold=0.8,
        skip_parity_check=True,
        parity_mode="gate",
        gate_status=("verified", True),
        parity_metrics=None,
    )

    assert report.executed
    assert report.unique_image_count >= 1
    assert report.image_count >= report.unique_image_count
    assert all(path.exists() for path in report.image_paths)

    discovered_dir = repo_root / "tmp" / "pdfs" / "validation_report" / "notebook_images" / topic
    discovered = sorted(discovered_dir.glob("*.png"))
    assert discovered
