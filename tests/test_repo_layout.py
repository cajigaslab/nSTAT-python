from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_uses_single_canonical_package_layout() -> None:
    assert (REPO_ROOT / "nstat").is_dir()
    assert not (REPO_ROOT / "src" / "nstat").exists()
    assert not (REPO_ROOT / "__init__.py").exists()
