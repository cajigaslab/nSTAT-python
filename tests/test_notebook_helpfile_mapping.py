from __future__ import annotations

import os
from pathlib import Path

import yaml


MANIFEST_PATH = Path("tools/notebooks/notebook_manifest.yml")
EXAMPLE_MAP_PATH = Path("parity/example_mapping.yaml")
NOTEBOOK_HELP_MAP_PATH = Path("parity/notebook_to_helpfile_map.yml")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_matlab_help_root() -> Path | None:
    env = os.environ.get("NSTAT_MATLAB_HELP_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.exists():
            return candidate
    default = Path("/tmp/upstream-nstat/helpfiles")
    if default.exists():
        return default
    return None


def test_notebook_helpfile_mapping_covers_manifest_topics() -> None:
    manifest = _load_yaml(MANIFEST_PATH)
    manifest_rows = manifest.get("notebooks", [])
    manifest_topics = {str(row["topic"]) for row in manifest_rows}
    manifest_by_topic = {str(row["topic"]): str(row["file"]) for row in manifest_rows}

    mapping = _load_yaml(NOTEBOOK_HELP_MAP_PATH)
    map_rows = mapping.get("mappings", [])
    map_topics = {str(row["topic"]) for row in map_rows}
    map_by_topic = {str(row["topic"]): row for row in map_rows}

    assert manifest_topics == map_topics
    for topic in sorted(manifest_topics):
        assert str(map_by_topic[topic]["notebook"]) == manifest_by_topic[topic]
        helpfile = str(map_by_topic[topic]["matlab_helpfile"])
        assert helpfile.endswith(".m")


def test_notebook_helpfile_mapping_aligns_with_example_mapping() -> None:
    example_map = _load_yaml(EXAMPLE_MAP_PATH)
    example_topics = {str(row["matlab_topic"]) for row in example_map.get("examples", [])}

    mapping = _load_yaml(NOTEBOOK_HELP_MAP_PATH)
    map_topics = {str(row["topic"]) for row in mapping.get("mappings", [])}
    assert map_topics == example_topics


def test_notebook_helpfiles_exist_when_matlab_checkout_is_available() -> None:
    matlab_help_root = _resolve_matlab_help_root()
    if matlab_help_root is None:
        return

    mapping = _load_yaml(NOTEBOOK_HELP_MAP_PATH)
    for row in mapping.get("mappings", []):
        helpfile = str(row["matlab_helpfile"])
        path = matlab_help_root / helpfile
        assert path.exists(), f"Missing MATLAB helpfile: {path}"

