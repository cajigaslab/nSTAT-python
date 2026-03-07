from __future__ import annotations

from tools.notebooks.changed_topics import infer_topics_from_paths, load_group, load_manifest


def test_changed_notebook_paths_map_to_manifest_topics() -> None:
    manifest = load_manifest()
    parity_core = load_group("parity_core")

    topics = infer_topics_from_paths(
        ["notebooks/TrialExamples.ipynb", "notebooks/AnalysisExamples.ipynb"],
        manifest,
        parity_core,
    )

    assert topics == ["AnalysisExamples", "TrialExamples"]


def test_notebook_infrastructure_changes_fall_back_to_parity_core_group() -> None:
    manifest = load_manifest()
    parity_core = load_group("parity_core")

    topics = infer_topics_from_paths(
        ["tools/notebooks/run_notebooks.py", "parity/notebook_fidelity.yml"],
        manifest,
        parity_core,
    )

    assert topics == sorted(set(parity_core))


def test_non_notebook_changes_do_not_trigger_notebook_execution() -> None:
    manifest = load_manifest()
    parity_core = load_group("parity_core")

    topics = infer_topics_from_paths(
        ["README.md", "nstat/cif.py"],
        manifest,
        parity_core,
    )

    assert topics == []
