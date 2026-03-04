from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat
import yaml


MANIFEST = Path("tools/notebooks/notebook_manifest.yml")
IMAGE_ROOT = Path("baseline/validation/notebook_images")
WEAK_TOPICS = [
    "DecodingExampleWithHist",
    "PSTHEstimation",
    "nstCollExamples",
    "TrialExamples",
    "CovCollExamples",
    "EventsExamples",
]


def _topic_code(topic: str) -> str:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    rows = payload.get("notebooks", [])
    by_topic = {str(row["topic"]): Path(str(row["file"])) for row in rows}
    nb_path = by_topic[topic]
    nb = nbformat.read(nb_path, as_version=4)
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_all_manifest_topics_have_validation_image_fixtures() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for row in payload["notebooks"]:
        topic = str(row["topic"])
        images = sorted((IMAGE_ROOT / topic).glob("*.png"))
        assert images, f"missing validation image fixtures for topic {topic}"


def test_weak_topics_have_topic_specific_assertions() -> None:
    expected_snippets = {
        "DecodingExampleWithHist": [
            "np.max(np.abs(np.sum(posterior, axis=0) - 1.0)) < 1e-6",
            "rmse_dec <= rmse_raw + 0.03",
        ],
        "PSTHEstimation": [
            "np.allclose(prob_mat, prob_mat.T",
            "np.all(np.diag(prob_mat) == 1.0)",
        ],
        "nstCollExamples": [
            "H.ndim == 2 and H.shape[1] == history.n_bins",
            "spikes.spike_times.size > 5",
        ],
        "TrialExamples": [
            "H.ndim == 2 and H.shape[1] == history.n_bins",
            "spikes.spike_times.size > 5",
        ],
        "CovCollExamples": [
            "H.ndim == 2 and H.shape[1] == history.n_bins",
            "spikes.spike_times.size > 5",
        ],
        "EventsExamples": [
            "events.times.size == 3",
            "np.all(np.diff(events.times) > 0.0)",
        ],
    }

    for topic in WEAK_TOPICS:
        code = _topic_code(topic)
        assert "Topic-specific checkpoint" in code
        assert "Notebook checkpoints passed" in code
        for snippet in expected_snippets[topic]:
            assert snippet in code, f"missing expected assertion snippet for {topic}: {snippet}"


def test_weak_topic_validation_images_are_not_all_identical() -> None:
    hashes: list[str] = []
    for topic in WEAK_TOPICS:
        images = sorted((IMAGE_ROOT / topic).glob("*.png"))
        assert images, f"missing validation image fixture for topic {topic}"
        first = images[0]
        assert first.stat().st_size > 0, f"empty validation image for topic {topic}"
        hashes.append(_sha256(first))

    # Guardrail against a broken renderer reusing one image for all topics.
    assert len(set(hashes)) >= 4
