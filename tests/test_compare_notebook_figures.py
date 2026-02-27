from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.compare_notebook_figures_to_matlab import _score_pair


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def test_score_pair_passes_for_identical_images(tmp_path: Path) -> None:
    arr = np.zeros((32, 32, 3), dtype=np.float32)
    arr[:, :, 1] = np.linspace(0.0, 1.0, 32)[None, :]

    py_path = tmp_path / "py.png"
    ref_path = tmp_path / "ref.png"
    _save_rgb(py_path, arr)
    _save_rgb(ref_path, arr)

    thresholds = {"ssim_min": 0.45, "psnr_min": 12.0, "hist_intersection_min": 0.70}
    out = _score_pair(py_path, ref_path, thresholds)
    assert out["pass"] is True
    assert out["votes"] == 3


def test_score_pair_fails_for_very_different_images(tmp_path: Path) -> None:
    a = np.zeros((40, 40, 3), dtype=np.float32)
    b = np.ones((40, 40, 3), dtype=np.float32)

    py_path = tmp_path / "py.png"
    ref_path = tmp_path / "ref.png"
    _save_rgb(py_path, a)
    _save_rgb(ref_path, b)

    thresholds = {"ssim_min": 0.45, "psnr_min": 12.0, "hist_intersection_min": 0.70}
    out = _score_pair(py_path, ref_path, thresholds)
    assert out["pass"] is False
    assert out["votes"] <= 1
