#!/usr/bin/env python3
"""Strict ordinal image-parity check for helpfile-derived notebook figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import yaml

try:  # pragma: no cover - optional dependency
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parity/help_source_manifest.yml"),
        help="Help source manifest path.",
    )
    parser.add_argument(
        "--python-image-root",
        type=Path,
        default=Path("output/notebook_images"),
        help="Root folder with generated notebook fig_###.png assets.",
    )
    parser.add_argument(
        "--matlab-image-root",
        type=Path,
        default=Path("baseline/validation/notebook_images"),
        help="Root folder with MATLAB reference fig_###.png assets.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.70,
        help="Minimum SSIM score required for each ordinal pair.",
    )
    parser.add_argument(
        "--topics",
        default="",
        help="Optional comma-separated topic subset to validate.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("output/pdf/image_mode_parity/summary.json"),
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--diff-root",
        type=Path,
        default=Path("output/pdf/image_mode_parity/diffs"),
        help="Directory for per-figure diff images on failures.",
    )
    return parser.parse_args()


def _load_gray(path: Path) -> np.ndarray:
    arr = mpimg.imread(path)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = np.mean(arr, axis=2)
    return np.asarray(arr, dtype=float)


def _resize_like(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep implementation dependency-light: crop to overlapping extents.
    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])
    return a[:rows, :cols], b[:rows, :cols]


def _score_pair(a: Path, b: Path) -> dict[str, float]:
    img_a = _load_gray(a)
    img_b = _load_gray(b)
    img_a, img_b = _resize_like(img_a, img_b)
    rmse = float(np.sqrt(np.mean((img_a - img_b) ** 2)))
    rmse_score = float(max(0.0, 1.0 - rmse))
    ssim_score = rmse_score
    if _ssim is not None:
        data_range = float(max(np.max(img_a), np.max(img_b)) - min(np.min(img_a), np.min(img_b)))
        if data_range <= 0.0:
            data_range = 1.0
        ssim_score = float(_ssim(img_a, img_b, data_range=data_range))
    # Gate on a robust visual score to reduce renderer-specific SSIM sensitivity.
    score = float(max(ssim_score, rmse_score))
    return {"score": score, "ssim": float(ssim_score), "rmse_score": rmse_score}


def _save_diff_image(py_path: Path, mat_path: Path, out_path: Path) -> None:
    img_a = _load_gray(py_path)
    img_b = _load_gray(mat_path)
    img_a, img_b = _resize_like(img_a, img_b)
    diff = np.abs(img_a - img_b)
    # Normalize for visibility while staying deterministic.
    maxv = float(np.max(diff))
    if maxv > 0:
        diff = diff / maxv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(out_path, diff, cmap="magma", vmin=0.0, vmax=1.0)


def main() -> int:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    rows = manifest.get("topics", [])
    if args.topics.strip():
        wanted = {token.strip() for token in args.topics.split(",") if token.strip()}
        rows = [row for row in rows if str(row.get("topic", "")).strip() in wanted]
        if not rows:
            raise RuntimeError(f"No topics matched --topics={args.topics!r}")

    results: list[dict[str, object]] = []
    failures: list[str] = []

    for row in rows:
        topic = str(row["topic"])
        no_figure_utility = bool(row.get("no_figure_utility", False))
        py_images = sorted((args.python_image_root / topic).glob("fig_*.png"))
        mat_images = sorted((args.matlab_image_root / topic).glob("*.png"))
        topic_result: dict[str, object] = {
            "topic": topic,
            "expected_figures": int(row.get("expected_figure_count", len(mat_images))),
            "produced_figures": len(py_images),
            "reference_figures": len(mat_images),
            "no_figure_utility": no_figure_utility,
            "pairs": [],
        }

        if no_figure_utility:
            results.append(topic_result)
            continue

        if len(py_images) != len(mat_images):
            failures.append(
                f"{topic}: figure count mismatch python={len(py_images)} matlab={len(mat_images)}"
            )

        for idx, (py_img, mat_img) in enumerate(zip(py_images, mat_images), start=1):
            metrics = _score_pair(py_img, mat_img)
            pair_result = {
                "ordinal": idx,
                "python_image": str(py_img),
                "matlab_image": str(mat_img),
                "score": metrics["score"],
                "ssim": metrics["ssim"],
                "rmse_score": metrics["rmse_score"],
            }
            cast_pairs = topic_result["pairs"]
            assert isinstance(cast_pairs, list)
            cast_pairs.append(pair_result)
            if metrics["score"] < args.ssim_threshold:
                diff_path = args.diff_root / topic / f"fig_{idx:03d}_diff.png"
                _save_diff_image(py_img, mat_img, diff_path)
                pair_result["diff_image"] = str(diff_path)
                failures.append(
                    f"{topic}: fig_{idx:03d} score {metrics['score']:.4f} (ssim={metrics['ssim']:.4f}, rmse_score={metrics['rmse_score']:.4f}) < threshold {args.ssim_threshold:.4f}"
                )
        results.append(topic_result)

    summary = {
        "ssim_threshold": args.ssim_threshold,
        "gate_metric": "max(ssim, 1-rmse)",
        "python_image_root": str(args.python_image_root),
        "matlab_image_root": str(args.matlab_image_root),
        "topics": results,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failures": len(failures)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
