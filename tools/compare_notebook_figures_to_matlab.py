from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reference" / "matlab_helpfigures" / "manifest.json"
THRESHOLDS_PATH = PROJECT_ROOT / "reference" / "matlab_helpfigures" / "thresholds.json"
DEFAULT_PY_FIG_ROOT = PROJECT_ROOT / "reports" / "figures" / "notebooks"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "notebook_figure_parity.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _resize_like(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    resized = pil.resize(size=size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _hist_intersection(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    score = 0.0
    for channel in range(3):
        ha, _ = np.histogram(a[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        hb, _ = np.histogram(b[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        ha = ha / max(np.sum(ha), 1e-12)
        hb = hb / max(np.sum(hb), 1e-12)
        score += float(np.minimum(ha, hb).sum())
    return score / 3.0


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _score_pair(py_path: Path, matlab_path: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    py_img = _load_image_rgb(py_path)
    ref_img = _load_image_rgb(matlab_path)

    if py_img.shape != ref_img.shape:
        py_img = _resize_like(py_img, size=(ref_img.shape[1], ref_img.shape[0]))

    ssim = float(structural_similarity(py_img, ref_img, channel_axis=2, data_range=1.0))
    psnr = _psnr(ref_img, py_img)
    hist = float(_hist_intersection(py_img, ref_img))

    checks = {
        "ssim": ssim >= float(thresholds["ssim_min"]),
        "psnr": psnr >= float(thresholds["psnr_min"]),
        "hist_intersection": hist >= float(thresholds["hist_intersection_min"]),
    }
    votes = sum(1 for ok in checks.values() if ok)
    rule = str(thresholds.get("pair_pass_rule", "2_of_3")).strip().lower()
    required_votes = 2 if rule == "2_of_3" else 3
    pair_pass = votes >= required_votes
    return {
        "py": str(py_path),
        "matlab": str(matlab_path),
        "metrics": {
            "ssim": ssim,
            "psnr": psnr,
            "hist_intersection": hist,
        },
        "threshold_checks": checks,
        "votes": votes,
        "required_votes": required_votes,
        "pass": pair_pass,
    }


def _write_diff(py_path: Path, matlab_path: Path, out_path: Path) -> None:
    py_img = _load_image_rgb(py_path)
    ref_img = _load_image_rgb(matlab_path)
    if py_img.shape != ref_img.shape:
        py_img = _resize_like(py_img, size=(ref_img.shape[1], ref_img.shape[0]))
    diff = np.abs(py_img - ref_img)
    heat = np.clip(diff.mean(axis=2) * 3.0, 0.0, 1.0)
    rgb = np.zeros((*heat.shape, 3), dtype=np.float32)
    rgb[:, :, 0] = heat
    rgb[:, :, 1] = 1.0 - heat
    rgb[:, :, 2] = 1.0 - heat
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare generated notebook figures to vendored MATLAB baselines")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--thresholds", default=str(THRESHOLDS_PATH))
    parser.add_argument("--python-figures", default=str(DEFAULT_PY_FIG_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    thresholds_path = Path(args.thresholds).resolve()
    py_root = Path(args.python_figures).resolve()
    report_path = Path(args.report).resolve()

    manifest = _load_json(manifest_path)
    thresholds = _load_json(thresholds_path)
    ref_root = manifest_path.parent

    rows: list[dict[str, Any]] = []
    topic_passes = 0
    pair_total = 0
    pair_passes = 0

    for topic, info in sorted(manifest.get("topics", {}).items()):
        expected = int(info.get("expected_figures", 0))
        baseline_files = [ref_root / rel for rel in info.get("baseline_files", [])]

        py_topic_dir = py_root / topic
        py_files = sorted(py_topic_dir.glob("fig_*.png"))

        topic_row: dict[str, Any] = {
            "topic": topic,
            "expected_figures": expected,
            "python_figures_found": len(py_files),
            "baseline_figures_found": len(baseline_files),
            "pairs": [],
            "pass": False,
            "reason": "",
        }

        if expected == 0:
            topic_row["pass"] = len(py_files) == 0
            topic_row["reason"] = "zero_figure_contract"
            if topic_row["pass"]:
                topic_passes += 1
            rows.append(topic_row)
            continue

        if len(py_files) != expected:
            topic_row["reason"] = "python_figure_count_mismatch"
            rows.append(topic_row)
            continue
        if len(baseline_files) != expected:
            topic_row["reason"] = "baseline_figure_count_mismatch"
            rows.append(topic_row)
            continue

        failed = False
        for idx, (py_path, ref_path) in enumerate(zip(py_files, baseline_files), start=1):
            pair_total += 1
            pair = _score_pair(py_path, ref_path, thresholds)
            pair["index"] = idx
            topic_row["pairs"].append(pair)
            if pair["pass"]:
                pair_passes += 1
            else:
                failed = True
                diff_path = PROJECT_ROOT / "reports" / "figures" / "diffs" / topic / f"fig_{idx:03d}_diff.png"
                _write_diff(py_path, ref_path, diff_path)
                pair["diff"] = str(diff_path)

        topic_row["pass"] = not failed
        topic_row["reason"] = "all_pairs_pass" if not failed else "pair_metric_failures"
        if topic_row["pass"]:
            topic_passes += 1
        rows.append(topic_row)

    summary = {
        "topics": len(rows),
        "topics_pass": topic_passes,
        "pairs": pair_total,
        "pairs_pass": pair_passes,
        "pass": topic_passes == len(rows),
        "thresholds": thresholds,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report_path.relative_to(PROJECT_ROOT))}, indent=2))

    if args.enforce_gate and not summary["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
