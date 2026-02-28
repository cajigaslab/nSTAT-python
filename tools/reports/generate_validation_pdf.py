#!/usr/bin/env python3
"""Generate a visual PDF validation report for the standalone nSTAT-python toolbox."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nstat.analysis import Analysis  # noqa: E402
from nstat.cif import CIFModel  # noqa: E402
from nstat.decoding import DecodingAlgorithms  # noqa: E402
from nstat.history import HistoryBasis  # noqa: E402
from nstat.signal import Covariate  # noqa: E402
from nstat.spikes import SpikeTrain, SpikeTrainCollection  # noqa: E402
from nstat.trial import CovariateCollection, Trial, TrialConfig  # noqa: E402


@dataclass(slots=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    stdout_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to nSTAT-python repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "pdf",
        help="Directory for final PDF.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=REPO_ROOT / "tmp" / "pdfs" / "validation_report",
        help="Directory for intermediate plots.",
    )
    parser.add_argument(
        "--notebook-group",
        choices=["smoke", "full"],
        default="smoke",
        help="Notebook validation group to execute in report.",
    )
    parser.add_argument(
        "--skip-command-tests",
        action="store_true",
        help="Skip command-driven checks and only render visual example pages.",
    )
    return parser.parse_args()


def run_command(name: str, cmd: list[str], cwd: Path) -> CommandResult:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    raw_lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    filtered: list[str] = []
    skip_tokens = (
        "Debugger warning:",
        "PYDEVD_DISABLE_FILE_VALIDATION",
        "-Xfrozen_modules=off",
    )
    for line in raw_lines:
        stripped = line.strip()
        if any(token in stripped for token in skip_tokens):
            continue
        if stripped.startswith("0.00s -"):
            continue
        filtered.append(stripped)
    tail = filtered[-20:]
    return CommandResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        duration_s=elapsed,
        stdout_tail="\n".join(tail),
    )


def load_tolerances(repo_root: Path) -> dict[str, Any]:
    tol_path = repo_root / "tests" / "parity" / "tolerances.yml"
    return yaml.safe_load(tol_path.read_text(encoding="utf-8"))


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _make_nstatpaper_figure(tmp_dir: Path, tolerances: dict[str, Any]) -> tuple[Path, dict[str, float | bool]]:
    rng = np.random.default_rng(2026)

    time_grid = np.linspace(0.0, 20.0, 20001)
    dt = float(time_grid[1] - time_grid[0])
    stimulus = np.sin(2.0 * np.pi * 1.5 * time_grid) + 0.3 * np.cos(2.0 * np.pi * 0.5 * time_grid)
    X = stimulus[:, None]

    true_intercept = float(np.log(20.0))
    true_beta = np.array([0.55])
    true_model = CIFModel(coefficients=true_beta, intercept=true_intercept, link="poisson")
    true_rate = true_model.evaluate(X)
    spike_times = true_model.simulate_by_thinning(time_grid, X, rng=rng)

    cov = Covariate(time=time_grid, data=stimulus, name="stimulus", labels=["stim"])
    train = SpikeTrain(spike_times=spike_times, t_start=float(time_grid[0]), t_end=float(time_grid[-1]))
    trial = Trial(spikes=SpikeTrainCollection([train]), covariates=CovariateCollection([cov]))
    cfg = TrialConfig(covariate_labels=["stim"], sample_rate_hz=1.0 / dt, fit_type="poisson")
    fit = Analysis.fit_trial(trial, cfg)
    est_rate = fit.predict(X)

    coef_error = abs(float(fit.coefficients[0]) - float(true_beta[0]))
    rel_rate_err = float(np.mean(np.abs(est_rate - true_rate) / np.maximum(true_rate, 1e-9)))
    pass_coef = coef_error <= float(tolerances["poisson_glm"]["coefficient_abs_tol"])
    pass_rate = rel_rate_err <= float(tolerances["poisson_glm"]["rate_rel_tol"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].plot(time_grid, stimulus, color="black", linewidth=1.0)
    axes[0].set_title("nSTATPaperExamples: stimulus")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(time_grid, true_rate, label="true rate", linewidth=1.1)
    axes[1].plot(time_grid, est_rate, label="estimated rate", linewidth=1.1)
    axes[1].set_title("Poisson CIF fit")
    axes[1].set_ylabel("Hz")
    axes[1].legend(loc="upper right")

    raster_train = SpikeTrain(spike_times=spike_times, t_start=0.0, t_end=float(time_grid[-1]))
    centers, counts = raster_train.bin_counts(bin_size_s=0.05)
    axes[2].bar(centers, counts, width=0.045, color="tab:gray")
    axes[2].set_title("Binned spike counts")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("count/bin")

    out = tmp_dir / "nstatpaper_examples.png"
    _save_figure(out)
    metrics = {
        "coefficient_abs_error": coef_error,
        "rate_relative_error": rel_rate_err,
        "pass_coefficient_tolerance": bool(pass_coef),
        "pass_rate_tolerance": bool(pass_rate),
    }
    return out, metrics


def _make_ppsim_figure(tmp_dir: Path, tolerances: dict[str, Any]) -> tuple[Path, dict[str, float | bool]]:
    rng = np.random.default_rng(2026)
    time_grid = np.linspace(0.0, 8.0, 8001)
    dt = float(time_grid[1] - time_grid[0])
    stimulus = np.sin(2.0 * np.pi * 1.4 * time_grid) + 0.35 * np.cos(2.0 * np.pi * 0.4 * time_grid)
    X = stimulus[:, None]

    true_intercept = float(np.log(14.0))
    true_beta = np.array([0.50])
    model = CIFModel(coefficients=true_beta, intercept=true_intercept, link="poisson")
    true_rate = model.evaluate(X)

    n_trials = 18
    spike_trains: list[SpikeTrain] = []
    counts_all: list[np.ndarray] = []
    for _ in range(n_trials):
        spikes = model.simulate_by_thinning(time_grid, X, rng=rng)
        st = SpikeTrain(spike_times=spikes, t_start=float(time_grid[0]), t_end=float(time_grid[-1]))
        centers, counts = st.bin_counts(bin_size_s=dt)
        spike_trains.append(st)
        counts_all.append(counts)

    X_bins = np.interp(centers, time_grid, stimulus)[:, None]
    X_stack = np.tile(X_bins, (n_trials, 1))
    y_stack = np.concatenate(counts_all)
    fit = Analysis.fit_glm(X=X_stack, y=y_stack, fit_type="poisson", dt=dt)
    est_rate = fit.predict(X)

    coef_error = abs(float(fit.coefficients[0]) - float(true_beta[0]))
    rel_rate_err = float(np.mean(np.abs(est_rate - true_rate) / np.maximum(true_rate, 1e-9)))
    pass_coef = coef_error <= float(tolerances["poisson_glm"]["coefficient_abs_tol"])
    pass_rate = rel_rate_err <= float(tolerances["poisson_glm"]["rate_rel_tol"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].plot(time_grid, stimulus, color="black", linewidth=1.0)
    axes[0].set_title("PPSimExample: driving stimulus")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(time_grid, true_rate, label="true rate", linewidth=1.1)
    axes[1].plot(time_grid, est_rate, label="estimated rate", linewidth=1.1)
    axes[1].set_title("Estimated vs. true CIF")
    axes[1].set_ylabel("Hz")
    axes[1].legend(loc="upper right")

    for i, st in enumerate(spike_trains[:10]):
        axes[2].vlines(st.spike_times, i + 0.6, i + 1.4, color="black", linewidth=0.6)
    axes[2].set_title("Spike raster, first 10 trials")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("trial")

    out = tmp_dir / "ppsim_example.png"
    _save_figure(out)
    metrics = {
        "coefficient_abs_error": coef_error,
        "rate_relative_error": rel_rate_err,
        "total_spikes": float(sum(st.spike_times.size for st in spike_trains)),
        "pass_coefficient_tolerance": bool(pass_coef),
        "pass_rate_tolerance": bool(pass_rate),
    }
    return out, metrics


def _make_decoding_history_figure(tmp_dir: Path, tolerances: dict[str, Any]) -> tuple[Path, dict[str, float | bool]]:
    rng = np.random.default_rng(2026)
    n_units = 14
    n_states = 15
    n_time = 260
    dt = 0.02

    transition = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        for j, w in ((i - 1, 0.2), (i, 0.6), (i + 1, 0.2)):
            if 0 <= j < n_states:
                transition[i, j] += w
        transition[i, :] /= np.sum(transition[i, :])

    latent = np.zeros(n_time, dtype=int)
    latent[0] = n_states // 2
    for t in range(1, n_time):
        latent[t] = rng.choice(n_states, p=transition[latent[t - 1]])

    centers = np.linspace(0.0, n_states - 1, n_units)
    widths = np.full(n_units, 2.2)
    state_axis = np.arange(n_states)[None, :]
    tuning = 0.06 + 0.42 * np.exp(-0.5 * ((state_axis - centers[:, None]) / widths[:, None]) ** 2)

    history_weight = 0.55
    history_gain = np.ones(n_time, dtype=float)
    spike_counts = np.zeros((n_units, n_time), dtype=float)
    prev_global = 0.0
    for t in range(n_time):
        history_gain[t] = math.exp(history_weight * prev_global)
        lam = tuning[:, latent[t]] * history_gain[t]
        spike_counts[:, t] = rng.poisson(lam)
        prev_global = float(np.mean(spike_counts[:, t]))

    decoded_raw, _ = DecodingAlgorithms.decode_state_posterior(
        spike_counts=spike_counts,
        tuning_rates=tuning,
        transition=transition,
    )
    corrected = spike_counts / history_gain[None, :]
    decoded_hist, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=corrected,
        tuning_rates=tuning,
        transition=transition,
    )

    nrmse_raw = float(np.sqrt(np.mean((decoded_raw - latent) ** 2)) / (n_states - 1))
    nrmse_hist = float(np.sqrt(np.mean((decoded_hist - latent) ** 2)) / (n_states - 1))
    posterior_err = float(np.max(np.abs(np.sum(posterior, axis=0) - 1.0)))
    tol = float(tolerances["decoding"]["normalized_rmse_tol"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(latent, label="true state", linewidth=1.2)
    axes[0].plot(decoded_raw, label="decoded raw", linewidth=1.0)
    axes[0].plot(decoded_hist, label="decoded history-corrected", linewidth=1.0)
    axes[0].set_ylabel("state")
    axes[0].set_title("DecodingExampleWithHist: state decoding")
    axes[0].legend(loc="upper right")

    axes[1].plot(history_gain, color="tab:purple", linewidth=1.2)
    axes[1].set_title("Applied global history gain")
    axes[1].set_xlabel("time bin")
    axes[1].set_ylabel("gain")

    out = tmp_dir / "decoding_with_history.png"
    _save_figure(out)
    metrics = {
        "normalized_rmse_raw": nrmse_raw,
        "normalized_rmse_history_corrected": nrmse_hist,
        "posterior_normalization_error": posterior_err,
        "pass_history_improves_or_matches": bool(nrmse_hist <= nrmse_raw + 1e-8),
        "pass_rmse_tolerance": bool(nrmse_hist <= tol),
    }
    return out, metrics


def _make_placecell_figure(tmp_dir: Path, tolerances: dict[str, Any]) -> tuple[Path, dict[str, float | bool]]:
    rng = np.random.default_rng(2026)
    n_units = 30
    n_time = 320
    grid_side = 14
    grid = np.linspace(0.0, 1.0, grid_side)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    states_xy = np.column_stack([gx.ravel(), gy.ravel()])
    n_states = states_xy.shape[0]

    traj = np.zeros((n_time, 2), dtype=float)
    traj[0] = np.array([0.5, 0.5])
    velocity = np.zeros(2, dtype=float)
    for t in range(1, n_time):
        velocity = 0.85 * velocity + 0.12 * rng.normal(size=2)
        traj[t] = np.clip(traj[t - 1] + velocity, 0.0, 1.0)

    d2 = np.sum((states_xy[None, :, :] - traj[:, None, :]) ** 2, axis=2)
    true_state = np.argmin(d2, axis=1)

    centers = rng.uniform(0.0, 1.0, size=(n_units, 2))
    sigma = 0.18
    dist2 = np.sum((states_xy[None, :, :] - centers[:, None, :]) ** 2, axis=2)
    tuning = 0.03 + 0.80 * np.exp(-0.5 * dist2 / (sigma**2))

    spike_counts = np.zeros((n_units, n_time), dtype=float)
    for t in range(n_time):
        spike_counts[:, t] = rng.poisson(tuning[:, true_state[t]])

    decoded_wc = DecodingAlgorithms.decode_weighted_center(spike_counts, tuning)
    decoded_wc = np.clip(np.rint(decoded_wc), 0, n_states - 1).astype(int)

    log_tuning = np.log(np.clip(tuning, 1e-12, None))
    decoded_ml = np.zeros(n_time, dtype=int)
    for t in range(n_time):
        k = spike_counts[:, t][:, None]
        ll = np.sum(k * log_tuning - tuning, axis=0)
        decoded_ml[t] = int(np.argmax(ll))

    xy_true = states_xy[true_state]
    xy_wc = states_xy[decoded_wc]
    xy_ml = states_xy[decoded_ml]
    rmse_wc = float(np.sqrt(np.mean(np.sum((xy_wc - xy_true) ** 2, axis=1))))
    rmse_ml = float(np.sqrt(np.mean(np.sum((xy_ml - xy_true) ** 2, axis=1))))
    rmse_norm = rmse_ml / math.sqrt(2.0)
    tol = float(tolerances["place_decoding"]["normalized_rmse_tol"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].plot(xy_true[:, 0], xy_true[:, 1], label="true", linewidth=1.2)
    axes[0].plot(xy_ml[:, 0], xy_ml[:, 1], label="decoded ML", linewidth=1.0)
    axes[0].set_title("HippocampalPlaceCellExample: trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="upper right")

    im = axes[1].imshow(
        tuning[8].reshape(grid_side, grid_side),
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        cmap="jet",
        aspect="equal",
    )
    axes[1].set_title("Example place field")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    out = tmp_dir / "placecell_example.png"
    _save_figure(out)
    metrics = {
        "rmse_weighted_center": rmse_wc,
        "rmse_ml": rmse_ml,
        "normalized_rmse_ml": rmse_norm,
        "pass_ml_beats_or_matches_wc": bool(rmse_ml <= rmse_wc + 1e-8),
        "pass_rmse_tolerance": bool(rmse_norm <= tol),
    }
    return out, metrics


def _draw_wrapped_text(pdf: canvas.Canvas, x: float, y: float, text: str, width_chars: int = 95) -> float:
    lines: list[str] = []
    for block in text.splitlines():
        if len(block) <= width_chars:
            lines.append(block)
            continue
        while len(block) > width_chars:
            split = block.rfind(" ", 0, width_chars)
            if split <= 0:
                split = width_chars
            lines.append(block[:split])
            block = block[split:].lstrip()
        if block:
            lines.append(block)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 12
    return y


def _draw_image_page(
    pdf: canvas.Canvas,
    title: str,
    subtitle: str,
    image_path: Path,
    metrics: dict[str, float | bool],
) -> None:
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, 760, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, 744, subtitle)

    image_reader = ImageReader(str(image_path))
    iw, ih = image_reader.getSize()
    max_w = 520.0
    max_h = 430.0
    scale = min(max_w / iw, max_h / ih)
    w = iw * scale
    h = ih * scale
    pdf.drawImage(image_reader, 40, 280, width=w, height=h)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, 250, "Metrics")
    pdf.setFont("Helvetica", 10)
    y = 234
    for key, value in metrics.items():
        if isinstance(value, float):
            msg = f"- {key}: {value:.6f}"
        else:
            msg = f"- {key}: {value}"
        pdf.drawString(50, y, msg)
        y -= 14
    pdf.showPage()


def generate_pdf_report(
    repo_root: Path,
    output_pdf: Path,
    tmp_dir: Path,
    notebook_group: str,
    run_commands: bool,
) -> Path:
    tolerances = load_tolerances(repo_root)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    command_results: list[CommandResult] = []
    if run_commands:
        commands = [
            (
                "Unit tests",
                ["pytest", "-q", "tests/test_parity_numerics.py", "tests/test_behavior_contracts.py"],
            ),
            (
                f"Notebook execution ({notebook_group})",
                ["python", "tools/notebooks/run_notebooks.py", "--group", notebook_group, "--timeout", "900"],
            ),
            (
                "No MATLAB dependency gate",
                ["python", "tools/compliance/check_no_matlab_dependency.py"],
            ),
        ]
        for name, cmd in commands:
            command_results.append(run_command(name=name, cmd=cmd, cwd=repo_root))

    fig_builders = [
        ("nSTATPaperExamples", _make_nstatpaper_figure),
        ("PPSimExample", _make_ppsim_figure),
        ("DecodingExampleWithHist", _make_decoding_history_figure),
        ("HippocampalPlaceCellExample", _make_placecell_figure),
    ]
    pages: list[tuple[str, Path, dict[str, float | bool]]] = []
    for title, builder in fig_builders:
        image_path, metrics = builder(tmp_dir=tmp_dir, tolerances=tolerances)
        pages.append((title, image_path, metrics))

    commit = (
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        .stdout.strip()
        or "unknown"
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = canvas.Canvas(str(output_pdf), pagesize=letter)
    pdf.setTitle("nSTAT-python Standalone Validation Report")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, 760, "nSTAT-python Standalone Validation Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, 736, f"Generated: {now}")
    pdf.drawString(40, 720, f"Repository: {repo_root}")
    pdf.drawString(40, 704, f"Commit: {commit}")
    pdf.drawString(40, 688, "Objective: verify that the standalone Python toolbox generates expected examples and outputs.")

    y = 658
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, "Command-driven validation checks")
    y -= 18
    pdf.setFont("Helvetica", 10)
    if command_results:
        for result in command_results:
            status = "PASS" if result.passed else "FAIL"
            pdf.drawString(
                46,
                y,
                f"- {result.name}: {status} ({result.duration_s:.2f}s) | {' '.join(result.command)}",
            )
            y -= 14
            if result.stdout_tail:
                y = _draw_wrapped_text(pdf, 58, y, result.stdout_tail, width_chars=92)
                y -= 6
            if y < 80:
                pdf.showPage()
                y = 760
                pdf.setFont("Helvetica", 10)
    else:
        pdf.drawString(46, y, "- Command checks skipped by user option.")
        y -= 14

    if y < 120:
        pdf.showPage()
    else:
        pdf.showPage()

    for title, image_path, metrics in pages:
        _draw_image_page(
            pdf=pdf,
            title=title,
            subtitle="Visual output and deterministic metrics from standalone nSTAT-python code paths.",
            image_path=image_path,
            metrics=metrics,
        )

    pdf.save()
    return output_pdf


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = args.output_dir / f"nstat_python_validation_report_{stamp}.pdf"

    report_path = generate_pdf_report(
        repo_root=args.repo_root,
        output_pdf=output_pdf,
        tmp_dir=args.tmp_dir,
        notebook_group=args.notebook_group,
        run_commands=not args.skip_command_tests,
    )
    print(f"Generated PDF report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
