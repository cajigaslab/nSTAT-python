#!/usr/bin/env python3
"""Generate deterministic reference fixtures for parity regression testing.

These fixtures are Python-native reference artifacts used to enforce numerical
regression stability in CI. They are designed to be replaced/augmented with
MATLAB-generated references when exported from the gold-standard toolbox.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms
from nstat.fit import FitResult, FitSummary
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/parity/fixtures"),
        help="Directory where fixture files and manifest are written.",
    )
    return parser.parse_args()



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _write_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "path": str(path.as_posix()),
        "sha256": _sha256(path),
    }



def _fixture_analysis_poisson(output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    n = 2500
    dt = 0.01
    x = rng.normal(size=n)
    X = x[:, None]
    true_intercept = np.log(10.0)
    true_beta = np.array([0.45])
    lam = np.exp(true_intercept + X @ true_beta)
    y = rng.poisson(lam * dt).astype(float)

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
    pred = fit.predict(X)

    file_info = _write_npz(
        output_dir / "analysis_poisson_glm.npz",
        X=X,
        y=y,
        dt=np.array([dt], dtype=float),
        expected_coefficients=fit.coefficients,
        expected_intercept=np.array([fit.intercept], dtype=float),
        expected_log_likelihood=np.array([fit.log_likelihood], dtype=float),
        expected_rate=pred,
    )
    file_info["name"] = "analysis_poisson_glm"
    file_info["source"] = "python_seeded_reference"
    return file_info



def _fixture_decoding_posterior(output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    n_units = 12
    n_states = 18
    n_time = 180

    centers = np.linspace(0.0, n_states - 1, n_units)
    widths = np.full(n_units, 2.0)
    states = np.arange(n_states)[None, :]
    tuning = 0.05 + 0.35 * np.exp(-0.5 * ((states - centers[:, None]) / widths[:, None]) ** 2)

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

    spike_counts = np.zeros((n_units, n_time), dtype=float)
    for t in range(n_time):
        spike_counts[:, t] = rng.poisson(tuning[:, latent[t]])

    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=spike_counts,
        tuning_rates=tuning,
        transition=transition,
    )

    file_info = _write_npz(
        output_dir / "decoding_state_posterior.npz",
        spike_counts=spike_counts,
        tuning_rates=tuning,
        transition=transition,
        expected_decoded=decoded,
        expected_posterior=posterior,
    )
    file_info["name"] = "decoding_state_posterior"
    file_info["source"] = "python_seeded_reference"
    return file_info



def _fixture_trial_alignment(output_dir: Path) -> dict[str, Any]:
    time = np.linspace(0.0, 1.0, 501)
    cov = Covariate(time=time, data=np.sin(2 * np.pi * 3.0 * time), name="stim", labels=["stim"])
    spikes = SpikeTrain(spike_times=np.array([0.10, 0.22, 0.51, 0.79]), t_start=0.0, t_end=1.0, name="u1")
    trial = Trial(
        spikes=SpikeTrainCollection([spikes]),
        covariates=CovariateCollection([cov]),
    )
    bin_size = 0.02
    tb, y, X = trial.aligned_binned_observation(bin_size_s=bin_size, unit_index=0, mode="count")

    file_info = _write_npz(
        output_dir / "trial_alignment.npz",
        time=time,
        cov_data=cov.data,
        spike_times=spikes.spike_times,
        bin_size=np.array([bin_size], dtype=float),
        expected_time_bins=tb,
        expected_counts=y,
        expected_design=X,
    )
    file_info["name"] = "trial_alignment"
    file_info["source"] = "python_seeded_reference"
    return file_info



def _fixture_fit_summary_structure(output_dir: Path) -> dict[str, Any]:
    coefficients = np.array(
        [
            [0.20, -0.10],
            [0.45, 0.30],
            [-0.05, 0.25],
        ],
        dtype=float,
    )
    intercepts = np.array([-1.0, -0.7, -0.9], dtype=float)
    log_likelihoods = np.array([-12.0, -10.8, -11.5], dtype=float)
    n_samples = np.array([200, 200, 200], dtype=int)
    n_parameters = np.array([3, 3, 3], dtype=int)
    fit_types = np.array(["binomial", "binomial", "binomial"], dtype="<U16")
    labels = np.array(
        [
            ["stim", "hist"],
            ["stim", "hist"],
            ["stim", "ctx"],
        ],
        dtype="<U16",
    )

    results: list[FitResult] = []
    for i in range(coefficients.shape[0]):
        results.append(
            FitResult(
                coefficients=coefficients[i],
                intercept=float(intercepts[i]),
                fit_type=str(fit_types[i]),
                log_likelihood=float(log_likelihoods[i]),
                n_samples=int(n_samples[i]),
                n_parameters=int(n_parameters[i]),
                parameter_labels=[str(labels[i, 0]), str(labels[i, 1])],
            )
        )

    summary = FitSummary(results=results)
    coeff_mat, unique_labels, se_mat = summary.get_coeffs()
    hist_counts, hist_edges, hist_percent_sig = summary.bin_coeffs(min_val=-1.0, max_val=1.0, bin_size=0.2)

    file_info = _write_npz(
        output_dir / "fit_summary_structure.npz",
        result_coefficients=coefficients,
        result_intercepts=intercepts,
        result_log_likelihoods=log_likelihoods,
        result_n_samples=n_samples,
        result_n_parameters=n_parameters,
        result_fit_types=fit_types,
        result_labels=labels,
        expected_unique_labels=np.array(unique_labels, dtype="<U16"),
        expected_coeff_matrix=coeff_mat,
        expected_se_matrix=se_mat,
        expected_hist_counts=hist_counts,
        expected_hist_edges=hist_edges,
        expected_hist_percent_sig=hist_percent_sig,
    )
    file_info["name"] = "fit_summary_structure"
    file_info["source"] = "python_seeded_reference"
    return file_info


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = [
        _fixture_analysis_poisson(output_dir),
        _fixture_decoding_posterior(output_dir),
        _fixture_trial_alignment(output_dir),
        _fixture_fit_summary_structure(output_dir),
    ]

    for row in fixtures:
        fixture_path = Path(row["path"]).resolve()
        row["path"] = fixture_path.relative_to(repo_root).as_posix()

    manifest = {
        "version": 1,
        "policy": {
            "purpose": "regression_guard",
            "matlab_reference_upgrade_expected": True,
        },
        "fixtures": fixtures,
    }
    manifest_path = output_dir / "manifest.yml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote fixture manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
