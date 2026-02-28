from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms


def _tol(path: str) -> float:
    payload = yaml.safe_load(Path("tests/parity/tolerances.yml").read_text(encoding="utf-8"))
    value: object = payload
    for key in path.split("."):
        value = value[key]
    return float(value)


def test_poisson_glm_parameter_recovery() -> None:
    rng = np.random.default_rng(42)
    n_samples = 4000
    dt = 0.01

    x = rng.normal(size=n_samples)
    x = (x - np.mean(x)) / np.std(x)
    X = x[:, None]

    true_intercept = np.log(12.0)
    true_beta = np.array([0.35])
    lam = np.exp(true_intercept + X @ true_beta)
    y = rng.poisson(lam * dt).astype(float)

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
    coeff_tol = _tol("poisson_glm.coefficient_abs_tol")
    assert abs(float(fit.coefficients[0]) - float(true_beta[0])) <= coeff_tol

    pred_rate = fit.predict(X)
    true_rate = lam
    rel_err = np.mean(np.abs(pred_rate - true_rate) / np.maximum(true_rate, 1e-9))
    assert rel_err <= _tol("poisson_glm.rate_rel_tol")


def test_binomial_glm_parameter_recovery() -> None:
    rng = np.random.default_rng(24)
    n_samples = 3500

    x = rng.normal(size=n_samples)
    x = (x - np.mean(x)) / np.std(x)
    X = x[:, None]

    true_intercept = -1.9
    true_beta = np.array([1.1])
    p = expit(true_intercept + X @ true_beta)
    y = rng.binomial(1, p).astype(float)

    fit = Analysis.fit_glm(X=X, y=y, fit_type="binomial")
    coeff_tol = _tol("binomial_glm.coefficient_abs_tol")
    assert abs(float(fit.coefficients[0]) - float(true_beta[0])) <= coeff_tol

    pred_p = fit.predict(X)
    mae = np.mean(np.abs(pred_p - p))
    assert mae <= _tol("binomial_glm.probability_abs_tol")


def test_decode_state_posterior_random_walk() -> None:
    rng = np.random.default_rng(7)
    n_units = 20
    n_states = 25
    n_time = 250

    centers = np.linspace(0.0, n_states - 1, n_units)
    widths = np.full(n_units, 2.5)
    states = np.arange(n_states)[None, :]
    tuning = 0.05 + 0.40 * np.exp(-0.5 * ((states - centers[:, None]) / widths[:, None]) ** 2)

    transition = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        for j, w in [(i - 1, 0.15), (i, 0.70), (i + 1, 0.15)]:
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

    assert posterior.shape == (n_states, n_time)
    assert np.allclose(np.sum(posterior, axis=0), 1.0, atol=1e-6)

    rmse = np.sqrt(np.mean((decoded - latent) ** 2))
    nrmse = rmse / float(max(n_states - 1, 1))
    assert nrmse <= _tol("decoding.normalized_rmse_tol")


def test_compute_spike_rate_cis_detects_large_trial_shift() -> None:
    rng = np.random.default_rng(77)
    n_bins = 600
    n_trials = 12
    low = rng.binomial(1, 0.03, size=(n_trials // 2, n_bins))
    high = rng.binomial(1, 0.10, size=(n_trials // 2, n_bins))
    matrix = np.vstack([low, high]).astype(float)

    _, _, sig = DecodingAlgorithms.compute_spike_rate_cis(matrix, alpha=0.05)

    low_idx = range(0, n_trials // 2)
    high_idx = range(n_trials // 2, n_trials)

    between = [sig[i, j] for i in low_idx for j in high_idx]
    between_rate = float(np.mean(between))
    required = 1.0 - _tol("decoding.ci_coverage_tol")
    assert between_rate >= required
