"""Demo: state-space estimation via the Dynamax bridge.

Exercises every routine in ``nstat.extras.em.dynamax_bridge`` on
synthetic fixtures with known parameters:

1. ``fit_linear_gaussian_em``  — KF_EM equivalent (Gaussian observations)
2. ``cmgf_poisson_filter`` / ``cmgf_poisson_smoother`` — point-process
   inference on a known Poisson-LGSSM (PPDecodeFilter / PP_fixedIntervalSmoother)
3. ``fit_point_process_em``    — PP_EM equivalent (Poisson observations)
4. ``fit_hybrid_em``           — mPPCO_EM equivalent (Poisson + Gaussian)
5. ``point_process_predictive_ll`` — true held-out predictive log-
   likelihood, the honest fit-quality metric (pure NumPy, no dynamax)

Together these close the AUDIT_REPORT.md §3.2 gap (KF_EM / PP_EM /
mPPCO_EM, 19 unported MATLAB methods).

Run::

    pip install nstat-toolbox[dynamax]   # pulls JAX (~200 MB)
    python examples/extras/em_dynamax_demo.py
"""
from __future__ import annotations

import numpy as np


def _demo_linear_gaussian_em() -> int:
    """KF_EM equivalent — linear-Gaussian observations."""
    from nstat.extras.em.dynamax_bridge import fit_linear_gaussian_em

    rng = np.random.default_rng(0)
    T, state_dim, emission_dim = 300, 2, 2
    A_true = np.array([[0.95, 0.05], [-0.05, 0.95]])
    C_true = np.eye(emission_dim)
    Q_true = np.eye(state_dim) * 0.02
    R_true = np.eye(emission_dim) * 0.1

    x = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim))
    x[0] = rng.multivariate_normal(np.zeros(state_dim), np.eye(state_dim))
    y[0] = C_true @ x[0] + rng.multivariate_normal(np.zeros(emission_dim), R_true)
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q_true)
        y[t] = C_true @ x[t] + rng.multivariate_normal(np.zeros(emission_dim), R_true)

    result = fit_linear_gaussian_em(y, state_dim=state_dim, n_iter=30, seed=0)
    lls = result.log_likelihoods
    print(f"[KF_EM]   {result.n_iter} iters, "
          f"ll {lls[0]:.1f} → {lls[-1]:.1f}, Δll_min={np.diff(lls).min():.2e}")
    return 0 if np.all(np.diff(lls) >= -1e-6) else 1


def _simulate_poisson_lgssm(T=300, state_dim=2, emission_dim=2, seed=1):
    """Synthetic Poisson-LGSSM: linear-Gaussian state, Poisson spike counts."""
    rng = np.random.default_rng(seed)
    A = np.eye(state_dim) * 0.95
    C = np.eye(emission_dim, state_dim) * 0.3
    Q = np.eye(state_dim) * 0.05
    x0 = np.zeros(state_dim)
    P0 = np.eye(state_dim) * 0.1
    x = np.zeros((T, state_dim))
    y = np.zeros((T, emission_dim), dtype=int)
    x[0] = rng.multivariate_normal(x0, P0)
    y[0] = rng.poisson(np.exp(C @ x[0]))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(state_dim), Q)
        y[t] = rng.poisson(np.exp(C @ x[t]))
    return y, A, C, Q, x0, P0, x


def _demo_cmgf_inference() -> int:
    """PPDecodeFilter / PP_fixedIntervalSmoother — inference on a known model."""
    from nstat.extras.em.dynamax_bridge import (
        cmgf_poisson_filter, cmgf_poisson_smoother,
    )
    y, A, C, Q, x0, P0, x_true = _simulate_poisson_lgssm()
    filt = cmgf_poisson_filter(y, A, C, Q, x0, P0)
    smooth = cmgf_poisson_smoother(y, A, C, Q, x0, P0)
    mse_f = float(np.mean((filt.state_means - x_true) ** 2))
    mse_s = float(np.mean((smooth.state_means - x_true) ** 2))
    print(f"[CMGF]    filter MSE={mse_f:.3f}, smoother MSE={mse_s:.3f} "
          f"(smoother ≤ filter: {mse_s <= mse_f + 1e-9})")
    return 0


def _demo_point_process_em() -> int:
    """PP_EM equivalent — learn A, C, Q, x0, P0 from spike counts alone."""
    from nstat.extras.em.dynamax_bridge import fit_point_process_em
    y, *_ = _simulate_poisson_lgssm()
    result = fit_point_process_em(y, state_dim=2, n_iter=20, seed=0)
    lls = result.marginal_log_likelihoods
    print(f"[PP_EM]   {result.n_iter} iters, "
          f"ll {lls[0]:.1f} → {lls[-1]:.1f}, Ĉ shape={result.observation_matrix.shape}")
    return 0


def _demo_hybrid_em() -> int:
    """mPPCO_EM equivalent — Poisson + Gaussian channels share one latent."""
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em
    y_pp, A, C, Q, x0, P0, x_true = _simulate_poisson_lgssm()
    rng = np.random.default_rng(2)
    # Gaussian (LFP-like) channel driven by the same latent state.
    C_g = np.array([[1.0, 0.0]])
    y_g = x_true @ C_g.T + rng.normal(scale=0.3, size=(x_true.shape[0], 1))
    result = fit_hybrid_em(y_pp, y_g, state_dim=2, n_iter=20, seed=0)
    lls = result.marginal_log_likelihoods
    print(f"[mPPCO_EM] {result.n_iter} iters, ll {lls[0]:.1f} → {lls[-1]:.1f}, "
          f"Ĉ_p={result.poisson_observation_matrix.shape} "
          f"Ĉ_g={result.gaussian_observation_matrix.shape}")
    return 0


def _demo_predictive_ll() -> int:
    """Held-out predictive log-likelihood — the honest quality metric.

    Pure NumPy (no dynamax): scores the *true* Poisson likelihood of
    held-out spikes under the one-step-ahead predictive state, and shows
    it ranks the true generating parameters above a flat-rate model.
    """
    from nstat.extras.em.dynamax_bridge import point_process_predictive_ll

    y, A, C, Q, x0, P0, _ = _simulate_poisson_lgssm()
    y_train, y_test = y[:240], y[240:]
    true = point_process_predictive_ll(y_test, A, C, Q, x0, P0).total
    flat = point_process_predictive_ll(y_test, A, C * 0.0, Q, x0, P0).total
    print(f"[PredLL]  held-out: true-params={true:.1f} > flat-rate={flat:.1f} "
          f"({true > flat})")
    return 0 if true > flat else 1


def main() -> int:
    try:
        import nstat.extras.em.dynamax_bridge  # noqa: F401
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    print("nstat.extras.em.dynamax_bridge — full state-space demo\n")
    # The predictive-LL diagnostic is pure NumPy — runs without dynamax.
    rc = _demo_predictive_ll()
    try:
        rc |= _demo_linear_gaussian_em()
        rc |= _demo_cmgf_inference()
        rc |= _demo_point_process_em()
        rc |= _demo_hybrid_em()
    except ImportError as exc:
        print(f"dynamax missing (EM/inference demos skipped): {exc}")
        return rc
    print("\nAll EM / inference routines ran." if rc == 0
          else "\nA routine reported a problem.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
