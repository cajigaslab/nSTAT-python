#!/usr/bin/env python3
"""Demo: motor-BCI decoder calibration -- naive-KF vs. ReFIT-KF recalibration.

**Question (design spec Sec. 4.3).** How should a decoder be *calibrated*
from neural + behavioral training data, and does closed-loop-aware (ReFIT)
recalibration correct a systematic directional bias baked into an
open-loop-calibrated decoder's training label?

**Scenario.**  A synthetic Wu-2006-style linear-Gaussian arm state
(position + velocity) is coupled to a 40-unit population with **linear
velocity encoding** (Wu, Gao, Bienenstock, Donoghue & Black 2006): each
unit's firing rate is a noisy linear function of reach velocity only, with
preferred directions spread uniformly around the circle -- there is no
position tuning, matching Wu (2006)'s decoding model.  Two Kalman-filter
decoders are calibrated from the *same* recorded population activity but
different assumed kinematic labels:

- **naive-KF** assumes the classic "open-loop"/passive-calibration label:
  a constant-velocity, never-decelerating vector toward the target, with a
  directional drift that is never corrected -- a stand-in for calibration
  blocks that never captured the deceleration/stop epoch of a real reach.
- **ReFIT-KF** relabels the *same* training data with the intended,
  target-directed, properly decelerating kinematics (Gilja, Nuyujukian,
  Chestek, Cunningham, Yu, Fan, Churchland, Kaufman, Kao, Ryu & Shenoy
  2012's "recalibrated feedback intention-trained Kalman filter",
  ReFIT-KF), i.e. it knows the true target and assumes a canonical
  decelerating reach toward it.

Both decoders share the same generic constant-velocity motion model (only
the *calibrated observation model* differs) and are evaluated on held-out
reach-and-hold ("target acquisition") trials neither decoder trained on.
The naive decoder's uncorrected directional bias makes it mis-aim on
every reach, so it never settles inside the acquisition radius and fails
to acquire any of the 8 targets; the ReFIT decoder tracks the target
closely and acquires and holds all 8 -- illustrating, in miniature, the
ReFIT-relabeling *principle* of Gilja et al. (2012): calibrating the
observation model on the inferred intended target-directed kinematics,
rather than on the raw open-loop/passive training label, is what
corrects a systematically mis-aiming decoder. (This demo measures
hold-period error, endpoint error, path efficiency, and fraction of
targets acquired -- it does not compute a target-acquisition-time metric,
and does not reproduce Gilja et al. (2012)'s specific acquisition-time
speedup; the two decoders' path-efficiency scores are in fact nearly
identical, since the naive failure mode here is a directional bias, not
overshoot or oscillation.)

This naive-vs-ReFIT contrast is **pure NumPy/SciPy** (no optional
dependency) and always runs, including in environments without the
``[dynamax]`` extra.

**Appendix: JAX-backed state-space EM.**  The demo also exercises every
routine in :mod:`nstat.extras.em.dynamax_bridge` -- the JAX-backed
KF_EM / PP_EM / mPPCO_EM equivalents used to actually *fit* transition and
observation matrices by EM (as opposed to the closed-form linear
regression used for the naive/ReFIT contrast above), on synthetic
fixtures with known parameters:

1. ``fit_linear_gaussian_em``  -- KF_EM equivalent (Gaussian observations)
2. ``cmgf_poisson_filter`` / ``cmgf_poisson_smoother`` -- point-process
   inference on a known Poisson-LGSSM (PPDecodeFilter / PP_fixedIntervalSmoother),
   the point-process adaptive-filtering framework of Eden, Frank, Barbieri,
   Solo & Brown (2004)
3. ``fit_point_process_em``    -- PP_EM equivalent (Poisson observations)
4. ``fit_hybrid_em``           -- mPPCO_EM equivalent (Poisson + Gaussian)
5. ``point_process_predictive_ll`` -- true held-out predictive log-
   likelihood, the honest fit-quality metric (pure NumPy, no dynamax)

Together these close the AUDIT_REPORT.md Sec. 3.2 gap (KF_EM / PP_EM /
mPPCO_EM, 19 unported MATLAB methods).  This appendix requires the
optional ``[dynamax]`` extra (JAX, ~200 MB) and is gracefully skipped
(not a failure) when it is absent -- the naive-KF vs. ReFIT-KF contrast
above already ran fully without it.

Run::

    python examples/extras/em_dynamax_demo.py                    # interactive
    python examples/extras/em_dynamax_demo.py --no-display
    python examples/extras/em_dynamax_demo.py --export-figures

    pip install nstat-toolbox[dynamax]   # optional: pulls JAX (~200 MB)
    python examples/extras/em_dynamax_demo.py   # also runs the EM appendix

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to ``docs/figures/extras/em_dynamax/``) and
are NOT committed to the repository -- the export flag exists for local
inspection only.  CI never invokes it.

Cross-reference: :mod:`nstat.extras.validation.pykalman_bridge` /
``examples/extras/validation_pykalman_demo.py`` cross-validates nstat's
own Kalman filter (used here to decode) against ``pykalman``.

References:
- Wu W, Gao Y, Bienenstock E, Donoghue JP, Black MJ (2006). *Bayesian
  population decoding of motor cortical activity using a Kalman filter.*
  Neural Computation 18(1):80-118.
- Gilja V, Nuyujukian P, Chestek CA, Cunningham JP, Yu BM, Fan JM,
  Churchland MM, Kaufman MT, Kao JC, Ryu SI, Shenoy KV (2012). *A
  high-performance neural prosthesis enabled by control algorithm
  design.* Nature Neuroscience 15(12):1752-1757. [ReFIT-KF]
- Eden UT, Frank LM, Barbieri R, Solo V, Brown EN (2004). *Dynamic
  analysis of neural encoding by point process adaptive filtering.*
  Neural Computation 16(5):971-998.
- Kim SP, Simeral JD, Hochberg LR, Donoghue JP, Friehs GM, Black MJ
  (2011). *Point-and-click cursor control with an intracortical neural
  interface system by humans with tetraplegia.* IEEE Transactions on
  Neural Systems and Rehabilitation Engineering 19(2):193-203.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# BCI decoder-calibration scenario constants (Wu-2006 arm state; population
# with linear velocity encoding; fully synthetic -- no real recording).
# ---------------------------------------------------------------------------
N_TARGETS = 8               # 8-direction center-out task
N_REPS_TRAIN = 6             # training repeats per target
R_REACH = 10.0               # target distance (arbitrary length units)
T_REACH = 1.2                # s, nominal reach duration (bell-shaped velocity)
T_HOLD = 0.8                 # s, post-reach hold ("target acquisition") window
DT = 0.02                    # s, 50 Hz bin width
N_UNITS = 40                 # population size (cf. Kim et al. 2011's ~40-unit ensemble)
GAIN0 = 0.18                 # mean per-unit velocity-tuning gain
NOISE_SIGMA = 0.6            # observation noise std
NAIVE_ANGLE_BIAS_DEG = 35.0  # max directional drift baked into the naive/open-loop label
ACQUISITION_RADIUS = 1.0     # success = decoded cursor settles within this radius of target
KF_VELOCITY_PERSISTENCE = 0.97  # shared, generic constant-velocity motion-model damping


def _minimum_jerk_s(tau: np.ndarray) -> np.ndarray:
    """Canonical minimum-jerk normalized position profile, tau in [0, 1]."""
    return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5


def _minimum_jerk_sdot(tau: np.ndarray) -> np.ndarray:
    """d/d(tau) of the minimum-jerk profile (zero at tau=0 and tau=1)."""
    return 30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _build_population(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A Wu (2006)-style population with linear velocity encoding only.

    Returns ``(C_true, baselines)`` where ``C_true`` is ``(N_UNITS, 4)``
    with zero position columns (no position tuning) and velocity columns
    ``gain_i * [cos(pd_i), sin(pd_i)]``.
    """
    preferred_dirs = np.linspace(0.0, 2.0 * np.pi, N_UNITS, endpoint=False)
    gains = GAIN0 * rng.uniform(0.8, 1.2, size=N_UNITS)
    baselines = rng.uniform(0.0, 0.2, size=N_UNITS)
    C_true = np.zeros((N_UNITS, 4))
    C_true[:, 2] = gains * np.cos(preferred_dirs)
    C_true[:, 3] = gains * np.sin(preferred_dirs)
    return C_true, baselines


def _center_out_targets() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, N_TARGETS, endpoint=False)
    return R_REACH * np.c_[np.cos(angles), np.sin(angles)]


def _true_reach_kinematics(direction: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """True (intended) minimum-jerk position/velocity profile toward *direction*."""
    tau = (np.arange(n_bins) * DT) / T_REACH
    p = np.outer(_minimum_jerk_s(tau), direction) * R_REACH
    v = np.outer(_minimum_jerk_sdot(tau) / T_REACH, direction) * R_REACH
    return p, v


def _calibrate_decoders(
    rng: np.random.Generator, C_true: np.ndarray, baselines: np.ndarray, targets: np.ndarray,
) -> dict:
    """Fit naive and ReFIT observation models from the *same* simulated
    neural training data, differing only in the assumed kinematic label.

    naive label: constant-speed, non-decelerating, with a directional
    drift that grows across the reach and is never corrected -- a stand-in
    for open-loop/passive calibration blocks that never captured the
    deceleration/stop epoch of a real reach.

    ReFIT label: the true target-directed, properly decelerating
    kinematics (Gilja et al. 2012) -- ReFIT knows the target and assumes a
    canonical decelerating reach toward it.
    """
    n_bins = int(round(T_REACH / DT))
    tau = (np.arange(n_bins) * DT) / T_REACH
    naive_angle = np.deg2rad(NAIVE_ANGLE_BIAS_DEG) * tau  # 0 -> max over the reach
    const_speed = R_REACH / T_REACH

    y_rows: list[np.ndarray] = []
    naive_rows: list[np.ndarray] = []
    refit_rows: list[np.ndarray] = []
    for k in range(N_TARGETS):
        direction = targets[k] / R_REACH
        for _ in range(N_REPS_TRAIN):
            speed_scale = rng.uniform(0.85, 1.15)
            _, v_true = _true_reach_kinematics(direction, n_bins)
            v_true = v_true * speed_scale
            y = (
                v_true @ C_true[:, 2:].T + baselines
                + rng.normal(scale=NOISE_SIGMA, size=(n_bins, N_UNITS))
            )
            v_naive = np.stack(
                [_rotation_matrix(a) @ direction * const_speed for a in naive_angle], axis=0,
            )
            y_rows.append(y)
            naive_rows.append(v_naive)
            refit_rows.append(v_true)

    Y = np.concatenate(y_rows, axis=0)
    L_naive = np.concatenate(naive_rows, axis=0)
    L_refit = np.concatenate(refit_rows, axis=0)

    def _fit_observation_model(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = np.c_[np.ones(label.shape[0]), label]
        theta, *_ = np.linalg.lstsq(design, Y, rcond=None)
        resid = Y - design @ theta
        b_hat = theta[0]
        gain_hat = theta[1:3]
        r_hat = np.maximum(np.var(resid, axis=0), 1e-3)
        c_hat = np.zeros((N_UNITS, 4))
        c_hat[:, 2:] = gain_hat.T
        return c_hat, b_hat, r_hat

    c_naive, b_naive, r_naive = _fit_observation_model(L_naive)
    c_refit, b_refit, r_refit = _fit_observation_model(L_refit)
    return {
        "naive": {"C": c_naive, "b": b_naive, "R": r_naive},
        "refit": {"C": c_refit, "b": b_refit, "R": r_refit},
    }


def _decode_closed_loop(
    rng: np.random.Generator,
    C_true: np.ndarray,
    baselines: np.ndarray,
    targets: np.ndarray,
    calib: dict,
) -> tuple[dict, list[np.ndarray]]:
    """Decode held-out reach + hold ("target acquisition") trials with both
    decoders.  Both share the same generic constant-velocity motion model
    (``nstat.decoding_algorithms.DecodingAlgorithms.kalman_filter``); only
    the calibrated observation model differs.
    """
    from nstat.decoding_algorithms import DecodingAlgorithms

    n_bins_reach = int(round(T_REACH / DT))
    n_bins_hold = int(round(T_HOLD / DT))
    n_bins = n_bins_reach + n_bins_hold

    A = np.array([
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, KF_VELOCITY_PERSISTENCE, 0.0],
        [0.0, 0.0, 0.0, KF_VELOCITY_PERSISTENCE],
    ])
    Q = np.diag([1e-6, 1e-6, 1.0, 1.0])
    x0 = np.zeros(4)
    P0 = np.diag([1.0, 1.0, 5.0, 5.0])

    results = {
        name: {"trajectories": [], "endpoint_err": [], "hold_err": [],
               "path_efficiency": [], "acquired": []}
        for name in ("naive", "refit")
    }
    true_trajectories: list[np.ndarray] = []

    for k in range(N_TARGETS):
        direction = targets[k] / R_REACH
        p_reach, v_reach = _true_reach_kinematics(direction, n_bins_reach)
        v_test = np.concatenate([v_reach, np.zeros((n_bins_hold, 2))], axis=0)
        p_true = np.concatenate([p_reach, np.tile(targets[k], (n_bins_hold, 1))], axis=0)
        y_test = (
            v_test @ C_true[:, 2:].T + baselines
            + rng.normal(scale=NOISE_SIGMA, size=(n_bins, N_UNITS))
        )
        true_trajectories.append(p_true)

        for name in ("naive", "refit"):
            c_hat, b_hat, r_hat = calib[name]["C"], calib[name]["b"], calib[name]["R"]
            out = DecodingAlgorithms.kalman_filter(
                observations=y_test - b_hat, transition=A, observation_matrix=c_hat,
                q_cov=Q, r_cov=np.diag(r_hat), x0=x0, p0=P0,
            )
            p_dec = out["state"][:, :2]
            hold_dist = np.linalg.norm(p_dec[n_bins_reach:] - targets[k], axis=1)
            arc_length = np.sum(np.linalg.norm(np.diff(p_dec, axis=0), axis=1))
            results[name]["trajectories"].append(p_dec)
            results[name]["endpoint_err"].append(float(np.linalg.norm(p_dec[-1] - targets[k])))
            results[name]["hold_err"].append(float(hold_dist.mean()))
            results[name]["path_efficiency"].append(float(R_REACH / max(arc_length, 1e-9)))
            results[name]["acquired"].append(bool(hold_dist.mean() < ACQUISITION_RADIUS))

    return results, true_trajectories


# ---------------------------------------------------------------------------
# Appendix: JAX-backed Dynamax state-space EM routines (opt-in, [dynamax]).
# ---------------------------------------------------------------------------


def _demo_linear_gaussian_em() -> int:
    """KF_EM equivalent -- linear-Gaussian observations."""
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
    print(f"  [KF_EM]   {result.n_iter} iters, "
          f"ll {lls[0]:.1f} -> {lls[-1]:.1f}, dll_min={np.diff(lls).min():.2e}")
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
    """PPDecodeFilter / PP_fixedIntervalSmoother -- inference on a known model
    (Eden, Frank, Barbieri, Solo & Brown 2004's point-process adaptive-filter
    framework)."""
    from nstat.extras.em.dynamax_bridge import (
        cmgf_poisson_filter, cmgf_poisson_smoother,
    )
    y, A, C, Q, x0, P0, x_true = _simulate_poisson_lgssm()
    filt = cmgf_poisson_filter(y, A, C, Q, x0, P0)
    smooth = cmgf_poisson_smoother(y, A, C, Q, x0, P0)
    mse_f = float(np.mean((filt.state_means - x_true) ** 2))
    mse_s = float(np.mean((smooth.state_means - x_true) ** 2))
    print(f"  [CMGF]    filter MSE={mse_f:.3f}, smoother MSE={mse_s:.3f} "
          f"(smoother <= filter: {mse_s <= mse_f + 1e-9})")
    return 0


def _demo_point_process_em() -> int:
    """PP_EM equivalent -- learn A, C, Q, x0, P0 from spike counts alone."""
    from nstat.extras.em.dynamax_bridge import fit_point_process_em
    y, *_ = _simulate_poisson_lgssm()
    result = fit_point_process_em(y, state_dim=2, n_iter=20, seed=0)
    lls = result.marginal_log_likelihoods
    print(f"  [PP_EM]   {result.n_iter} iters, "
          f"ll {lls[0]:.1f} -> {lls[-1]:.1f}, C_hat shape={result.observation_matrix.shape}")
    return 0


def _demo_hybrid_em() -> int:
    """mPPCO_EM equivalent -- Poisson + Gaussian channels share one latent."""
    from nstat.extras.em.dynamax_bridge import fit_hybrid_em
    y_pp, A, C, Q, x0, P0, x_true = _simulate_poisson_lgssm()
    rng = np.random.default_rng(2)
    # Gaussian (LFP-like) channel driven by the same latent state.
    C_g = np.array([[1.0, 0.0]])
    y_g = x_true @ C_g.T + rng.normal(scale=0.3, size=(x_true.shape[0], 1))
    result = fit_hybrid_em(y_pp, y_g, state_dim=2, n_iter=20, seed=0)
    lls = result.marginal_log_likelihoods
    print(f"  [mPPCO_EM] {result.n_iter} iters, ll {lls[0]:.1f} -> {lls[-1]:.1f}, "
          f"C_p_hat={result.poisson_observation_matrix.shape} "
          f"C_g_hat={result.gaussian_observation_matrix.shape}")
    return 0


def _demo_predictive_ll() -> int:
    """Held-out predictive log-likelihood -- the honest quality metric.

    Pure NumPy (no dynamax): scores the *true* Poisson likelihood of
    held-out spikes under the one-step-ahead predictive state, and shows
    it ranks the true generating parameters above a flat-rate model.
    """
    from nstat.extras.em.dynamax_bridge import point_process_predictive_ll

    y, A, C, Q, x0, P0, _ = _simulate_poisson_lgssm()
    y_train, y_test = y[:240], y[240:]
    true = point_process_predictive_ll(y_test, A, C, Q, x0, P0).total
    flat = point_process_predictive_ll(y_test, A, C * 0.0, Q, x0, P0).total
    print(f"  [PredLL]  held-out: true-params={true:.1f} > flat-rate={flat:.1f} "
          f"({true > flat})")
    return 0 if true > flat else 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_demo(
    *,
    seed: int = 20260703,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
    plot_style: str = "legacy",
) -> dict:
    """Run the naive-KF vs. ReFIT-KF BCI calibration contrast, then the
    optional dynamax-backed EM appendix.

    Returns
    -------
    dict
        ``{"summary", "refit_better_than_naive", "dynamax_appendix_ok",
        "figure_paths"}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style

    print("=" * 72)
    print("Motor-BCI decoder calibration: naive-KF vs. ReFIT-KF (Gilja et al. 2012)")
    print("=" * 72)
    print(
        f"Wu (2006)-style {N_UNITS}-unit population, linear velocity encoding, "
        f"{N_TARGETS}-target center-out reach+hold task (fully synthetic -- "
        "no real recording)."
    )

    rng = np.random.default_rng(seed)
    C_true, baselines = _build_population(rng)
    targets = _center_out_targets()
    calib = _calibrate_decoders(rng, C_true, baselines, targets)

    gain_true = float(np.mean(np.abs(C_true[:, 2:])))
    gain_naive = float(np.mean(np.abs(calib["naive"]["C"][:, 2:])))
    gain_refit = float(np.mean(np.abs(calib["refit"]["C"][:, 2:])))
    print()
    print("Observation-model calibration (mean |velocity-tuning gain|, residual variance):")
    print(f"  {'':>10} | {'true':>8} | {'naive':>8} | {'refit':>8}")
    print(f"  {'|gain|':>10} | {gain_true:8.4f} | {gain_naive:8.4f} | {gain_refit:8.4f}")
    print(f"  {'resid var':>10} | {'':>8} | {calib['naive']['R'].mean():8.4f} | "
          f"{calib['refit']['R'].mean():8.4f}")

    decode, true_trajectories = _decode_closed_loop(rng, C_true, baselines, targets, calib)

    print()
    print(f"Closed-loop target-acquisition recovery (mean over {N_TARGETS} targets, "
          f"acquisition radius={ACQUISITION_RADIUS:.1f}):")
    print(f"  {'decoder':>10} | {'hold err':>9} | {'endpoint err':>13} | "
          f"{'path eff':>9} | {'acquired':>9}")
    summary: dict[str, dict] = {}
    for name in ("naive", "refit"):
        hold_err = float(np.mean(decode[name]["hold_err"]))
        endpoint_err = float(np.mean(decode[name]["endpoint_err"]))
        path_eff = float(np.mean(decode[name]["path_efficiency"]))
        acquired = int(sum(decode[name]["acquired"]))
        print(f"  {name:>10} | {hold_err:9.3f} | {endpoint_err:13.3f} | "
              f"{path_eff:9.3f} | {acquired:d}/{N_TARGETS}")
        summary[name] = {
            "hold_err_mean": hold_err,
            "endpoint_err_mean": endpoint_err,
            "path_efficiency_mean": path_eff,
            "targets_acquired": acquired,
        }

    refit_better = summary["refit"]["hold_err_mean"] < summary["naive"]["hold_err_mean"]
    print()
    print(
        f"ReFIT acquisition error < naive-KF: {summary['refit']['hold_err_mean']:.3f} < "
        f"{summary['naive']['hold_err_mean']:.3f}  ({'PASS' if refit_better else 'FAIL'})"
    )

    # ---- Figures ----
    # === FIGURE: fig01_bci_reach_trajectories.png ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True, sharey=True)
    panel_specs = (
        ("naive", "Naive-KF (open-loop calibration)", "tab:red"),
        ("refit", "ReFIT-KF (target-relabeled recalibration)", "tab:green"),
    )
    for ax, (name, title, color) in zip(axes1, panel_specs):
        for k in range(N_TARGETS):
            ax.plot(
                true_trajectories[k][:, 0], true_trajectories[k][:, 1],
                color="black", lw=1.0, ls=":", alpha=0.6,
                label="true reach+hold" if k == 0 else None,
            )
            traj = decode[name]["trajectories"][k]
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.7,
                     label="decoded" if k == 0 else None)
        ax.scatter(targets[:, 0], targets[:, 1], color="tab:blue", s=30, zorder=5,
                    label="target")
        ax.scatter([0.0], [0.0], color="black", marker="+", s=60, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7)
    axes1[0].set_ylabel("y (cm)")
    fig1.suptitle("True vs. decoded target-acquisition trajectories (8-target center-out)")
    # === END FIGURE ===

    # === FIGURE: fig02_acquisition_metrics.png ===
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x_pos = np.arange(N_TARGETS)
    width = 0.35
    ax2a.bar(x_pos - width / 2, decode["naive"]["hold_err"], width,
              color="tab:red", label="naive-KF")
    ax2a.bar(x_pos + width / 2, decode["refit"]["hold_err"], width,
              color="tab:green", label="ReFIT-KF")
    ax2a.axhline(ACQUISITION_RADIUS, color="black", ls="--", lw=1.0,
                  label="acquisition radius")
    ax2a.set_xlabel("target index")
    ax2a.set_ylabel("mean hold-period distance to target (cm)")
    ax2a.set_title("Acquisition error per target")
    ax2a.legend(fontsize=8)

    acquired_counts = [summary["naive"]["targets_acquired"], summary["refit"]["targets_acquired"]]
    ax2b.bar(["naive-KF", "ReFIT-KF"], acquired_counts, color=["tab:red", "tab:green"])
    ax2b.set_ylim(0, N_TARGETS)
    ax2b.set_ylabel(f"targets acquired (of {N_TARGETS})")
    ax2b.set_title("Successful target acquisitions")
    fig2.suptitle("Naive-KF vs. ReFIT-KF acquisition performance")
    # === END FIGURE ===

    figures = [fig1, fig2]
    fig_names = ("fig01_bci_reach_trajectories", "fig02_acquisition_metrics")
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "extras" / "em_dynamax"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figures, fig_names):
            path = export_dir / f"{name}.png"
            fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
            figure_paths.append(path)
            print(f"  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    # ---- Appendix: JAX-backed EM/CMGF routines (nstat.extras.em.dynamax_bridge) ----
    print()
    print("-" * 72)
    print("Appendix: KF_EM / PP_EM / mPPCO_EM state-space fitting via the Dynamax bridge")
    print("-" * 72)
    rc_em = _demo_predictive_ll()  # pure NumPy, always runs regardless of dynamax
    dynamax_available = True
    try:
        rc_em |= _demo_linear_gaussian_em()
        rc_em |= _demo_cmgf_inference()
        rc_em |= _demo_point_process_em()
        rc_em |= _demo_hybrid_em()
    except ImportError as exc:
        dynamax_available = False
        print(f"  dynamax not installed ({exc}); skipping the JAX-backed "
              "KF_EM/CMGF/PP_EM/mPPCO_EM demos -- the naive-KF vs. ReFIT-KF "
              "contrast above already ran fully without it.")
    dynamax_appendix_ok = rc_em == 0

    return {
        "gain_true": gain_true,
        "gain_naive": gain_naive,
        "gain_refit": gain_refit,
        "summary": summary,
        "refit_better_than_naive": bool(refit_better),
        "dynamax_available": bool(dynamax_available),
        "dynamax_appendix_ok": bool(dynamax_appendix_ok),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Motor-BCI decoder calibration: naive-KF vs. ReFIT-KF "
                    "closed-loop recalibration, plus the Dynamax EM appendix.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
        help="np.random.default_rng seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the two PNGs to --export-dir.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="Override the PNG export directory.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write a compact recovery/GOF summary as JSON.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing figures (headless).",
    )
    parser.add_argument(
        "--plot-style", choices=("modern", "legacy"), default="legacy",
        help="Figure styling forwarded to nstat.apply_plot_style.",
    )
    args = parser.parse_args(argv)

    if args.no_display:
        import matplotlib
        matplotlib.use("Agg")
        visible = False
    else:
        visible = bool(args.show)

    result = run_demo(
        seed=args.seed,
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    ok = result["refit_better_than_naive"] and result["dynamax_appendix_ok"]
    if ok:
        print("\nAll BCI-calibration and EM/inference routines ran." if result["dynamax_available"]
              else "\nBCI-calibration recovery assertion passed (dynamax appendix skipped).")
    else:
        print("\nA routine reported a problem.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
