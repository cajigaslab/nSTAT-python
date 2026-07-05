#!/usr/bin/env python3
"""Demo: clinical velocity Kalman decoder -- nstat vs. pykalman cross-check
+ chronic-drift recalibration panel.

**Question (design spec Sec. 4.5).** Does the clinical workhorse **velocity
Kalman decoder** -- the linear-Gaussian Kalman filter that decodes hand/
cursor velocity from population activity (Wu, Gao, Bienenstock, Donoghue &
Black 2006), the same recursion underlying the steady-state clinical
implementation of Malik, Truccolo, Brown & Hochberg (2011) -- replicate
across independent software implementations, and how does its decode
accuracy degrade under the **chronic drift** every intracortical BCI faces
across days: gradual per-unit tuning instability (Perge, Homer, Malik,
Cash, Eskandar, Friehs, Donoghue & Hochberg 2013) and unit turnover
(Downey, Schwed, Chase, Schwartz & Collinger 2018)?

**Scenario -- Part 1: cross-implementation agreement** (unchanged bridge
contract, requires the optional ``pykalman`` dependency). A synthetic
M1-style population with Wu (2006) linear-velocity tuning drives a 2-state
``(v_x, v_y)`` linear-Gaussian fixture. Both nstat's
``DecodingAlgorithms.kalman_filter``/``kalman_smoother`` and
``pykalman.KalmanFilter`` filter/smooth the same simulated population
activity; :func:`~nstat.extras.validation.pykalman_bridge.cross_validate_kalman`
reports the empirical disagreement and
:meth:`~nstat.extras.validation.pykalman_bridge.KalmanComparison.assert_filtered_agree`
is the regression-guard hook. Gracefully skipped (not a failure) when
``pykalman`` is absent -- see the ``[test-parity]`` extra.

**Scenario -- Part 2: chronic-drift recalibration panel** (new, pure
NumPy/SciPy, always runs -- no optional dependency, no ``pykalman``
required). A larger M1 velocity-encoding population (Wu 2006-style
preferred-direction tuning) is calibrated once from a Day-0 recording,
then decoded across ``N_DAYS`` simulated recording days with nstat's own
``DecodingAlgorithms.kalman_filter``. Two nonstationarities documented in
chronic human intracortical recordings accumulate day over day:

- **Per-unit gain/baseline drift** (Perge et al. 2013): each unit's
  velocity-tuning gain undergoes a multiplicative random walk and its
  baseline an additive random walk, modeling the intra-/across-day signal
  instability Perge et al. (2013) documented in a chronic BrainGate
  recording.
- **Unit turnover** (Downey et al. 2018): a fraction of units are "lost"
  and replaced by fresh units with unrelated tuning each day, modeling the
  gradual single-unit yield/identity turnover Downey et al. (2018)
  documented over months of chronic array recording.

A decoder frozen at its Day-0 calibration decodes a steadily worsening
velocity correlation coefficient (the standard Wu/Malik decode-accuracy
metric) as both nonstationarities accumulate; a decoder periodically
recalibrated from a short same-day calibration block holds its accuracy
close to Day-0 levels throughout -- illustrating why periodic
recalibration, not just decoder architecture, is what keeps a chronic
intracortical BCI clinically usable session after session.

Run::

    python examples/extras/validation_pykalman_demo.py
    python examples/extras/validation_pykalman_demo.py --no-display
    python examples/extras/validation_pykalman_demo.py --export-figures

    pip install nstat-toolbox[test-parity]   # also pulls statsmodels, nitime
    python examples/extras/validation_pykalman_demo.py   # also runs the pykalman cross-check

PNGs from ``--export-figures`` land in
``docs/figures/extras/validation_pykalman/``.

Cross-reference: ``examples/extras/em_dynamax_demo.py``'s naive-KF vs.
ReFIT-KF contrast also decodes a Wu (2006)-style M1 velocity-encoding
population with nstat's own Kalman filter; this demo instead cross-checks
the filter's numerics against an independent implementation and studies
its robustness to chronic drift.

References:
- Wu W, Gao Y, Bienenstock E, Donoghue JP, Black MJ (2006). Bayesian
  population decoding of motor cortical activity using a Kalman filter.
  Neural Computation 18(1):80-118.
- Malik WQ, Truccolo W, Brown EN, Hochberg LR (2011). Efficient decoding
  with steady-state Kalman filter in neural interface systems. IEEE
  Transactions on Neural Systems and Rehabilitation Engineering
  19(1):25-34.
- Perge JA, Homer ML, Malik WQ, Cash S, Eskandar E, Friehs G, Donoghue
  JP, Hochberg LR (2013). Intra-day signal instabilities affect decoding
  performance in an intracortical neural interface system. Journal of
  Neural Engineering 10(3):036004.
- Downey JE, Schwed N, Chase SM, Schwartz AB, Collinger JL (2018).
  Intracortical recording stability in human brain-computer interface
  users. Journal of Neural Engineering 15(4):046016.
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
# Scenario constants (fully synthetic -- no real recording).
# ---------------------------------------------------------------------------
DT = 0.02                      # s, 50 Hz bin width (typical intracortical BCI decode rate)
KF_VEL_PERSISTENCE = 0.95      # velocity AR(1) damping shared by generator + decoders
GAIN0 = 0.5                    # nominal per-unit velocity-tuning gain

# Part 1: nstat <-> pykalman agreement fixture (M1 velocity-encoding population).
N_UNITS_XVAL = 14
T_XVAL = 100

# Part 2: chronic-drift recalibration panel.
N_UNITS_DRIFT = 24             # population size
N_DAYS = 15                    # simulated recording days
RECAL_INTERVAL = 5             # recalibrate every this many days (days 0, 5, 10, ...)
T_CALIB = 200                  # bins of same-day calibration data per recalibration
T_TEST = 200                   # bins of held-out test data decoded each day
GAIN_DRIFT_STD = 0.05          # per-day multiplicative log-gain random-walk std (Perge 2013)
BASELINE_DRIFT_STD = 0.03      # per-day additive baseline random-walk std (Perge 2013)
TURNOVER_FRACTION = 0.12       # fraction of units replaced each day (Downey 2018)
OBS_NOISE_STD = 0.3            # observation noise std (drift panel)
DECAY_MARGIN = 0.1             # required accuracy drop for "decays without recalibration"
RECAL_MARGIN = 0.1             # required accuracy gap for "recalibration restores accuracy"


def _velocity_tuned_population(
    rng: np.random.Generator, n_units: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Wu (2006)-style population with linear velocity tuning only.

    Returns ``(preferred_dirs, gains, baselines)``.
    """
    preferred_dirs = rng.uniform(0.0, 2.0 * np.pi, size=n_units)
    gains = GAIN0 * rng.uniform(0.8, 1.2, size=n_units)
    baselines = rng.uniform(0.0, 0.2, size=n_units)
    return preferred_dirs, gains, baselines


def _observation_matrix(preferred_dirs: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """``(n_units, 2)`` velocity-tuning matrix from preferred dirs + gains."""
    return np.stack(
        [gains * np.cos(preferred_dirs), gains * np.sin(preferred_dirs)], axis=1,
    )


def _simulate_velocity(rng: np.random.Generator, n_bins: int, q_std: float = 0.15) -> np.ndarray:
    """AR(1) 2-D velocity process shared by the true generator and decoders."""
    v = np.zeros((n_bins, 2))
    v[0] = rng.normal(scale=1.0, size=2)
    for t in range(1, n_bins):
        v[t] = KF_VEL_PERSISTENCE * v[t - 1] + rng.normal(scale=q_std, size=2)
    return v


def _simulate_population_activity(
    rng: np.random.Generator, v: np.ndarray, C: np.ndarray, baselines: np.ndarray,
    obs_noise_std: float,
) -> np.ndarray:
    """Gaussian-observation population activity: ``y = v @ C.T + baseline + noise``.

    Treating binned population activity as approximately Gaussian is the
    same simplifying convention the classic Wu (2006) / Malik et al.
    (2011) Kalman velocity decoders use.
    """
    n_units = C.shape[0]
    return v @ C.T + baselines + rng.normal(scale=obs_noise_std, size=(v.shape[0], n_units))


def _correlation_coefficient(decoded: np.ndarray, true: np.ndarray) -> float:
    """Mean per-axis Pearson correlation -- the standard Wu/Malik decode-accuracy metric."""
    ccs = [np.corrcoef(decoded[:, i], true[:, i])[0, 1] for i in range(true.shape[1])]
    return float(np.mean(ccs))


# ---------------------------------------------------------------------------
# Part 1: nstat <-> pykalman cross-validation on the velocity fixture.
# ---------------------------------------------------------------------------


def _run_agreement_check(rng: np.random.Generator) -> dict:
    """Cross-validate nstat's Kalman filter against pykalman on a synthetic
    M1 velocity-encoding population.  Gracefully reports unavailability
    (not a failure) when pykalman is not installed.
    """
    from nstat.decoding_algorithms import DecodingAlgorithms

    preferred_dirs, gains, _ = _velocity_tuned_population(rng, N_UNITS_XVAL)
    C = _observation_matrix(preferred_dirs, gains)
    A = np.eye(2) * KF_VEL_PERSISTENCE
    Q = np.eye(2) * 0.01
    R = np.eye(N_UNITS_XVAL) * 0.05
    x0 = np.zeros(2)
    P0 = np.eye(2)

    v_true = np.zeros((T_XVAL, 2))
    y = np.zeros((T_XVAL, N_UNITS_XVAL))
    v_true[0] = rng.multivariate_normal(x0, P0)
    y[0] = C @ v_true[0] + rng.multivariate_normal(np.zeros(N_UNITS_XVAL), R)
    for t in range(1, T_XVAL):
        v_true[t] = A @ v_true[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
        y[t] = C @ v_true[t] + rng.multivariate_normal(np.zeros(N_UNITS_XVAL), R)

    print(
        f"Cross-check fixture : T={T_XVAL}, {N_UNITS_XVAL}-unit M1 velocity-encoding "
        "population, 2-state (v_x, v_y) linear-Gaussian"
    )

    try:
        from nstat.extras.validation.pykalman_bridge import cross_validate_kalman
    except ImportError as exc:
        print(f"Install required: {exc}")
        return {"available": False, "v_true": v_true, "nstat_filtered": None,
                "pykalman_filtered": None, "filtered_agree": False}

    try:
        cmp = cross_validate_kalman(y, A, C, Q, R, x0, P0)
    except ImportError as exc:
        print(f"pykalman missing: {exc}")
        print(
            "  Skipping the nstat<->pykalman agreement check (not a failure) -- "
            "the chronic-drift panel below is pure NumPy and runs regardless. "
            "Install with: pip install nstat-toolbox[test-parity]"
        )
        # nstat's own filter still runs so the agreement panel has something
        # to plot even without the pykalman overlay.
        nstat_out = DecodingAlgorithms.kalman_filter(
            observations=y, transition=A, observation_matrix=C,
            q_cov=Q, r_cov=R, x0=x0, p0=P0,
        )
        return {"available": False, "v_true": v_true,
                "nstat_filtered": np.asarray(nstat_out["state"], dtype=float),
                "pykalman_filtered": None, "filtered_agree": False}

    print(f"filtered_inf_norm : {cmp.filtered_inf_norm:.3e}  "
          f"(empirical baseline ~3.7e-3, t=0 init convention)")
    print(f"smoothed_inf_norm : {cmp.smoothed_inf_norm:.3e}  "
          f"(nstat RTS smoother vs. pykalman RTS smoother)")

    cmp.assert_filtered_agree(atol=1e-2)
    print("FILTER PARITY OK   : nstat <-> pykalman filtered means within tolerance.")

    return {
        "available": True,
        "v_true": v_true,
        "nstat_filtered": cmp.nstat_filtered_means,
        "pykalman_filtered": cmp.pykalman_filtered_means,
        "filtered_agree": True,
    }


# ---------------------------------------------------------------------------
# Part 2: chronic-drift recalibration panel (pure NumPy, always runs).
# ---------------------------------------------------------------------------


def _calibrate_decoder(
    rng: np.random.Generator, preferred_dirs: np.ndarray, gains: np.ndarray,
    baselines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a linear observation model from a fresh same-day calibration block.

    Mirrors the OLS calibration step in ``em_dynamax_demo.py``: regress the
    simulated population activity on ``[1, v_x, v_y]`` to recover
    ``(C_hat, b_hat, R_hat)``.
    """
    n_units = preferred_dirs.size
    v = _simulate_velocity(rng, T_CALIB)
    C_true = _observation_matrix(preferred_dirs, gains)
    y = _simulate_population_activity(rng, v, C_true, baselines, OBS_NOISE_STD)

    design = np.c_[np.ones(T_CALIB), v]
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    b_hat = theta[0]
    c_hat = theta[1:].T
    resid = y - design @ theta
    r_hat = np.maximum(np.var(resid, axis=0), 1e-3)
    assert c_hat.shape == (n_units, 2)
    return c_hat, b_hat, r_hat


def _run_chronic_drift(rng: np.random.Generator) -> dict:
    """Decode accuracy across ``N_DAYS`` simulated days, with and without
    periodic recalibration, under Perge-style gain/baseline drift plus
    Downey-style unit turnover.
    """
    from nstat.decoding_algorithms import DecodingAlgorithms

    preferred_dirs, gains, baselines = _velocity_tuned_population(rng, N_UNITS_DRIFT)

    A = np.eye(2) * KF_VEL_PERSISTENCE
    Q = np.eye(2) * 0.01
    x0 = np.zeros(2)
    P0 = np.eye(2)

    decoder_day0 = _calibrate_decoder(rng, preferred_dirs, gains, baselines)
    decoder_no_recal = decoder_day0
    decoder_recal = decoder_day0
    recal_days = sorted(set(range(0, N_DAYS, RECAL_INTERVAL)))

    acc_no_recal = np.zeros(N_DAYS)
    acc_recal = np.zeros(N_DAYS)
    n_turned_over = np.zeros(N_DAYS, dtype=int)

    for day in range(N_DAYS):
        if day > 0:
            # Perge (2013): per-unit gain/baseline random-walk drift.
            gains = gains * np.exp(rng.normal(scale=GAIN_DRIFT_STD, size=N_UNITS_DRIFT))
            baselines = baselines + rng.normal(scale=BASELINE_DRIFT_STD, size=N_UNITS_DRIFT)
            # Downey (2018): unit turnover -- a fraction of units are
            # replaced by fresh units with unrelated tuning.
            n_turn = int(round(TURNOVER_FRACTION * N_UNITS_DRIFT))
            turn_idx = rng.choice(N_UNITS_DRIFT, size=n_turn, replace=False)
            preferred_dirs[turn_idx] = rng.uniform(0.0, 2.0 * np.pi, size=n_turn)
            gains[turn_idx] = GAIN0 * rng.uniform(0.8, 1.2, size=n_turn)
            baselines[turn_idx] = rng.uniform(0.0, 0.2, size=n_turn)
            n_turned_over[day] = n_turn

        v_test = _simulate_velocity(rng, T_TEST)
        y_test = _simulate_population_activity(
            rng, v_test, _observation_matrix(preferred_dirs, gains), baselines, OBS_NOISE_STD,
        )

        c_hat, b_hat, r_hat = decoder_no_recal
        out = DecodingAlgorithms.kalman_filter(
            observations=y_test - b_hat, transition=A, observation_matrix=c_hat,
            q_cov=Q, r_cov=np.diag(r_hat), x0=x0, p0=P0,
        )
        acc_no_recal[day] = _correlation_coefficient(out["state"], v_test)

        if day in recal_days and day > 0:
            decoder_recal = _calibrate_decoder(rng, preferred_dirs, gains, baselines)
        c_hat_r, b_hat_r, r_hat_r = decoder_recal
        out_r = DecodingAlgorithms.kalman_filter(
            observations=y_test - b_hat_r, transition=A, observation_matrix=c_hat_r,
            q_cov=Q, r_cov=np.diag(r_hat_r), x0=x0, p0=P0,
        )
        acc_recal[day] = _correlation_coefficient(out_r["state"], v_test)

    return {
        "acc_no_recal": acc_no_recal,
        "acc_recal": acc_recal,
        "recal_days": recal_days,
        "n_turned_over": n_turned_over,
    }


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
    """Run the nstat<->pykalman agreement check, then the chronic-drift
    recalibration panel, and render the figure.

    Returns
    -------
    dict
        ``{"agreement_available", "filtered_agree", "drift", "decays_without_recal",
        "recalibration_restores_accuracy", "figure_paths"}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style

    print("=" * 72)
    print("Clinical velocity Kalman decoder: nstat<->pykalman cross-check "
          "+ chronic-drift recalibration")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    agreement = _run_agreement_check(rng)

    print()
    print(f"Chronic-drift recalibration panel: {N_UNITS_DRIFT}-unit M1 "
          f"population, {N_DAYS} simulated days, recalibrating every "
          f"{RECAL_INTERVAL} days (Perge et al. 2013 gain/baseline drift + "
          "Downey et al. 2018 unit turnover)")
    drift = _run_chronic_drift(rng)

    print(f"  {'day':>4} | {'turned over':>11} | {'no recal':>9} | {'recal':>9}")
    for day in range(N_DAYS):
        print(f"  {day:4d} | {drift['n_turned_over'][day]:11d} | "
              f"{drift['acc_no_recal'][day]:9.3f} | {drift['acc_recal'][day]:9.3f}")

    no_recal_start = float(drift["acc_no_recal"][0])
    no_recal_late = float(drift["acc_no_recal"][-5:].mean())
    recal_late = float(drift["acc_recal"][-5:].mean())
    decays_without_recal = no_recal_late < no_recal_start - DECAY_MARGIN
    recalibration_restores_accuracy = recal_late > no_recal_late + RECAL_MARGIN

    print()
    print(f"Day-0 accuracy               : {no_recal_start:.3f}")
    print(f"Last-5-day accuracy, no recal : {no_recal_late:.3f}  "
          f"(decayed by >= {DECAY_MARGIN}: {'PASS' if decays_without_recal else 'FAIL'})")
    print(f"Last-5-day accuracy, recal    : {recal_late:.3f}  "
          f"(exceeds no-recal by >= {RECAL_MARGIN}: "
          f"{'PASS' if recalibration_restores_accuracy else 'FAIL'})")

    # ---- Figure ----
    # === FIGURE: fig01_chronic_drift.png ===
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    t_axis = np.arange(T_XVAL) * DT
    for ax, dim, label in ((ax0, 0, "v_x"), (ax1, 1, "v_y")):
        ax.plot(t_axis, agreement["v_true"][:, dim], color="black", lw=1.0, ls=":",
                 label="true" if dim == 0 else None)
        if agreement["nstat_filtered"] is not None:
            ax.plot(t_axis, agreement["nstat_filtered"][:, dim], color="tab:blue", lw=1.4,
                     label="nstat" if dim == 0 else None)
        if agreement["pykalman_filtered"] is not None:
            ax.plot(t_axis, agreement["pykalman_filtered"][:, dim], color="tab:orange",
                     lw=1.0, ls="--", label="pykalman" if dim == 0 else None)
        elif agreement["nstat_filtered"] is None:
            ax.text(0.5, 0.03, "agreement check unavailable", transform=ax.transAxes,
                     ha="center", fontsize=8, color="gray")
        else:
            ax.text(0.5, 0.03, "pykalman not installed", transform=ax.transAxes,
                     ha="center", fontsize=8, color="gray")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(label)
    ax0.set_title("nstat vs. pykalman decoded velocity")
    ax0.legend(fontsize=7, loc="upper right")

    days = np.arange(N_DAYS)
    ax2.plot(days, drift["acc_no_recal"], color="tab:red", marker="o", ms=4,
              label="no recalibration")
    ax2.plot(days, drift["acc_recal"], color="tab:green", marker="o", ms=4,
              label="periodic recalibration")
    for d in drift["recal_days"]:
        if d > 0:
            ax2.axvline(d, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("simulated day")
    ax2.set_ylabel("decode accuracy (velocity CC)")
    ax2.set_title("Chronic drift: recalibration effect")
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Clinical velocity-KF cross-check (left/middle) + chronic-drift "
        "recalibration (right)"
    )
    # === END FIGURE ===

    fig.tight_layout()
    apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "extras" / "validation_pykalman"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "fig01_chronic_drift.png"
        fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
        figure_paths.append(path)
        print(f"  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    return {
        "agreement_available": bool(agreement["available"]),
        "filtered_agree": bool(agreement["filtered_agree"]),
        "no_recal_start": no_recal_start,
        "no_recal_late": no_recal_late,
        "recal_late": recal_late,
        "decays_without_recal": bool(decays_without_recal),
        "recalibration_restores_accuracy": bool(recalibration_restores_accuracy),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clinical velocity Kalman decoder: nstat<->pykalman cross-check "
                    "+ chronic-drift recalibration panel.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
        help="np.random.default_rng seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write fig01_chronic_drift.png to --export-dir.",
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

    agreement_ok = result["filtered_agree"] if result["agreement_available"] else True
    drift_ok = result["decays_without_recal"] and result["recalibration_restores_accuracy"]
    ok = agreement_ok and drift_ok

    print()
    if ok:
        if result["agreement_available"]:
            print("All checks passed: nstat<->pykalman agree; chronic drift decays "
                  "without recalibration and is restored with it.")
        else:
            print("Chronic-drift recalibration checks passed (pykalman agreement "
                  "check skipped -- pykalman not installed).")
    else:
        print("A check reported a problem.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
