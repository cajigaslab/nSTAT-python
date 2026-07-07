#!/usr/bin/env python3
"""Demo: continuous-time drive filtering + inhibitory self-history refractoriness.

End-to-end exercise of :func:`nstat.extras.continuous_cif.simulate_cif_continuous`,
the native-Python port of the MATLAB Simulink model
``PointProcessSimulationCont.slx``. Unlike the discrete ``CIF.simulateCIF``
path in core ``nstat`` (whose stimulus/ensemble/history drives are all
sampled directly on the spike-generation bin grid), this simulator realises
the stimulus filter ``S`` (and ensemble filter ``E``) as genuinely
**continuous** LTI systems integrated with :func:`scipy.signal.lsim`, while
the self-history feedback filter ``H`` is discretized to the bin grid via
:func:`scipy.signal.cont2discrete` (zero-order hold) and stepped once per
bin. That distinction matters biophysically: synaptic/dendritic
integration of an external drive is a continuous-time low-pass process,
independent of whatever bin width a spike-generation model happens to use.

**Scenario (fully synthetic).** A single simulated cortical unit's
instantaneous firing probability is driven by a slow (1 Hz) oscillatory
"local field potential" stimulus, filtered through a continuous first-order
low-pass synaptic filter ``S(s) = K / (tau*s + 1)`` before it reaches the
spike generator -- and, independently, the unit carries an inhibitory
continuous self-history filter ``H(s) = -G / (tau_h*s + 1)`` modeling
post-spike refractoriness (afterhyperpolarization) as a continuous decay,
not a fixed-bin dead time.

Demonstrates:

1. **Ground truth vs. recovered continuous filtering.** The lowpass ``S``
   filter's *analytic* steady-state sinusoidal response (closed-form from
   linear-systems theory: gain attenuation ``K/sqrt(1+(w*tau)^2)`` and
   phase lag ``-atan(w*tau)``) is compared against the *numerically
   realised* continuous drive returned by ``simulate_cif_continuous(...,
   return_details=True).stim_drive`` (which internally calls
   ``scipy.signal.lsim``). They agree once the initial ``lsim`` transient
   (a few time constants) has died out -- confirming the continuous
   realization matches known filter theory, independent of any spike-
   generation bin width.
2. **Effect of continuous inhibitory self-history.** Two runs share the
   exact same injected ``U(0,1)`` variates (so any difference in the
   realised spike train is attributable to ``H`` alone, not RNG luck):
   one with ``H = 0`` (no self-history) and one with the nonzero
   inhibitory ``H``. The nonzero-``H`` run has a measurably more regular
   (lower coefficient-of-variation) interspike-interval distribution --
   the expected refractory signature -- while the mean firing rate stays
   comparable.
3. A composite figure showing (a) the ground-truth-vs-recovered filtered
   stimulus drive, (b) the resulting instantaneous rate lambda(t)/Ts with
   and without history feedback, and (c) the two conditions' spike
   rasters with their empirical ISI CVs annotated.

The script is **fully synthetic** -- no figshare dataset access required,
and it depends only on NumPy/SciPy/Matplotlib (no optional extras
dependency).

Run::

    python examples/extras/continuous_cif_demo.py            # interactive
    python examples/extras/continuous_cif_demo.py --no-display
    python examples/extras/continuous_cif_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/continuous_cif/``) and are NOT committed as a side
effect of running the script -- the export flag exists for local
inspection / doc regeneration only. CI never invokes it.

See also :mod:`nstat.extras.continuous_cif` for the full model semantics
and validation strategy (deterministic-lambda-vs-gold-fixture when
``H = 0``; injected-uniform-history validation when ``H != 0``).
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
# Ground-truth model parameters (fully synthetic).
# ---------------------------------------------------------------------------
Ts = 0.001          # 1 ms spike-generation / random-source sample time (s)
T_TRIAL = 5.0       # seconds of continuous recording simulated
MU = -4.0           # baseline log-odds (binomial link): sigmoid(-4) ~ 1.8% per 1ms bin (~18 Hz)

F_STIM_HZ = 1.0     # slow oscillatory "LFP" stimulus drive frequency
STIM_AMPLITUDE = 1.0

S_GAIN = 1.5        # continuous lowpass stimulus filter S(s) = K / (tau*s + 1)
S_TAU = 0.05        # 50 ms synaptic-integration time constant

H_GAIN = 4.0        # continuous inhibitory self-history filter H(s) = -G / (tau_h*s + 1)
H_TAU = 0.01        # 10 ms refractory decay time constant

BURN_IN_S = 0.5     # skip this much of the S-filter transient in the ground-truth overlay


def _lowpass_steady_state(t: np.ndarray, amplitude: float, w: float, gain: float, tau: float) -> np.ndarray:
    """Closed-form steady-state response of a first-order lowpass K/(tau*s+1)
    to a sinusoidal input ``amplitude * sin(w*t)`` (linear-systems theory)."""
    mag = gain / np.sqrt(1.0 + (w * tau) ** 2)
    phase = -np.arctan(w * tau)
    return mag * amplitude * np.sin(w * t + phase)


def run_demo(
    *,
    seed: int = 20260706,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
    plot_style: str = "legacy",
) -> dict:
    """Run the continuous-time CIF filtering + self-history demo.

    Returns
    -------
    dict
        ``{"ground_truth_rmse": float, "cv_zero_history": float,
        "cv_nonzero_history": float, "mean_rate_zero_history_hz": float,
        "mean_rate_nonzero_history_hz": float, "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style
    from nstat.extras.continuous_cif import simulate_cif_continuous

    print("=" * 72)
    print("Continuous-time CIF: filtered stimulus drive + inhibitory self-history")
    print("=" * 72)

    n = int(np.round(T_TRIAL / Ts)) + 1
    t = np.arange(n) * Ts
    u_stim = STIM_AMPLITUDE * np.sin(2.0 * np.pi * F_STIM_HZ * t)
    u_ens = np.zeros_like(t)

    stim_filter = (np.array([S_GAIN]), np.array([S_TAU, 1.0]))
    zero_hist = 0.0
    nonzero_hist = (np.array([-H_GAIN]), np.array([H_TAU, 1.0]))

    # Same injected uniform draws for both runs, so any difference in the
    # realised spike train is attributable to H alone, not RNG luck.
    rng = np.random.default_rng(seed)
    draws = rng.random(n)

    result_zero = simulate_cif_continuous(
        MU, stim_filter, 0.0, zero_hist, (t, u_stim), (t, u_ens),
        Ts=Ts, simType="binomial", uniform_values=draws, return_details=True,
    )
    result_hist = simulate_cif_continuous(
        MU, stim_filter, 0.0, nonzero_hist, (t, u_stim), (t, u_ens),
        Ts=Ts, simType="binomial", uniform_values=draws, return_details=True,
    )

    # ---- (1) Ground-truth vs. recovered continuous stimulus filtering ----
    w = 2.0 * np.pi * F_STIM_HZ
    analytic_drive = _lowpass_steady_state(t, STIM_AMPLITUDE, w, S_GAIN, S_TAU)
    steady_mask = t >= BURN_IN_S
    ground_truth_rmse = float(
        np.sqrt(np.mean((analytic_drive[steady_mask] - result_zero.stim_drive[steady_mask]) ** 2))
    )
    print(
        f"Continuous lowpass S(s) = {S_GAIN:.1f}/({S_TAU*1e3:.0f}ms*s + 1): "
        f"analytic-vs-lsim steady-state RMSE (t >= {BURN_IN_S:.1f}s) = {ground_truth_rmse:.4f}"
    )

    # ---- (2) Effect of the continuous inhibitory self-history filter ----
    def _isi_cv(spike_times: np.ndarray) -> float:
        isis = np.diff(np.sort(np.asarray(spike_times, dtype=float)))
        if isis.size < 2:
            return float("nan")
        return float(np.std(isis, ddof=1) / np.mean(isis))

    cv_zero = _isi_cv(result_zero.spikes.spikeTimes)
    cv_hist = _isi_cv(result_hist.spikes.spikeTimes)
    rate_zero = result_zero.spike_indicator.mean() / Ts
    rate_hist = result_hist.spike_indicator.mean() / Ts

    print()
    print(f"H = 0 (no self-history)    : {result_zero.spikes.spikeTimes.size} spikes, "
          f"mean rate={rate_zero:.2f} Hz, ISI CV={cv_zero:.3f}")
    print(f"H = -{H_GAIN:.0f}/({H_TAU*1e3:.0f}ms*s+1) (inhibitory): "
          f"{result_hist.spikes.spikeTimes.size} spikes, "
          f"mean rate={rate_hist:.2f} Hz, ISI CV={cv_hist:.3f}")
    print(
        "Inhibitory continuous self-history regularizes spiking "
        f"(lower ISI CV) relative to H=0: "
        f"{'PASS' if cv_hist < cv_zero else 'FAIL'} "
        f"({cv_hist:.3f} < {cv_zero:.3f})"
    )

    # ---- Figure ----
    # === FIGURE: fig01_continuous_stim_filter_and_history.png ===
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5))

    ax0 = axes[0]
    ax0.plot(t, analytic_drive, color="black", lw=1.6, ls=":",
             label="analytic steady-state (linear-systems theory)")
    ax0.plot(t, result_zero.stim_drive, color="tab:blue", lw=1.4, alpha=0.85,
             label="numerically realised (scipy.signal.lsim)")
    ax0.axvspan(0.0, BURN_IN_S, color="gray", alpha=0.15, label=f"transient (< {BURN_IN_S:.1f}s)")
    ax0.set_ylabel("filtered stimulus drive")
    ax0.set_title(
        "Ground truth vs. recovered: continuous lowpass S(s) applied to a "
        f"{F_STIM_HZ:.0f} Hz stimulus (RMSE={ground_truth_rmse:.4f}, t>={BURN_IN_S:.1f}s)",
        fontsize=10,
    )
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.plot(t, result_zero.rate_hz, color="tab:gray", lw=1.2, label="H = 0")
    ax1.plot(t, result_hist.rate_hz, color="tab:red", lw=1.2, alpha=0.85,
              label="H = inhibitory (refractory)")
    ax1.set_ylabel("instantaneous rate (Hz)")
    ax1.set_title("lambda(t)/Ts: with vs. without continuous self-history feedback", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[2]
    show_t = 2.0
    show_zero = result_zero.spikes.spikeTimes
    show_hist = result_hist.spikes.spikeTimes
    ax2.eventplot(show_zero[show_zero <= show_t], lineoffsets=1.0, linelengths=0.8,
                  colors="tab:gray")
    ax2.eventplot(show_hist[show_hist <= show_t], lineoffsets=0.0, linelengths=0.8,
                  colors="tab:red")
    ax2.set_yticks([0.0, 1.0])
    ax2.set_yticklabels([f"H=inhib.\n(CV={cv_hist:.2f})", f"H=0\n(CV={cv_zero:.2f})"], fontsize=8)
    ax2.set_xlim(0.0, show_t)
    ax2.set_xlabel("time (s)")
    ax2.set_title(
        f"Spike rasters (first {show_t:.0f}s): inhibitory continuous self-history "
        "regularizes ISIs", fontsize=10,
    )
    # === END FIGURE ===

    fig.suptitle("nstat.extras.continuous_cif.simulate_cif_continuous demo")
    fig.tight_layout()
    apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "extras" / "continuous_cif"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "fig01_continuous_stim_filter_and_history.png"
        fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
        figure_paths.append(path)
        print(f"  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    return {
        "ground_truth_rmse": ground_truth_rmse,
        "cv_zero_history": cv_zero,
        "cv_nonzero_history": cv_hist,
        "mean_rate_zero_history_hz": float(rate_zero),
        "mean_rate_nonzero_history_hz": float(rate_hist),
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Continuous-time CIF simulator demo "
                    "(continuous stimulus filtering + inhibitory self-history)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260706,
        help="np.random.default_rng base seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the composite PNG to --export-dir.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="Override the PNG export directory.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write a compact summary as JSON.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing the figure (headless).",
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

    assert result["cv_nonzero_history"] < result["cv_zero_history"], (
        "expected the inhibitory continuous self-history filter to regularize "
        "ISIs (lower coefficient of variation) relative to H=0, but got "
        f"cv_hist={result['cv_nonzero_history']:.3f} >= "
        f"cv_zero={result['cv_zero_history']:.3f}"
    )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
