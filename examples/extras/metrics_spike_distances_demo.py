#!/usr/bin/env python3
"""Demo: pathological beta-burst synchrony in Parkinson's disease -- OFF
vs. adaptive DBS.

End-to-end exercise of the parameter-free, PySpike-backed spike-train
distance/synchrony metrics shipped in
:mod:`nstat.extras.metrics.spike_distances`, grounded in a realistic
**subthalamic-nucleus (STN) deep-brain-stimulation (DBS) programming**
scenario:

**Clinical question.**  In Parkinson's disease, STN neurons and the local
field potential both show excessive, transient bouts of 15-30 Hz beta-band
oscillatory synchrony -- "beta bursts" -- and burst *duration*, not
average beta power, is the feature that tracks pathological, parkinsonian
synchrony most closely (Kuhn et al. 2009).  Adaptive (closed-loop) DBS is
explicitly built to detect and truncate those bursts, shortening burst
duration relative to the untreated ("OFF") state (Tinkhauser et al.
2017).  This demo asks: **can time-resolved, parameter-free spike-train
synchrony metrics -- ISI-distance, SPIKE-distance, and
SPIKE-synchronization (Kreuz et al. 2011; Kreuz et al. 2012; Satuvuori
et al. 2017) -- track the emergence and collapse of that population-level
beta-burst synchrony directly from spike trains, well enough to separate
a Parkinsonian-OFF regime from an adaptive-DBS regime?**

**Scenario.**  An 8-unit synthetic STN population shares one time-varying
beta-burst "drive": a semi-Markov sequence of burst (ON) and inter-burst
(OFF) intervals, each ON interval carrying its own randomly drawn
15-30 Hz oscillation that phase-modulates every unit's instantaneous
firing rate in common (only a small per-unit phase offset differs across
units).  The 60 s recording toggles between two 30 s regimes built from
the *same* generative model with different burst-duration statistics:

1. **Parkinsonian OFF** (0-30 s) -- long beta bursts (~1.2-2.6 s)
   separated by short gaps: units spend most of the epoch phase-locked to
   the shared drive, producing tight, population-wide coincident firing.
2. **Adaptive DBS** (30-60 s) -- the same drive with bursts truncated to
   ~0.08-0.30 s and longer inter-burst gaps: units spend most of the
   epoch back at independent, drive-free baseline firing.

Both regimes are simulated as independent inhomogeneous-Poisson draws
around the shared, time-varying rate function by direct binned thinning
-- fully synthetic (no real recording or dataset is used or claimed).

Demonstrates, in sliding windows swept across the whole recording:

1. :func:`nstat.extras.metrics.spike_distances.isi_distance` and
   :func:`spike_distance` -- averaged over all unit pairs, track
   dissimilarity in ISI structure / spike timing (lower = more
   synchronous).
2. :func:`spike_synchronization` -- averaged over all unit pairs, the
   fraction of near-coincident spikes (higher = more synchronous).
3. :func:`pairwise_spike_distance_matrix` -- the population-level N x N
   SPIKE-distance matrix, evaluated per window.

The time-resolved metric trace is compared against the *injected*
ground-truth synchrony envelope (the burst-duty fraction within each
window, which the metrics never see), and the two regimes'
burst-duration distributions are compared directly -- the same "how long
are the beta bursts" question adaptive-DBS controllers are built to
answer online (Tinkhauser et al. 2017).

The script is **fully synthetic** -- no figshare dataset access required.
When the optional ``pyspike`` dependency (``pip install
nstat-toolbox[metrics]``) is not installed, the beta-burst simulation and
the ground-truth-only figure still run unconditionally; only the
quantitative metric-vs-envelope comparison is skipped.

Run::

    python examples/extras/metrics_spike_distances_demo.py            # interactive
    python examples/extras/metrics_spike_distances_demo.py --no-display
    python examples/extras/metrics_spike_distances_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/metrics_spike_distances/``). CI never invokes the
export flag -- the committed PNGs are regenerated and committed
separately.

See also :mod:`modulated_renewal_microelectrode_demo` -- that demo
characterizes single-unit MER firing phenotypes (tonic-regular /
tremor-locked bursting / irregular) one depth at a time along one MER
pass; this demo picks up at the population level once the electrode is
parked in the STN, tracking synchrony across a whole ensemble as the
DBS-programming biomarker that motivates closed-loop, adaptive
stimulation.

References:

- Kreuz T et al. (2011). J Neurosci Methods 195:92 (ISI-distance).
- Kreuz T et al. (2012). J Neurophysiol 109:1457 (SPIKE-distance /
  SPIKE-synchronization).
- Satuvuori E et al. (2017). J Neurosci Methods 287:25 (spike-train
  synchrony measures across multiple time scales).
- Tinkhauser G, Pogosyan A, Little S, Beudel M, Herz DM, Tan H, Brown P
  (2017). The modulatory effect of adaptive deep brain stimulation on
  beta bursts in Parkinson's disease. Brain 140:1053-1067.
- Kuhn AA et al. (2009). Exp Neurol 215:380 (pathological STN beta
  synchrony relates to parkinsonian motor signs).
- Levy R, Hutchison WD, Lozano AM, Dostrovsky JO (2000). High-frequency
  synchronization of neuronal activity in the subthalamic nucleus of
  parkinsonian patients with limb tremor. J Neurosci 20(20):7766-7775.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import nspikeTrain


# ---------------------------------------------------------------------------
# Ground-truth scenario constants
# ---------------------------------------------------------------------------

N_UNITS = 8
T_OFF = 30.0     # Parkinsonian-OFF epoch duration (s)
T_ADBS = 30.0    # adaptive-DBS epoch duration (s)
T_TOTAL = T_OFF + T_ADBS

BASE_RATE_HZ = 25.0   # shared baseline STN firing rate
# Beta-burst phase-locking: each cycle contributes a narrow, von-Mises
# -shaped "pulse" of extra rate concentrated near a shared phase (a phase
# -locked oscillatory discharge, not a broad half-cycle rate boost) --
# this is what actually drives near-coincident cross-unit spike timing
# during a burst, the substrate SPIKE-synchronization is built to detect.
KAPPA = 30.0          # von Mises concentration (higher = narrower phase lock)
PEAK_BOOST = 12.6      # pulse-peak rate multiple (mean multiplier over a full ON
                       # cycle is engineered to ~1.0 -- see BURST_FLOOR below)
BURST_FLOOR = 0.08     # residual (non-phase-locked) rate multiplier during ON
                       # bursts -- entrainment mostly *replaces*, not adds to,
                       # independent tonic firing during a beta burst
FREQ_RANGE_HZ = (15.0, 30.0)   # beta band; each burst draws its own frequency
PHASE_JITTER_SIGMA = 0.04      # per-unit phase offset (rad) -- imperfect sync

# Burst (ON) / inter-burst (OFF-gap) duration ranges, per regime.  OFF
# (Parkinsonian) bursts are long with short gaps; adaptive-DBS bursts are
# truncated with longer gaps (Tinkhauser et al. 2017).
ON_RANGE_OFF = (1.2, 2.6)
GAP_RANGE_OFF = (0.3, 0.9)
ON_RANGE_ADBS = (0.08, 0.30)
GAP_RANGE_ADBS = (0.9, 2.0)

DT_SIM = 1e-3   # 1 ms binned-thinning grid for spike simulation

WINDOW_LEN = 2.0    # sliding-window length for time-resolved metrics (s)
WINDOW_STEP = 1.0   # sliding-window step (s)

# Regime-separation margins (see "Regime comparison" printout below).
SYNC_MARGIN = 0.05      # OFF SPIKE-sync must exceed aDBS SPIKE-sync by this
DIST_MARGIN = 0.03      # aDBS ISI-/SPIKE-distance must exceed OFF's by this


class BurstInterval(NamedTuple):
    """One ground-truth beta-burst (ON) interval shared by every unit."""

    t0: float
    t1: float
    freq_hz: float
    phase0: float


def _generate_burst_envelope(
    rng: np.random.Generator,
    t_start: float,
    t_end: float,
    on_range: tuple[float, float],
    gap_range: tuple[float, float],
    freq_range: tuple[float, float],
) -> tuple[list[BurstInterval], list[float]]:
    """Semi-Markov ON/OFF burst-envelope generator over ``[t_start, t_end)``.

    Returns the list of ON (burst) intervals and the *drawn* (pre-
    truncation) duration of each -- the ground-truth "synchrony-burst
    duration" this demo recovers indirectly from the metric trace.
    """
    intervals: list[BurstInterval] = []
    durations: list[float] = []
    t = t_start + rng.uniform(0.0, gap_range[1])
    while t < t_end:
        dur = float(rng.uniform(*on_range))
        t1 = min(t + dur, t_end)
        freq = float(rng.uniform(*freq_range))
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        intervals.append(BurstInterval(t, t1, freq, phase0))
        durations.append(dur)
        t = t1 + float(rng.uniform(*gap_range))
    return intervals, durations


def _rate_multiplier(
    t: np.ndarray, intervals: list[BurstInterval], phase_shift: float = 0.0,
) -> np.ndarray:
    """Vectorized shared beta-burst rate multiplier (1.0 outside bursts).

    Inside a burst, firing is (mostly) *entrained*: replaced by a narrow
    von-Mises-shaped pulse of rate concentrated at each beta-cycle phase
    (a phase-locked oscillatory discharge) plus a small residual
    (``BURST_FLOOR``) of non-phase-locked background firing -- rather
    than a broad half-cycle rate boost layered on an unchanged tonic
    baseline.  It is the tight, near-coincident timing of that pulse
    across units (up to ``phase_shift``) that actually drives the
    population-level synchrony this demo tracks; the mean multiplier
    over a full ON cycle is engineered to stay near 1.0 so overall
    firing rate is comparable inside and outside a burst -- only the
    *timing structure* differs.
    """
    mult = np.ones_like(t)
    for iv in intervals:
        mask = (t >= iv.t0) & (t < iv.t1)
        if np.any(mask):
            phase = 2.0 * np.pi * iv.freq_hz * (t[mask] - iv.t0) + iv.phase0 + phase_shift
            mult[mask] = BURST_FLOOR + PEAK_BOOST * np.exp(KAPPA * (np.cos(phase) - 1.0))
    return mult


def _duty_fraction(intervals: list[BurstInterval], t0: float, t1: float) -> float:
    """Fraction of ``[t0, t1)`` covered by a burst -- the ground-truth
    synchrony envelope sampled at the metric-trace's own window."""
    total = 0.0
    for iv in intervals:
        total += max(0.0, min(iv.t1, t1) - max(iv.t0, t0))
    return total / (t1 - t0)


def _slice_train(train: nspikeTrain, t0: float, t1: float) -> nspikeTrain:
    """Restrict a spike train to ``[t0, t1)`` as its own recording window."""
    times = np.asarray(train.spikeTimes, dtype=float)
    mask = (times >= t0) & (times < t1)
    return nspikeTrain(
        spikeTimes=times[mask], name=train.name, sampleRate=train.sampleRate,
        minTime=t0, maxTime=t1, makePlots=-1,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_demo(
    *,
    seed: int = 20260704,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool = True,
    plot_style: str = "legacy",
) -> dict:
    """Run the STN beta-burst-synchrony (OFF vs. adaptive-DBS) demo.

    Simulates an 8-unit STN population under a shared, time-varying
    beta-burst drive and tracks the injected synchrony envelope with
    sliding-window ISI-distance / SPIKE-distance / SPIKE-synchronization
    (when PySpike is installed) -- always simulates and always emits the
    figure, regardless of whether the optional ``pyspike`` dependency is
    present.

    Returns
    -------
    dict
        ``{"have_pyspike": bool, "windows": {...}, "regime_checks": {...},
        "burst_durations": {...}, "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt

    from nstat import apply_plot_style

    print("=" * 72)
    print("Beta-burst synchrony: Parkinsonian OFF vs. adaptive-DBS")
    print("=" * 72)

    # ---- Ground-truth beta-burst envelope (shared across all units) ----
    env_rng = np.random.default_rng(seed)
    intervals_off, durations_off = _generate_burst_envelope(
        env_rng, 0.0, T_OFF, ON_RANGE_OFF, GAP_RANGE_OFF, FREQ_RANGE_HZ,
    )
    intervals_adbs, durations_adbs = _generate_burst_envelope(
        env_rng, T_OFF, T_TOTAL, ON_RANGE_ADBS, GAP_RANGE_ADBS, FREQ_RANGE_HZ,
    )
    intervals_all = intervals_off + intervals_adbs

    duty_off = _duty_fraction(intervals_all, 0.0, T_OFF)
    duty_adbs = _duty_fraction(intervals_all, T_OFF, T_TOTAL)
    print(
        f"Injected ground truth: {len(intervals_off)} OFF bursts "
        f"(mean duration {np.mean(durations_off):.2f}s, duty={duty_off:.2f}), "
        f"{len(intervals_adbs)} adaptive-DBS bursts "
        f"(mean duration {np.mean(durations_adbs):.2f}s, duty={duty_adbs:.2f})"
    )

    # ---- Simulate the 8-unit population (pure NumPy; always runs) ----
    t_grid = np.arange(0.0, T_TOTAL, DT_SIM)
    units: list[nspikeTrain] = []
    for i in range(N_UNITS):
        phi_i = float(np.random.default_rng(seed + 300 + i).normal(0.0, PHASE_JITTER_SIGMA))
        rate_i = BASE_RATE_HZ * _rate_multiplier(t_grid, intervals_all, phase_shift=phi_i)
        spike_rng = np.random.default_rng(seed + 400 + i)
        spikes_per_bin = spike_rng.poisson(rate_i * DT_SIM)
        spike_times = t_grid[spikes_per_bin > 0]
        units.append(
            nspikeTrain(
                spikeTimes=spike_times, name=f"STN_unit_{i + 1}", sampleRate=1000.0,
                minTime=0.0, maxTime=T_TOTAL, makePlots=-1,
            )
        )
    print(
        f"Population    : {N_UNITS} STN units over [0, {T_TOTAL:.0f}] s, "
        f"{[len(u.spikeTimes) for u in units]} spikes each"
    )

    # ---- Time-resolved population synchrony metrics (PySpike, optional) ----
    try:
        import pyspike as _pyspike_probe  # noqa: F401
        have_pyspike = True
    except ImportError:
        have_pyspike = False

    centers: list[float] = []
    envelope_duty: list[float] = []
    isi_trace: list[float] = []
    spike_trace: list[float] = []
    sync_trace: list[float] = []
    regime_checks: dict = {}

    if have_pyspike:
        from nstat.extras.metrics.spike_distances import (
            isi_distance,
            spike_distance,
            spike_synchronization,
            pairwise_spike_distance_matrix,
        )

        pairs = list(itertools.combinations(range(N_UNITS), 2))
        t0 = 0.0
        while t0 + WINDOW_LEN <= T_TOTAL + 1e-9:
            t1 = t0 + WINDOW_LEN
            sub = [_slice_train(u, t0, t1) for u in units]
            isi_vals = [isi_distance(sub[a], sub[b]) for a, b in pairs]
            sync_vals = [spike_synchronization(sub[a], sub[b]) for a, b in pairs]
            D = pairwise_spike_distance_matrix(sub)
            spike_vals = D[np.triu_indices(N_UNITS, k=1)]

            centers.append(0.5 * (t0 + t1))
            envelope_duty.append(_duty_fraction(intervals_all, t0, t1))
            isi_trace.append(float(np.nanmean(isi_vals)))
            sync_trace.append(float(np.nanmean(sync_vals)))
            spike_trace.append(float(np.nanmean(spike_vals)))
            t0 += WINDOW_STEP

        centers_arr = np.asarray(centers)
        isi_arr = np.asarray(isi_trace)
        spike_arr = np.asarray(spike_trace)
        sync_arr = np.asarray(sync_trace)
        duty_arr = np.asarray(envelope_duty)

        # Exclude windows straddling the OFF/aDBS boundary (their content is
        # a mix of both regimes) so the regime-level comparison isn't
        # diluted by boundary contamination.
        off_mask = centers_arr < (T_OFF - 0.5 * WINDOW_LEN)
        adbs_mask = centers_arr >= (T_OFF + 0.5 * WINDOW_LEN)

        sync_off_mean = float(sync_arr[off_mask].mean())
        sync_adbs_mean = float(sync_arr[adbs_mask].mean())
        isi_off_mean = float(isi_arr[off_mask].mean())
        isi_adbs_mean = float(isi_arr[adbs_mask].mean())
        spike_off_mean = float(spike_arr[off_mask].mean())
        spike_adbs_mean = float(spike_arr[adbs_mask].mean())

        sync_ok = (sync_off_mean - sync_adbs_mean) > SYNC_MARGIN
        isi_ok = (isi_adbs_mean - isi_off_mean) > DIST_MARGIN
        spike_ok = (spike_adbs_mean - spike_off_mean) > DIST_MARGIN

        corr_sync = float(np.corrcoef(duty_arr, sync_arr)[0, 1])
        corr_isi = float(np.corrcoef(duty_arr, isi_arr)[0, 1])
        corr_spike = float(np.corrcoef(duty_arr, spike_arr)[0, 1])

        print()
        print(f"Time-resolved metrics: {len(centers)} sliding windows "
              f"(length={WINDOW_LEN:.1f}s, step={WINDOW_STEP:.1f}s), "
              f"{len(pairs)} unit pairs")
        print("Regime comparison (window-averaged, population pairwise mean):")
        print(f"  {'metric':>24} | {'OFF':>8} | {'adaptive-DBS':>13} | result")
        print(
            f"  {'SPIKE-synchronization':>24} | {sync_off_mean:8.3f} | "
            f"{sync_adbs_mean:13.3f} | "
            f"{'PASS (OFF > aDBS)' if sync_ok else 'FAIL'}"
        )
        print(
            f"  {'ISI-distance':>24} | {isi_off_mean:8.3f} | {isi_adbs_mean:13.3f} | "
            f"{'PASS (aDBS > OFF)' if isi_ok else 'FAIL'}"
        )
        print(
            f"  {'SPIKE-distance':>24} | {spike_off_mean:8.3f} | {spike_adbs_mean:13.3f} | "
            f"{'PASS (aDBS > OFF)' if spike_ok else 'FAIL'}"
        )
        print(
            "Correlation of windowed metric vs. ground-truth burst-duty envelope: "
            f"SPIKE-sync r={corr_sync:+.3f} (expect positive), "
            f"ISI-distance r={corr_isi:+.3f}, SPIKE-distance r={corr_spike:+.3f} "
            "(expect negative)"
        )

        regime_checks = {
            "sync_off_mean": sync_off_mean,
            "sync_adbs_mean": sync_adbs_mean,
            "isi_off_mean": isi_off_mean,
            "isi_adbs_mean": isi_adbs_mean,
            "spike_off_mean": spike_off_mean,
            "spike_adbs_mean": spike_adbs_mean,
            "sync_ok": bool(sync_ok),
            "isi_ok": bool(isi_ok),
            "spike_ok": bool(spike_ok),
            "corr_sync_vs_envelope": corr_sync,
            "corr_isi_vs_envelope": corr_isi,
            "corr_spike_vs_envelope": corr_spike,
        }
    else:
        print()
        print(
            "PySpike not installed -- skipping quantitative spike-distance "
            "metrics (pip install nstat-toolbox[metrics]). The beta-burst "
            "simulation and ground-truth-envelope figure still run."
        )

    # ---- Burst-duration summary (pure NumPy; always computed) ----
    duration_off_mean = float(np.mean(durations_off))
    duration_adbs_mean = float(np.mean(durations_adbs))
    duration_ok = duration_off_mean > duration_adbs_mean
    print()
    print("Synchrony-burst duration (ground truth):")
    print(
        f"  Parkinsonian OFF   : mean={duration_off_mean:.3f}s over "
        f"{len(durations_off)} bursts"
    )
    print(
        f"  adaptive DBS       : mean={duration_adbs_mean:.3f}s over "
        f"{len(durations_adbs)} bursts"
    )
    print(
        f"  OFF burst duration > adaptive-DBS burst duration: "
        f"{'PASS' if duration_ok else 'FAIL'}"
    )

    # ---- Figure ----
    # === FIGURE: fig01_beta_burst_synchrony.png ===
    fig = plt.figure(figsize=(13.0, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.0, 0.9], hspace=0.55, wspace=0.15)

    # Zoomed representative-window rasters -- at a 25+ Hz population rate,
    # a full 60 s raster compresses to an unreadable solid black bar, so
    # each regime gets its own zoom around one typical burst instead.
    def _representative_window(
        intervals: list[BurstInterval], target_dur: float,
        pad_before: float, pad_after: float, lo: float, hi: float,
    ) -> tuple[float, float]:
        idx = int(np.argmin([abs((iv.t1 - iv.t0) - target_dur) for iv in intervals]))
        iv = intervals[idx]
        return max(lo, iv.t0 - pad_before), min(hi, iv.t1 + pad_after)

    win_off = _representative_window(
        intervals_off, duration_off_mean, 0.8, 1.2, 0.0, T_OFF,
    )
    win_adbs = _representative_window(
        intervals_adbs, duration_adbs_mean, 1.0, 1.5, T_OFF, T_TOTAL,
    )

    for col, (win, label, bg) in enumerate((
        (win_off, "Parkinsonian OFF (representative burst)", "tab:blue"),
        (win_adbs, "adaptive DBS (representative burst)", "tab:green"),
    )):
        ax = fig.add_subplot(gs[0, col])
        ax.axvspan(win[0], win[1], color=bg, alpha=0.06, zorder=0)
        for iv in intervals_all:
            if iv.t1 >= win[0] and iv.t0 <= win[1]:
                ax.axvspan(
                    max(iv.t0, win[0]), min(iv.t1, win[1]),
                    color="tab:red", alpha=0.22, zorder=1,
                )
        for i, u in enumerate(units):
            spikes = np.asarray(u.spikeTimes)
            in_win = spikes[(spikes >= win[0]) & (spikes <= win[1])]
            ax.eventplot(in_win, lineoffsets=i + 1, linelengths=0.8, colors="black", zorder=2)
        ax.set_yticks(range(1, N_UNITS + 1))
        if col == 0:
            ax.set_yticklabels([f"unit {i + 1}" for i in range(N_UNITS)], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(win[0], win[1])
        ax.set_xlabel("time (s)")
        ax.set_title(label, fontsize=9)

    ax_metric = fig.add_subplot(gs[1, :])
    ax_metric.axvspan(0.0, T_OFF, color="tab:blue", alpha=0.06, zorder=0)
    ax_metric.axvspan(T_OFF, T_TOTAL, color="tab:green", alpha=0.06, zorder=0)
    if centers:
        ax_metric.fill_between(
            centers, 0.0, envelope_duty, color="tab:red", alpha=0.22, zorder=1,
            label="true synchrony envelope (burst duty fraction)",
        )
    if have_pyspike and centers:
        ax_metric.plot(
            centers, sync_trace, color="tab:purple", lw=1.8, zorder=2,
            label="SPIKE-synchronization (higher = more synchronous)",
        )
        ax_metric.plot(
            centers, isi_trace, color="tab:orange", lw=1.4, ls="--", zorder=2,
            label="ISI-distance (lower = more synchronous)",
        )
        ax_metric.plot(
            centers, spike_trace, color="tab:brown", lw=1.4, ls=":", zorder=2,
            label="SPIKE-distance (lower = more synchronous)",
        )
    else:
        ax_metric.text(
            0.5, 0.55, "PySpike not installed -- metric traces skipped",
            transform=ax_metric.transAxes, ha="center", va="center", fontsize=10, color="gray",
        )
        ax_metric.text(
            0.5, 0.40, "pip install nstat-toolbox[metrics]",
            transform=ax_metric.transAxes, ha="center", va="center", fontsize=9, color="gray",
        )
    ax_metric.axvline(T_OFF, color="black", lw=1.2, ls="--")
    ax_metric.set_xlim(0.0, T_TOTAL)
    ax_metric.set_ylim(0.0, 1.05)
    ax_metric.set_xlabel("time (s)")
    ax_metric.set_ylabel("metric value / duty fraction")
    if centers:
        ax_metric.legend(loc="upper right", fontsize=7)
    ax_metric.set_title(
        "Time-resolved synchrony metrics (sliding window) vs. ground-truth burst envelope",
        fontsize=10,
    )

    ax_hist = fig.add_subplot(gs[2, :])
    all_durations = list(durations_off) + list(durations_adbs)
    bins = np.linspace(0.0, max(all_durations) * 1.05, 25)
    ax_hist.hist(
        durations_off, bins=bins, alpha=0.6, color="tab:blue",
        label=f"Parkinsonian OFF (mean={duration_off_mean:.2f}s, n={len(durations_off)})",
    )
    ax_hist.hist(
        durations_adbs, bins=bins, alpha=0.6, color="tab:green",
        label=f"adaptive DBS (mean={duration_adbs_mean:.2f}s, n={len(durations_adbs)})",
    )
    ax_hist.axvline(duration_off_mean, color="tab:blue", lw=1.5, ls="--")
    ax_hist.axvline(duration_adbs_mean, color="tab:green", lw=1.5, ls="--")
    ax_hist.set_xlabel("beta-burst duration (s)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Synchrony-burst duration distributions per regime", fontsize=10)
    ax_hist.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Beta-burst synchrony: Parkinsonian OFF vs. adaptive-DBS-truncated bursts\n"
        "Top: zoomed population raster (red shading = true beta-burst envelope)"
    )
    # === END FIGURE ===

    fig.tight_layout()
    apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "extras" / "metrics_spike_distances"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / "fig01_beta_burst_synchrony.png"
        fig.savefig(path, dpi=180, facecolor="w", edgecolor="none")
        figure_paths.append(path)
        print(f"\n  Saved: {path}")

    if visible:
        plt.show()
    else:
        plt.close("all")

    return {
        "have_pyspike": have_pyspike,
        "windows": {
            "centers": centers,
            "envelope_duty": envelope_duty,
            "isi_trace": isi_trace,
            "spike_trace": spike_trace,
            "sync_trace": sync_trace,
        },
        "regime_checks": regime_checks,
        "burst_durations": {
            "off_mean": duration_off_mean,
            "adbs_mean": duration_adbs_mean,
            "off_n": len(durations_off),
            "adbs_n": len(durations_adbs),
            "duration_ok": bool(duration_ok),
        },
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STN beta-burst-synchrony demo (Parkinsonian OFF vs. adaptive-DBS, "
                    "PySpike-backed time-resolved synchrony metrics)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260704,
        help="np.random.default_rng base seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write fig01_beta_burst_synchrony.png to --export-dir.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="Override the PNG export directory.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write a compact recovery/regime summary as JSON.",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
