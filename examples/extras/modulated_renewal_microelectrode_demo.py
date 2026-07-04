#!/usr/bin/env python3
"""Demo: recovering STN/GPi firing phenotypes along one MER pass.

End-to-end exercise of the rate-modulated renewal (conditional-ISI)
point-process model shipped in :mod:`nstat.extras.spatial`, grounded in a
realistic **deep-brain-stimulation (DBS) microelectrode recording (MER)**
targeting scenario:

**Clinical question.**  During awake DBS surgery for Parkinson's disease,
the neurosurgical team advances a microelectrode along a planned
trajectory through candidate target nuclei (subthalamic nucleus, STN, or
globus pallidus internus, GPi) and listens to the extracellular firing
pattern in real time to confirm the electrode has actually reached the
target -- MRI-based stereotactic coordinates alone are not precise enough
(Sterio et al. 2002).  Three firing **phenotypes** are the classical
targeting cues along that trajectory: **tonic-regular** high-rate cells
(sub-Poisson interspike-interval regularity), **tremor-locked bursting**
cells phase-locked to the patient's peripheral limb tremor at 4-6 Hz
(Levy et al. 2000), and **irregular / high-entropy** cells whose spike
timing carries little structure beyond the mean rate.  Elevated firing
rate and 4-8 Hz oscillatory bursting are hallmarks of the parkinsonian
STN/GPi in the classical primate MPTP model (Bergman et al. 1994), and a
recent human intraoperative series shows that spike-timing *variability
and randomness* -- not firing rate or oscillatory power alone -- are the
most informative single-unit features for classifying the underlying
movement-disorder pathology from pallidal recordings (Sedov et al. 2025).
That is exactly the quantity a conditional-ISI model is built to
estimate: **can a modulated-renewal CIF recover, from spikes alone, the
same rate + regularity signatures neurosurgeons use to read out "which
phenotype, hence which nucleus, is this electrode in"?**

**Scenario.**  Three **synthetic single units**, encountered at three
depths along one simulated MER pass, are each driven by a covariate
-modulated baseline rate lambda_0(t) combined with a renewal (conditional
-ISI) spike-timing term of known regularity:

1. **Tonic-regular** -- high-shape gamma renewal (CV well below 1,
   sub-Poisson) with a slowly modulated baseline rate.
2. **Tremor-locked bursting** -- baseline rate strongly modulated at a
   single tremor-band frequency (4-6 Hz), producing periodic high-rate
   bursts separated by near-silent troughs, with a shape-1 (Poisson
   -like) *local* renewal term -- the burstiness here is a rate-
   modulation effect, not local spike-timing sub/super-regularity.
3. **Irregular / high-entropy** -- a plain shape-1 (exact Poisson)
   renewal with an unmodulated baseline rate: no periodic structure and
   no sub-Poisson regularity, the "no useful phenotype signature" null.

All three are simulated as gamma modulated-renewal processes (Cox 1955;
Barbieri, Quirk, Frank, Wilson & Brown 2001) by the time-rescaling
inverse method -- fully synthetic (no real recording or dataset is used
or claimed).

Demonstrates, independently for each of the three phenotypes:

1. :func:`nstat.extras.spatial.simulate_modulated_renewal` -- simulate a
   gamma renewal process with known shape (hence known CV) and a known
   covariate-modulated baseline rate.
2. :func:`nstat.extras.spatial.fit_modulated_renewal` -- recover the
   covariate GLM (the rate / tremor-drive) *and* the renewal shape / CV
   jointly by alternating penalized MLE.
3. :func:`nstat.extras.spatial.renewal_cdf` -- the continuous
   time-rescaling-theorem check: the fitted model's rescaled
   (operational-time) ISIs pass a KS test against Uniform(0,1), whereas
   the same data tested under a **naive Poisson (CV=1) assumption**
   fails for the tonic-regular phenotype.
4. :mod:`nstat.extras.spatial.marked_gof` -- the discrete-time-rescaling
   correction (Haslinger, Pipa & Brown 2010) tied to each fitted model's
   own per-bin conditional intensity, confirming each fit passes the
   finite-bin-width-corrected KS test too.
5. A raw-ISI **burst index** (coefficient of variation of the
   *un-rescaled* interspike intervals) recovered by resimulating from
   each fit's own recovered rate + shape -- the statistic that separates
   tremor-locked bursting from plain irregularity even when both fit a
   shape-1 (Poisson-like) *renewal* term, since the burstiness there is
   carried by the rate modulation, not by the renewal shape.

Together, the three fits reproduce the three-way phenotype separation:
tonic-regular shows low renewal CV; tremor-locked bursting shows a
Poisson-like renewal CV but an elevated, oscillation-driven burst index;
irregular shows both a Poisson-like renewal CV and a Poisson-like burst
index.

The script is **fully synthetic** -- no figshare dataset access required.

Run::

    python examples/extras/modulated_renewal_microelectrode_demo.py            # interactive
    python examples/extras/modulated_renewal_microelectrode_demo.py --no-display
    python examples/extras/modulated_renewal_microelectrode_demo.py --export-figures

PNGs from ``--export-figures`` are written into a user-chosen directory
(``--export-dir``, defaulting to
``docs/figures/extras/modulated_renewal_microelectrode/``) and are NOT
committed to the repository -- the export flag exists for local
inspection only.  CI never invokes it.

See also :mod:`metrics_spike_distances_demo` -- once a single-unit MER
pass has been characterized phenotype-by-phenotype here, that demo turns
to the population-level DBS biomarker: time-resolved synchrony metrics
tracking pathological beta-burst synchrony across a multi-unit ensemble.

References:

Clinical motivation (DBS/MER firing-phenotype targeting):

- Bergman H, Wichmann T, Karmon B, DeLong MR (1994). *The primate
  subthalamic nucleus. II. Neuronal activity in the MPTP model of
  parkinsonism.* J Neurophysiol 72(2):507-520.
- Levy R, Hutchison WD, Lozano AM, Dostrovsky JO (2000). *High-frequency
  synchronization of neuronal activity in the subthalamic nucleus of
  parkinsonian patients with limb tremor.* J Neurosci 20(20):7766-7775.
- Sterio D, Zonenshayn M, Mogilner AY, Rezai AR, Kiprovski K, Kelly PJ,
  Beric A (2002). *Neurophysiological refinement of subthalamic nucleus
  targeting.* Neurosurgery 50(1):58-67.
- Sedov A, Pavlovsky P, Filyushkina V, Dzhalagoniya I, Semenova U,
  Zakharov N, Gamaleya A, Tomskiy A, Shaikh AG (2025). *Pallidal
  spike-train variability and randomness are the most important
  signatures to classify Parkinson's disease and cervical dystonia.*
  Eur J Neurosci 61(2):e16653.

Statistical methods (renewal / conditional-ISI point-process model;
unchanged from the pre-reground single-unit demo):

- Barbieri R, Quirk MC, Frank LM, Wilson MA, Brown EN (2001).
  *Construction and analysis of non-Poisson stimulus-response models of
  neural spiking activity.* J Neurosci Methods 105:25-37.
- Barbieri R, Matten EC, Alabi AA, Brown EN (2004). *A point-process
  model of human heartbeat intervals: new definitions of heart rate and
  heart rate variability.* Am J Physiol Heart Circ Physiol
  288(1):H424-435.  (Barbieri-Brown renewal/time-rescaling lineage.)
- Iyengar S, Liao Q (1997). *Modeling neural activity using the
  generalized inverse Gaussian distribution.* Biol Cybern 77(4):289-295.
- Kass RE, Ventura V (2001). *A spike-train probability model.* Neural
  Computation 13:1713-1720.
- Cox DR (1955). *Some statistical methods connected with series of
  events.* J R Stat Soc B 17(2):129-164.
- Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002). *The
  time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2):325-346.
- Haslinger R, Pipa G, Brown E (2010). *Discrete time rescaling theorem:
  determining goodness of fit for discrete time statistical models of
  neural spiking.* Neural Computation 22(10):2477.
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). *A point
  process framework for relating neural spiking activity to spiking
  history, neural ensemble, and extrinsic covariate effects.* J
  Neurophysiol 93:1074-1089.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


T_TRIAL = 30.0  # seconds of continuous recording simulated per phenotype
RENEWAL = "gamma"  # gamma(shape=1, scale=1) collapses to Exponential(1) exactly

# Discretization for the model fit.  A fine dt (Hz*dt << 1) keeps the
# O(rate * dt) discrete-time-rescaling bias (Haslinger, Pipa & Brown 2010)
# small, and resolves the fastest (tremor-band, ~5 Hz) rate oscillation
# with ~100 bins/cycle.
DT_FIT = 0.002

# The tremor-locked phenotype's burst peaks reach instantaneous rates far
# above the other two phenotypes' (e.g. ~45-90 Hz at the peak of a
# strongly modulated 5 Hz cycle, vs <30 Hz elsewhere), so DT_FIT alone
# pushes rate*dt uncomfortably close to 1 right at the burst peaks for
# the discrete-time-rescaling *correction* (marked_gof) specifically --
# it is finer than the fit's own dt just for that one diagnostic, using
# the fit's already-recovered (dt-grid-independent) rate_fn() callable.
DT_GOF = DT_FIT / 4.0


class Phenotype(NamedTuple):
    """Ground-truth parameters for one synthetic MER-pass firing unit.

    A ``NamedTuple`` (not ``@dataclass``) deliberately: combined with this
    module's ``from __future__ import annotations``, a ``@dataclass``
    triggers a ``sys.modules`` lookup of the *defining* module at class
    -definition time (to resolve ``ClassVar``/``InitVar`` string
    annotations) that fails when the module is loaded via
    ``importlib.util.spec_from_file_location`` without being registered
    in ``sys.modules`` first -- exactly how
    ``tests/test_extras_examples.py::test_example_script_imports_cleanly``
    loads every example script.
    """

    key: str
    label: str
    color: str
    beta0_true: float  # log-baseline rate (Hz, log units)
    beta1_true: float  # covariate-modulation depth (log-rate units)
    freq_hz: float  # covariate (rate-modulation) frequency
    shape_true: float  # gamma renewal shape (CV = 1/sqrt(shape))
    raster_window_s: float  # window shown in the raster/rate panel


PHENOTYPES: tuple[Phenotype, ...] = (
    Phenotype(
        key="tonic_regular", label="Tonic-regular", color="tab:blue",
        beta0_true=float(np.log(15.0)), beta1_true=0.5, freq_hz=0.5,
        shape_true=9.0, raster_window_s=10.0,
    ),
    Phenotype(
        key="tremor_bursting", label="Tremor-locked bursting", color="tab:red",
        beta0_true=float(np.log(12.0)), beta1_true=2.0, freq_hz=5.0,
        shape_true=1.0, raster_window_s=2.0,
    ),
    Phenotype(
        key="irregular", label="Irregular / high-entropy", color="tab:green",
        beta0_true=float(np.log(12.0)), beta1_true=0.0, freq_hz=0.5,
        shape_true=1.0, raster_window_s=10.0,
    ),
)

# Three-way separation margins (see "Three-way separation" printout below).
TONIC_CV_MAX = 0.6          # tonic-regular renewal CV must stay well below 1
IRREGULAR_CV_BAND = (0.6, 1.5)  # irregular renewal CV must sit near Poisson (1)
BURST_SEPARATION_MARGIN = 0.25  # tremor burst index must exceed irregular's by this


def _rate_fn(beta0: float, beta1: float, freq_hz: float):
    """Vectorized covariate-modulated baseline rate lambda_0(t) (Hz)."""

    def f(t: np.ndarray) -> np.ndarray:
        return np.exp(beta0 + beta1 * np.cos(2.0 * np.pi * freq_hz * t))

    return f


def _raw_isi_cv(spike_times: np.ndarray) -> float:
    """Coefficient of variation of the *un-rescaled* (real-time) ISIs.

    Unlike the renewal-model's own CV (``fit.cv``, computed from the
    operational-time rescaled ISIs), this raw-ISI CV is inflated by any
    rate modulation the unit carries -- a strongly rate-modulated Poisson
    process has raw-ISI CV well above 1 even though its underlying
    renewal term is exactly Exponential(1).  That is precisely what
    makes it a useful **burst index**: it separates tremor-locked
    bursting (rate-modulation-driven) from plain Poisson irregularity
    even when both share a shape~1 renewal term.
    """
    isis = np.diff(np.sort(np.asarray(spike_times, dtype=float)))
    if isis.size < 2:
        return float("nan")
    return float(np.std(isis, ddof=1) / np.mean(isis))


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
    """Run the STN/GPi firing-phenotype MER-targeting demo.

    Simulates and independently fits three synthetic single units
    (tonic-regular / tremor-locked bursting / irregular) with the
    modulated-renewal CIF, recovering rate, renewal regularity (CV),
    a raw-ISI burst index, and rescaled-ISI KS goodness-of-fit for each.

    Returns
    -------
    dict
        ``{"phenotypes": {key: {...per-phenotype results...}},
        "separation": {...three-way separation checks...},
        "figure_paths": [...]}``.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    from nstat import apply_plot_style
    from nstat.extras.spatial import (
        fit_modulated_renewal,
        marked_gof,
        renewal_cdf,
        simulate_modulated_renewal,
    )

    print("=" * 72)
    print("Modulated-renewal firing-phenotype recovery along one MER pass")
    print("=" * 72)

    n_bins = int(np.ceil(T_TRIAL / DT_FIT))
    bin_edges = np.arange(n_bins + 1, dtype=float) * DT_FIT
    bin_centers = bin_edges[:-1] + 0.5 * DT_FIT
    n_bins_gof = int(np.ceil(T_TRIAL / DT_GOF))
    bin_edges_gof = np.arange(n_bins_gof + 1, dtype=float) * DT_GOF
    bin_centers_gof = bin_edges_gof[:-1] + 0.5 * DT_GOF
    ks_alpha_const = 1.358  # two-sided KS critical value, alpha=0.05

    phenotype_results: dict[str, dict] = {}

    for i, ph in enumerate(PHENOTYPES):
        print()
        print(f"--- {ph.label} "
              f"(true shape={ph.shape_true:.1f}, CV={1.0 / np.sqrt(ph.shape_true):.3f}, "
              f"freq={ph.freq_hz:.1f} Hz, modulation depth={ph.beta1_true:.2f}) ---")

        true_rate = _rate_fn(ph.beta0_true, ph.beta1_true, ph.freq_hz)
        rng = np.random.default_rng(seed + i)
        spike_times = simulate_modulated_renewal(
            true_rate, ph.shape_true, T=T_TRIAL, renewal=RENEWAL, rng=rng, dt=1e-3,
        )
        n_spikes = spike_times.size
        print(f"Simulated {n_spikes} spikes over {T_TRIAL:.0f}s "
              f"({n_spikes / T_TRIAL:.2f} Hz mean rate) -- fully synthetic")

        covariates = np.cos(2.0 * np.pi * ph.freq_hz * bin_centers)[:, None]
        fit = fit_modulated_renewal(
            spike_times, covariates, renewal=RENEWAL, dt=DT_FIT,
            max_iter=100, n_inner=12,
        )
        print("Recovery table:")
        print(f"  {'param':>18} | {'true':>8} | {'fitted':>8}")
        print(f"  {'beta0 (intercept)':>18} | {ph.beta0_true:8.3f} | {fit.beta[0]:8.3f}")
        print(f"  {'beta1 (drive)':>18} | {ph.beta1_true:8.3f} | {fit.beta[1]:8.3f}")
        print(f"  {'renewal shape':>18} | {ph.shape_true:8.3f} | {fit.shape_param:8.3f}")
        print(f"  {'CV (renewal)':>18} | {1.0 / np.sqrt(ph.shape_true):8.3f} | {fit.cv:8.3f}")
        print(f"  converged={fit.converged}, n_iter={fit.n_iter}")

        # ---- Continuous time-rescaling KS check (true-family fit vs a
        # naive Poisson/CV=1 assumption on the *same* rescaled ISIs). ----
        u_fit = renewal_cdf(fit.rescaled_isis, fit.shape_param, RENEWAL)
        ks_fit = stats.kstest(u_fit, "uniform")
        u_poisson = renewal_cdf(fit.rescaled_isis, 1.0, RENEWAL)
        ks_poisson = stats.kstest(u_poisson, "uniform")
        ks_band = ks_alpha_const / np.sqrt(len(u_fit))

        print("Continuous time-rescaling KS test (rescaled ISIs vs Uniform(0,1)):")
        print(
            f"  fitted {RENEWAL} model : D={ks_fit.statistic:.4f}  "
            f"(band={ks_band:.4f})  {'PASS' if ks_fit.statistic < ks_band else 'FAIL'}"
        )
        print(
            f"  naive Poisson (CV=1)  : D={ks_poisson.statistic:.4f}  "
            f"(band={ks_band:.4f})  {'PASS' if ks_poisson.statistic < ks_band else 'FAIL'}"
        )

        # ---- Discrete-time-rescaling tie to marked_gof (Haslinger-Pipa-Brown).
        # Uses the finer DT_GOF grid (see module docstring) -- rate_fn() is a
        # continuous-time callable, independent of the dt the model was fit on. ----
        rate_fn_fitted = fit.rate_fn()
        p_k = np.clip(
            rate_fn_fitted(bin_centers_gof) * DT_GOF, np.finfo(float).eps, 1.0 - 1e-12
        )
        spike_bins = np.clip(
            np.searchsorted(bin_edges_gof, spike_times, side="right") - 1,
            0, n_bins_gof - 1,
        )
        gof = marked_gof.marked_time_rescaling(
            spike_bins, None, p_k, rng=np.random.default_rng(seed + 51 + i)
        )
        print("Discrete-time-rescaling correction (marked_gof.marked_time_rescaling):")
        print(
            f"  uncorrected: D={gof.ks_uncorrected:.4f}  "
            f"{'PASS' if gof.inside_uncorrected else 'FAIL'}"
        )
        print(
            f"  corrected  : D={gof.ks_corrected:.4f}  "
            f"{'PASS' if gof.inside_corrected else 'FAIL'}  (band={gof.ks_band:.4f})"
        )

        # ---- Burst index: raw (un-rescaled) ISI CV, true vs recovered
        # -from-fit (resimulate from the fitted rate/shape and recompute). ----
        burst_index_true = _raw_isi_cv(spike_times)
        fitted_lam0_fn = _rate_fn(fit.beta[0], fit.beta[1], ph.freq_hz)
        recon_spikes = simulate_modulated_renewal(
            fitted_lam0_fn, fit.shape_param, T=T_TRIAL, renewal=RENEWAL,
            rng=np.random.default_rng(seed + 101 + i), dt=1e-3,
        )
        burst_index_recovered = _raw_isi_cv(recon_spikes)
        print(
            f"Burst index (raw ISI CV): true-data={burst_index_true:.3f}  "
            f"recovered-from-fit={burst_index_recovered:.3f}"
        )

        phenotype_results[ph.key] = {
            "phenotype": ph,
            "spike_times": spike_times,
            "n_spikes": n_spikes,
            "fit": fit,
            "cv_true": 1.0 / np.sqrt(ph.shape_true),
            "u_fit": u_fit,
            "u_poisson": u_poisson,
            "ks_fit_statistic": float(ks_fit.statistic),
            "ks_fit_pass": bool(ks_fit.statistic < ks_band),
            "ks_poisson_statistic": float(ks_poisson.statistic),
            "ks_poisson_pass": bool(ks_poisson.statistic < ks_band),
            "ks_band": float(ks_band),
            "gof": gof,
            "marked_gof_uncorrected_pass": bool(gof.inside_uncorrected),
            "marked_gof_corrected_pass": bool(gof.inside_corrected),
            "burst_index_true": burst_index_true,
            "burst_index_recovered": burst_index_recovered,
        }

    # ---- Three-way separation summary ----
    tonic = phenotype_results["tonic_regular"]
    tremor = phenotype_results["tremor_bursting"]
    irregular = phenotype_results["irregular"]

    tonic_cv_ok = tonic["fit"].cv < TONIC_CV_MAX
    irregular_cv_ok = IRREGULAR_CV_BAND[0] <= irregular["fit"].cv <= IRREGULAR_CV_BAND[1]
    cv_order_ok = tonic["fit"].cv < irregular["fit"].cv
    burst_separation_ok = (
        tremor["burst_index_recovered"]
        > irregular["burst_index_recovered"] + BURST_SEPARATION_MARGIN
    )

    print()
    print("=" * 72)
    print("Three-way phenotype separation")
    print("=" * 72)
    print(f"  {'phenotype':>24} | {'CV (renewal)':>13} | {'burst index':>12} | KS(fit)")
    for res in (tonic, tremor, irregular):
        ph = res["phenotype"]
        print(
            f"  {ph.label:>24} | {res['fit'].cv:13.3f} | "
            f"{res['burst_index_recovered']:12.3f} | "
            f"{'PASS' if res['ks_fit_pass'] else 'FAIL'}"
        )
    print(
        f"  tonic-regular CV < {TONIC_CV_MAX}: "
        f"{'PASS' if tonic_cv_ok else 'FAIL'} (CV={tonic['fit'].cv:.3f})"
    )
    print(
        f"  irregular CV in {IRREGULAR_CV_BAND}: "
        f"{'PASS' if irregular_cv_ok else 'FAIL'} (CV={irregular['fit'].cv:.3f})"
    )
    print(
        f"  tonic CV < irregular CV: "
        f"{'PASS' if cv_order_ok else 'FAIL'} "
        f"({tonic['fit'].cv:.3f} vs {irregular['fit'].cv:.3f})"
    )
    print(
        f"  tremor burst index > irregular burst index + {BURST_SEPARATION_MARGIN}: "
        f"{'PASS' if burst_separation_ok else 'FAIL'} "
        f"({tremor['burst_index_recovered']:.3f} vs "
        f"{irregular['burst_index_recovered']:.3f})"
    )
    print(
        "  (Note: tremor-locked bursting and irregular both fit a near-Poisson "
        "renewal CV -- the burst index, not the renewal CV, is what separates "
        "a rate-oscillation-driven phenotype from plain irregularity.)"
    )

    # ---- Figures ----
    # === FIGURE: fig01_raster_rate.png ===
    fig1, axes1 = plt.subplots(3, 2, figsize=(11.5, 10.0))
    for i, ph in enumerate(PHENOTYPES):
        res = phenotype_results[ph.key]
        spike_times = res["spike_times"]
        fit = res["fit"]
        show_t = ph.raster_window_s
        show_mask = spike_times <= show_t

        ax_raster = axes1[i, 0]
        ax_raster.eventplot(
            spike_times[show_mask], lineoffsets=0.5, linelengths=0.8, colors="black",
        )
        ax_raster.set_yticks([])
        ax_raster.set_ylabel(ph.label, fontsize=9)
        ax_raster.set_title(
            f"{ph.label}: raster (first {show_t:.1f}s of {T_TRIAL:.0f}s)", fontsize=9,
        )
        ax_raster.set_xlim(0.0, show_t)
        if i == 2:
            ax_raster.set_xlabel("time (s)")

        ax_rate = axes1[i, 1]
        t_plot = np.linspace(0.0, show_t, 1000)
        true_rate = _rate_fn(ph.beta0_true, ph.beta1_true, ph.freq_hz)
        ax_rate.plot(t_plot, true_rate(t_plot), color="black", lw=1.4, ls=":",
                     label="true lambda_0(t)")
        fitted_lam0 = np.exp(
            fit.beta[0] + fit.beta[1] * np.cos(2.0 * np.pi * ph.freq_hz * t_plot)
        )
        ax_rate.plot(t_plot, fitted_lam0, color=ph.color, lw=1.8,
                     label="fitted lambda_0(t)")
        ax_rate.set_ylabel("rate (Hz)")
        ax_rate.legend(loc="upper right", fontsize=7)
        ax_rate.set_title(f"{ph.label}: rate recovery", fontsize=9)
        if i == 2:
            ax_rate.set_xlabel("time (s)")
    # === END FIGURE ===

    # === FIGURE: fig02_isi_vs_renewal.png ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(15.0, 4.8))
    u_grid = np.linspace(1e-4, 3.0, 400)
    h = 1e-4
    for i, ph in enumerate(PHENOTYPES):
        res = phenotype_results[ph.key]
        fit = res["fit"]
        ax = axes2[i]
        pdf_fit = (
            renewal_cdf(u_grid + h, fit.shape_param, RENEWAL)
            - renewal_cdf(np.maximum(u_grid - h, 0.0), fit.shape_param, RENEWAL)
        ) / (2.0 * h)
        pdf_poisson = (
            renewal_cdf(u_grid + h, 1.0, RENEWAL)
            - renewal_cdf(np.maximum(u_grid - h, 0.0), 1.0, RENEWAL)
        ) / (2.0 * h)
        pdf_true = (
            renewal_cdf(u_grid + h, ph.shape_true, RENEWAL)
            - renewal_cdf(np.maximum(u_grid - h, 0.0), ph.shape_true, RENEWAL)
        ) / (2.0 * h)
        ax.hist(
            fit.rescaled_isis, bins=30, density=True, color=ph.color, alpha=0.40,
            label="operational-time ISIs",
        )
        ax.plot(u_grid, pdf_true, color="black", lw=1.6, ls=":",
                label=f"true (shape={ph.shape_true:.1f})")
        ax.plot(u_grid, pdf_fit, color=ph.color, lw=1.8,
                label=f"fitted (shape={fit.shape_param:.1f}, CV={fit.cv:.2f})")
        ax.plot(u_grid, pdf_poisson, color="gray", lw=1.2, ls="--",
                label="Exp(1) (Poisson null)")
        ax.set_xlabel("operational-time ISI u (mean 1)")
        if i == 0:
            ax.set_ylabel("density")
        ax.set_title(f"{ph.label}", fontsize=9)
        ax.legend(loc="upper right", fontsize=6.5)
    fig2.suptitle("Rescaled-ISI histograms: true vs fitted renewal density, per phenotype")
    # === END FIGURE ===

    # === FIGURE: fig03_ks_rescaled.png ===
    fig3 = plt.figure(figsize=(13.5, 9.0))
    gs = fig3.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
    grid01 = np.linspace(0.0, 1.0, 200)
    for i, ph in enumerate(PHENOTYPES):
        res = phenotype_results[ph.key]
        gof = res["gof"]
        ax = fig3.add_subplot(gs[0, i])
        ax.plot(grid01, grid01, color="black", lw=1.0, ls=":", label="Uniform(0,1)")
        ax.plot(
            np.sort(res["u_fit"]),
            np.linspace(0, 1, len(res["u_fit"]), endpoint=False),
            color=ph.color, lw=1.8,
            label=f"fitted (D={res['ks_fit_statistic']:.3f})",
        )
        ax.plot(
            np.sort(res["u_poisson"]),
            np.linspace(0, 1, len(res["u_poisson"]), endpoint=False),
            color="tab:gray", lw=1.2, ls="--",
            label=f"naive Poisson (D={res['ks_poisson_statistic']:.3f})",
        )
        ax.plot(
            np.sort(gof.u_corrected),
            np.linspace(0, 1, len(gof.u_corrected), endpoint=False),
            color="tab:green", lw=1.4, ls="-.",
            label=f"discrete-corrected (D={gof.ks_corrected:.3f})",
        )
        ax.set_xlabel("u")
        if i == 0:
            ax.set_ylabel("empirical CDF")
        ax.set_title(f"{ph.label}\nKS diagnostic", fontsize=9)
        ax.legend(loc="lower right", fontsize=6)

    ax_bar = fig3.add_subplot(gs[1, :])
    labels = [ph.label for ph in PHENOTYPES]
    x = np.arange(len(labels))
    width = 0.35
    cv_hat = [phenotype_results[ph.key]["fit"].cv for ph in PHENOTYPES]
    cv_true = [phenotype_results[ph.key]["cv_true"] for ph in PHENOTYPES]
    burst_hat = [phenotype_results[ph.key]["burst_index_recovered"] for ph in PHENOTYPES]
    burst_true = [phenotype_results[ph.key]["burst_index_true"] for ph in PHENOTYPES]
    colors = [ph.color for ph in PHENOTYPES]

    ax_bar.bar(x - width / 2, cv_hat, width, color=colors, alpha=0.75,
               label="recovered renewal CV")
    ax_bar.bar(x + width / 2, burst_hat, width, color=colors, alpha=0.75,
               hatch="//", edgecolor="black", label="recovered burst index (raw ISI CV)")
    ax_bar.scatter(x - width / 2, cv_true, marker="D", color="black", zorder=5,
                   label="true renewal CV")
    ax_bar.scatter(x + width / 2, burst_true, marker="D", color="black", zorder=5,
                   label="true-data burst index")
    ax_bar.axhline(1.0, color="gray", lw=1.0, ls=":", label="Poisson (CV=1)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("coefficient of variation")
    ax_bar.set_title(
        "Recovered regularity (CV) and burst index: three-way phenotype separation",
        fontsize=9,
    )
    ax_bar.legend(loc="upper left", fontsize=6.5, ncol=2)

    fig3.suptitle(
        "Goodness-of-fit + recovered regularity: rescaled ISIs vs Uniform(0,1)"
    )
    # === END FIGURE ===

    figures = [fig1, fig2, fig3]
    fig_names = ("fig01_raster_rate", "fig02_isi_vs_renewal", "fig03_ks_rescaled")
    for fig in figures:
        fig.tight_layout()
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = (
                REPO_ROOT / "docs" / "figures" / "extras"
                / "modulated_renewal_microelectrode"
            )
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

    return {
        "phenotypes": phenotype_results,
        "separation": {
            "tonic_cv_ok": bool(tonic_cv_ok),
            "irregular_cv_ok": bool(irregular_cv_ok),
            "cv_order_ok": bool(cv_order_ok),
            "burst_separation_ok": bool(burst_separation_ok),
        },
        "figure_paths": [str(p) for p in figure_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="STN/GPi firing-phenotype MER-targeting demo "
                    "(modulated-renewal conditional-ISI recovery)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
        help="np.random.default_rng base seed.",
    )
    parser.add_argument(
        "--export-figures", action="store_true",
        help="Write the three PNGs to --export-dir.",
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
        summary = {
            "separation": result["separation"],
            "figure_paths": result["figure_paths"],
            "phenotypes": {
                key: {
                    "n_spikes": res["n_spikes"],
                    "beta_hat": list(res["fit"].beta),
                    "shape_param_hat": res["fit"].shape_param,
                    "cv_hat": res["fit"].cv,
                    "cv_true": res["cv_true"],
                    "converged": bool(res["fit"].converged),
                    "ks_fit_statistic": res["ks_fit_statistic"],
                    "ks_fit_pass": res["ks_fit_pass"],
                    "ks_poisson_statistic": res["ks_poisson_statistic"],
                    "ks_poisson_pass": res["ks_poisson_pass"],
                    "marked_gof_corrected_pass": res["marked_gof_corrected_pass"],
                    "marked_gof_uncorrected_pass": res["marked_gof_uncorrected_pass"],
                    "burst_index_true": res["burst_index_true"],
                    "burst_index_recovered": res["burst_index_recovered"],
                }
                for key, res in result["phenotypes"].items()
            },
        }
        args.output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
