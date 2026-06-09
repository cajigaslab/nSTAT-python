#!/usr/bin/env python3
"""Teaching capstone — a rhythmic cell and a beta biomarker, the clinical way.

This is a *tutorial* script: read it top-to-bottom as a lesson. It uses only
simulated data (no download), so it runs anywhere, and it casts the standard
nSTAT workflow in the setting of a microelectrode advanced into a deep brain
nucleus during deep brain stimulation (DBS) surgery — the recording made to
localize a target such as the subthalamic nucleus (STN) and to read out the
oscillatory activity that guides therapy.

Nothing here is a new method. It is the same encode -> check -> spectrum ->
decode arc as ``encoding_to_goodness_of_fit.py`` and the place-cell capstone,
pointed at a clinically resonant cell: a *tremor cell* whose firing is
phase-locked to a few-hertz limb tremor.

The story, in five acts
-----------------------
1. ENCODING. Simulate a tremor cell: a point process whose rate oscillates at
   the tremor frequency (Levy et al. 2000; Hutchison et al. 1998). The rhythm
   enters the conditional intensity as an ordinary covariate (Truccolo 2005).
2. FITTING. Fit a rhythm-aware point-process GLM and recover the drive.
3. GOODNESS-OF-FIT. Contrast the rhythm-aware model against a constant-rate
   model with the time-rescaling KS test (Brown et al. 2002): same mean rate,
   opposite verdicts — only the rhythm-aware model captures the timing.
4. THE BETA BIOMARKER. Low-pass the same electrode to a field potential and
   estimate beta-band (13-30 Hz) power with the multitaper spectrum
   (Mitra & Pesaran 1999). Elevated/bursting beta is the feedback signal for
   adaptive DBS (Little et al. 2013; Tinkhauser et al. 2017; Zaidel et al. 2010).
5. DECODING. Read the tremor rhythm back out of a small ensemble with the
   point-process adaptive filter (Eden et al. 2004), with a calibrated band.

Concepts pages
--------------
- Rhythmic firing & the clinical microelectrode:
      docs/concepts/rhythmic_firing_and_clinical_microelectrode.md
- Spikes & point-process GLMs:  docs/concepts/spike_trains_and_glms.md
- LFP & spectral analysis:      docs/concepts/lfp_and_spectral.md
- Goodness-of-fit & decoding:   docs/concepts/goodness_of_fit_and_decoding.md

References (PMIDs verified against PubMed; see docs/concepts/bibliography.md)
- Truccolo et al. 2005, J Neurophysiol 93:1074         PMID 15356183
- Brown et al. 2002, Neural Comput 14:325              PMID 11802915
- Eden et al. 2004, Neural Comput 16:971               PMID 15070506
- Mitra & Pesaran 1999, Biophys J 76:691               PMID 9929474
- Hutchison et al. 1998, Ann Neurol 44:622             PMID 9778260
- Levy et al. 2000, J Neurosci 20:7766                 PMID 11027240
- Little et al. 2013, Ann Neurol 74:449                PMID 23852650
- Tinkhauser et al. 2017, Brain 140:1053               PMID 28334851
- Zaidel et al. 2010, Brain 133:2007                   PMID 20534648

Run
---
    python examples/tutorials/clinical_microelectrode_walkthrough.py
    python examples/tutorials/clinical_microelectrode_walkthrough.py --save-fig out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make nstat importable when run from a fresh checkout.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import (  # noqa: E402
    DecodingAlgorithms,
    SignalObj,
    fit_poisson_glm,
    population_time_rescale,
    simulate_cif_from_stimulus,
)

TREMOR_HZ = 5.0   # limb tremor; STN/thalamic tremor cells lock to ~3-8 Hz
BETA_HZ = 22.0    # within the 13-30 Hz beta band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-fig", metavar="PATH", default=None,
        help="If given, save a summary figure to PATH (needs matplotlib).",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(11)

    # =====================================================================
    # ACT 1 — ENCODING: simulate a tremor cell (a rhythmic point process)
    # =====================================================================
    # A single unit recorded for 90 s at 1 kHz inside the nucleus. Its rate
    # oscillates with the limb tremor:  log lambda = beta0 + beta1 * drive(t),
    # where drive(t) = sin(2*pi*f*t) is the tremor rhythm acting as a covariate.
    dt = 0.001
    t = np.arange(0.0, 90.0, dt)
    drive = np.sin(2.0 * np.pi * TREMOR_HZ * t)

    true_beta0 = 2.6    # baseline ~ exp(2.6) ~ 13 spikes/s (brisk STN-like cell)
    true_beta1 = 1.3    # strength of tremor locking

    spikes, _, _ = simulate_cif_from_stimulus(
        time=t, stimulus=drive, beta0=true_beta0, beta1=true_beta1, rng=rng
    )

    bin_width = 0.001
    edges = np.arange(0.0, 90.0 + bin_width, bin_width)
    y = spikes.to_binned_counts(edges)[: t.size]
    offset = np.full(y.shape, np.log(bin_width))
    x = drive[:, None]

    n_spikes = int(y.sum())
    print("ACT 1 — Encoding a tremor cell")
    print(f"  Simulated {n_spikes} spikes over {t[-1]:.0f} s "
          f"({n_spikes / t[-1]:.1f} spikes/s), locked to a {TREMOR_HZ:.0f} Hz tremor")
    print(f"  True parameters:  beta0={true_beta0:+.3f}  beta1={true_beta1:+.3f}\n")

    # =====================================================================
    # ACT 2 — FITTING: a rhythm-aware point-process GLM
    # =====================================================================
    fit = fit_poisson_glm(x, y, offset=offset, l2=1e-4, max_iter=100, tol=1e-10)
    est_beta0, est_beta1 = fit.intercept, fit.coefficients[0]
    print("ACT 2 — Fitting (rhythm-aware GLM)")
    print(f"  Estimated:        beta0={est_beta0:+.3f}  beta1={est_beta1:+.3f}")
    print(f"  Recovery error:   |dbeta0|={abs(est_beta0 - true_beta0):.3f}  "
          f"|dbeta1|={abs(est_beta1 - true_beta1):.3f}\n")

    # =====================================================================
    # ACT 3 — GOODNESS-OF-FIT: rhythm-aware vs. constant-rate (KS test)
    # =====================================================================
    lam_rhythm = np.exp(est_beta0 + est_beta1 * x[:, 0] + offset)
    lam_constant = np.full_like(y, y.mean())   # same mean rate, no rhythm

    gof_rhythm = population_time_rescale([y], [lam_rhythm])
    gof_constant = population_time_rescale([y], [lam_constant])

    print("ACT 3 — Goodness-of-fit (time-rescaling KS)")
    print(f"  RHYTHM-AWARE model: KS p = {gof_rhythm.ground_ks_pvalue:.3g}  -> "
          f"{_verdict(gof_rhythm.ground_ks_pvalue)}")
    print(f"  CONSTANT-RATE model: KS p = {gof_constant.ground_ks_pvalue:.3g}  -> "
          f"{_verdict(gof_constant.ground_ks_pvalue)}")
    print("  Lesson: both match the MEAN rate; only the rhythm-aware model")
    print("  reproduces the TIMING, so only it passes goodness-of-fit.\n")

    # =====================================================================
    # ACT 4 — THE BETA BIOMARKER: multitaper spectrum of the field potential
    # =====================================================================
    # Low-pass the same electrode -> a field potential. In Parkinson's disease
    # the STN field potential carries elevated beta (13-30 Hz) power; its level
    # and burst structure are the feedback signal for adaptive DBS.
    fs = 1000.0
    t_lfp = np.arange(0.0, 30.0, 1 / fs)
    x_lfp = (0.6 * np.sin(2 * np.pi * TREMOR_HZ * t_lfp)   # tremor-band component
             + 1.0 * np.sin(2 * np.pi * BETA_HZ * t_lfp)   # beta biomarker
             + 0.5 * rng.standard_normal(t_lfp.size))
    lfp = SignalObj(t_lfp, x_lfp, name="STN field potential")
    freqs, power, _ = lfp.MTMspectrum(NW=4.0)
    beta = (freqs >= 13) & (freqs <= 30)
    peak_hz = float(freqs[beta][np.argmax(power[beta])])
    print("ACT 4 — Beta biomarker (multitaper field-potential spectrum)")
    print(f"  Beta-band (13-30 Hz) power peaks at {peak_hz:.1f} Hz "
          f"(simulated biomarker at {BETA_HZ:.0f} Hz)")
    print("  Elevated/bursting beta is the feedback signal for adaptive DBS.\n")

    # =====================================================================
    # ACT 5 — DECODING: read the tremor rhythm from a small ensemble (PPAF)
    # =====================================================================
    delta = 0.001
    td = np.arange(0.0, 4.0 + delta, delta)
    rhythm = np.sin(2 * np.pi * TREMOR_HZ * td)     # latent state to recover
    n_cells = 15
    b1 = 0.8 + 0.6 * rng.standard_normal(n_cells)   # tremor-locking strengths
    b0 = np.log(12.0 * delta) + 0.3 * rng.standard_normal(n_cells)
    dN = np.zeros((n_cells, td.size))
    for c in range(n_cells):
        eta = np.clip(b0[c] + b1[c] * rhythm, -20.0, 20.0)
        p = np.exp(eta) / (1.0 + np.exp(eta))
        dN[c, :] = (rng.random(td.size) < p).astype(float)

    A = np.array([[1.0]])
    Q = np.array([[float(np.std(np.diff(rhythm)))]])
    _, _, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, b0, b1.reshape(1, -1), "binomial", delta, None, None,
        np.array([0.0]), 0.5 * np.eye(1),
    )
    x_hat = x_u[0, :]
    rmse = float(np.sqrt(np.mean((x_hat - rhythm) ** 2)))
    print("ACT 5 — Decoding the tremor rhythm (point-process adaptive filter)")
    print(f"  Recovered the {TREMOR_HZ:.0f} Hz rhythm from {n_cells} cells "
          f"(RMSE={rmse:.2f}), with a calibrated credible band.")
    print("  This is the spiking analogue of a Kalman filter (Eden 2004).\n")
    print("Where next: docs/concepts/rhythmic_firing_and_clinical_microelectrode.md")

    if args.save_fig:
        _save_figure(args.save_fig, t, edges, y, lam_rhythm,
                     freqs, power, td, rhythm, x_hat, W_u)
        print(f"\n  Saved figure to {args.save_fig}")


def _verdict(pvalue: float, alpha: float = 0.05) -> str:
    return "FITS (not rejected)" if pvalue > alpha else "REJECTED"


def _save_figure(path, t, edges, y, lam_rhythm, freqs, power,
                 td, rhythm, x_hat, W_u) -> None:
    """Optional summary figure (kept off the main path so the tutorial runs
    without matplotlib installed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])[: y.size]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    axes[0].bar(centers, y, width=(edges[1] - edges[0]), align="center",
                alpha=0.35, label="counts/bin")
    axes[0].plot(centers, lam_rhythm, color="C3", lw=1.2,
                 label="fitted rhythm-aware rate")
    axes[0].set(xlim=(0, 1.0), xlabel="time (s)", ylabel="counts / bin",
                title="Tremor cell: rhythm-aware GLM fit")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].axvspan(13, 30, color="#2f855a", alpha=0.12)
    axes[1].semilogy(freqs, power, color="#2c5282", lw=1)
    axes[1].set(xlim=(0, 60), xlabel="frequency (Hz)", ylabel="power (log)",
                title="Field potential: beta biomarker (13–30 Hz)")

    sigma = np.sqrt(np.maximum(W_u[0, 0, :], 0.0))
    axes[2].plot(td, rhythm, color="#dd6b20", lw=2, label="true tremor rhythm")
    axes[2].plot(td, x_hat, color="#2c5282", lw=1.3, label="PPAF decode")
    axes[2].fill_between(td, x_hat - 1.96 * sigma, x_hat + 1.96 * sigma,
                         color="#2c5282", alpha=0.2, label="95% band")
    axes[2].set(xlim=(0, 1.0), xlabel="time (s)", ylabel="rhythm",
                title="Decode the rhythm from the ensemble")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
