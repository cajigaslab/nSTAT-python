#!/usr/bin/env python3
"""Teaching example — from an encoding model to a goodness-of-fit test.

This is a *tutorial* script: it is written to be read top-to-bottom as a
lesson, not just run. It ties together the ideas in the Concepts pages of
the documentation using only simulated data (no dataset download required),
so you can run it anywhere and see every number.

The story, in four acts
-----------------------
1. ENCODING. We invent a "true" neuron whose instantaneous firing rate (its
   conditional intensity function, CIF) is driven by a sinusoidal stimulus,
   and simulate a spike train from it. This is the generative point-process
   model of Truccolo et al. (2005).
2. FITTING. Pretending we do not know the truth, we fit a point-process GLM
   to the binned spikes and recover the stimulus drive. Because the
   point-process-GLM log-likelihood is concave (Paninski 2004), the fit has a
   unique optimum and converges reliably.
3. GOODNESS-OF-FIT. We ask the crucial question "does the model actually
   fit?" using time-rescaling (Brown et al. 2002; Tao et al. 2018). We
   contrast the CORRECT model against a deliberately WRONG (constant-rate)
   model and watch the test reject the wrong one.
4. WHERE NEXT. A pointer to decoding — reading the stimulus back out of the
   spikes (Eden et al. 2004; Paper Example 05).

Concepts pages
--------------
- Spikes & point-process GLMs:  docs/concepts/spike_trains_and_glms.md
- Goodness-of-fit & decoding:   docs/concepts/goodness_of_fit_and_decoding.md

References (PMIDs verified against PubMed; see docs/concepts/bibliography.md)
- Truccolo et al. 2005, J Neurophysiol 93:1074   PMID 15356183
- Paninski 2004, Network 15:243                  PMID 15600233
- Brown et al. 2002, Neural Comput 14:325        PMID 11802915
- Tao et al. 2018, J Comput Neurosci 45:147      PMID 30298220
- Eden et al. 2004, Neural Comput 16:971         PMID 15070506

Run
---
    python examples/tutorials/encoding_to_goodness_of_fit.py
    python examples/tutorials/encoding_to_goodness_of_fit.py --save-fig out.png
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
    fit_poisson_glm,
    population_time_rescale,
    simulate_cif_from_stimulus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-fig",
        metavar="PATH",
        default=None,
        help="If given, save a raster + rate figure to PATH (needs matplotlib).",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(7)

    # =====================================================================
    # ACT 1 — ENCODING: simulate spikes from a known stimulus-driven CIF
    # =====================================================================
    # A neuron observed for 120 s, with the experiment sampled at 1 kHz.
    dt = 0.001
    t = np.arange(0.0, 120.0, dt)

    # The stimulus the neuron "sees": a slow 1 Hz sinusoid (think of a
    # drifting sensory drive, or a covariate like running speed).
    stimulus = np.sin(2.0 * np.pi * 1.0 * t)

    # The TRUE encoding model.  The CIF is log-linear in the stimulus:
    #     log lambda(t) = beta0 + beta1 * stimulus(t)
    # beta0 sets the baseline rate (it is the log of the baseline rate in Hz);
    # beta1 sets how strongly the stimulus pushes the rate up and down (the
    # "tuning").  This is exactly the point-process GLM form (Truccolo 2005)
    # used here as a *generator*.  We pick a brisk ~12 Hz baseline with strong
    # tuning so there are plenty of spikes for the goodness-of-fit test in
    # Act 3 to have statistical power.
    true_beta0 = 2.5    # baseline ~ exp(2.5) ~ 12 spikes/s
    true_beta1 = 1.5    # stimulus drive (strong tuning)

    spikes, _, _ = simulate_cif_from_stimulus(
        time=t, stimulus=stimulus, beta0=true_beta0, beta1=true_beta1, rng=rng
    )

    # Discretize: count spikes in fine 1 ms bins.  Bin width matters — see the
    # log(bin_width) offset below, which converts a per-bin GLM into a rate.
    # We use 1 ms bins so each bin holds at most one spike; time-rescaling
    # (Act 3) reconstructs spike times from the bins, and coarse bins would
    # quantize the rescaled intervals and corrupt the KS test.
    bin_width = 0.001
    edges = np.arange(0.0, 120.0 + bin_width, bin_width)
    y = spikes.to_binned_counts(edges)                 # observed counts per bin

    # The stimulus, averaged into the same bins, is our design matrix column.
    samples_per_bin = int(round(bin_width / dt))
    x = stimulus.reshape(-1, samples_per_bin).mean(axis=1)[:, None]
    offset = np.full(y.shape, np.log(bin_width))        # makes coefficients rates

    n_spikes = int(y.sum())
    print("ACT 1 — Encoding")
    print(f"  Simulated {n_spikes} spikes over {t[-1]:.0f} s "
          f"({n_spikes / t[-1]:.1f} spikes/s mean rate)")
    print(f"  True parameters:  beta0={true_beta0:+.3f}  beta1={true_beta1:+.3f}\n")

    # =====================================================================
    # ACT 2 — FITTING: recover the encoding model from the spikes alone
    # =====================================================================
    # Fit log lambda = b0 + b1 * stimulus by maximum likelihood (IRLS).
    fit = fit_poisson_glm(x, y, offset=offset, l2=1e-4, max_iter=100, tol=1e-10)
    est_beta0 = fit.intercept
    est_beta1 = fit.coefficients[0]

    print("ACT 2 — Fitting (point-process GLM)")
    print(f"  Estimated:        beta0={est_beta0:+.3f}  beta1={est_beta1:+.3f}")
    print(f"  Recovery error:   |dbeta0|={abs(est_beta0 - true_beta0):.3f}  "
          f"|dbeta1|={abs(est_beta1 - true_beta1):.3f}")
    print(f"  Converged: {fit.converged} in {fit.n_iter} iterations "
          f"(log-likelihood {fit.log_likelihood:.1f})\n")

    # =====================================================================
    # ACT 3 — GOODNESS-OF-FIT: does the model actually fit?  (time-rescaling)
    # =====================================================================
    # The model's expected spike count per bin is its fitted CIF integrated
    # over the bin:  lambda_hat * bin_width = exp(b0 + b1*x + offset).
    lam_correct = np.exp(est_beta0 + est_beta1 * x[:, 0] + offset)

    # A deliberately WRONG model: constant rate, ignoring the stimulus
    # entirely (beta1 = 0).  It will match the *average* rate but miss the
    # stimulus-locked structure.
    lam_constant = np.full_like(y, y.mean())

    # population_time_rescale implements the (marked) time-rescaling KS test.
    # With a single neuron it reduces to the classic Brown-et-al. (2002) KS
    # test on the rescaled inter-spike intervals.  A LARGE p-value means
    # "consistent with the model"; a tiny p-value means "model rejected".
    gof_correct = population_time_rescale([y], [lam_correct])
    gof_constant = population_time_rescale([y], [lam_constant])

    print("ACT 3 — Goodness-of-fit (time-rescaling KS)")
    print(f"  CORRECT model (stimulus + baseline): "
          f"KS p = {gof_correct.ground_ks_pvalue:.3g}  -> "
          f"{_verdict(gof_correct.ground_ks_pvalue)}")
    print(f"  WRONG model   (constant rate):       "
          f"KS p = {gof_constant.ground_ks_pvalue:.3g}  -> "
          f"{_verdict(gof_constant.ground_ks_pvalue)}")
    print("  Lesson: a model can match the mean rate yet fail the KS test")
    print("  because it gets the *timing* wrong.  Always check goodness-of-fit,")
    print("  not just the fitted parameters or the log-likelihood.\n")

    # =====================================================================
    # ACT 4 — WHERE NEXT: decoding
    # =====================================================================
    print("ACT 4 — Where next: decoding")
    print("  We just did ENCODING (stimulus -> spikes) and checked the fit.")
    print("  DECODING inverts it (spikes -> stimulus/state) with the")
    print("  point-process adaptive filter (Eden 2004).  See Paper Example 05")
    print("  and docs/concepts/goodness_of_fit_and_decoding.md.")

    if args.save_fig:
        _save_figure(args.save_fig, t, stimulus, edges, y, lam_correct)
        print(f"\n  Saved figure to {args.save_fig}")


def _verdict(pvalue: float, alpha: float = 0.05) -> str:
    return "FITS (not rejected)" if pvalue > alpha else "REJECTED"


def _save_figure(path, t, stimulus, edges, y, lam_correct) -> None:
    """Optional raster + rate figure (kept out of the main path so the
    tutorial runs without matplotlib installed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t, stimulus, lw=0.8)
    axes[0].set_ylabel("stimulus")
    axes[0].set_title("Encoding model: stimulus drives the firing rate")
    axes[1].bar(centers, y, width=(edges[1] - edges[0]), align="center",
                alpha=0.4, label="observed counts/bin")
    axes[1].plot(centers, lam_correct, color="C3", lw=1.2,
                 label="fitted expected counts/bin")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("counts / bin")
    axes[1].set_xlim(0, 5)   # zoom to the first 5 s so structure is visible
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
