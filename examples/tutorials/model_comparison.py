#!/usr/bin/env python3
"""Teaching example — model comparison and uncertainty.

How do you decide *which* covariates a spike-train GLM needs? You fit nested
models of increasing complexity, then compare them on two axes:

  * **Penalized likelihood** (AIC / BIC) — does the extra term improve the fit
    enough to justify its parameters?
  * **Goodness-of-fit** (time-rescaling KS) — does the model actually fit in
    absolute terms? A model can win on AIC and still fail KS.

We also quantify **uncertainty**: a coefficient is an estimate, so we report a
95% confidence interval (from the GLM's Fisher information). "The stimulus
coefficient is 0.9 ± 0.1" says far more than "0.9".

Ground truth here is a neuron with BOTH a stimulus drive and spike-history
(a refractory period), so the history term is genuinely needed — the
stimulus-only model should fail goodness-of-fit.

Concepts pages: docs/concepts/spike_trains_and_glms.md,
                docs/concepts/goodness_of_fit_and_decoding.md
References (verified vs PubMed): Brown et al. 2002 (PMID 11802915);
Truccolo et al. 2005 (PMID 15356183).

Run:
    python examples/tutorials/model_comparison.py
    python examples/tutorials/model_comparison.py --save-fig out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nstat import fit_poisson_glm, population_time_rescale  # noqa: E402


def simulate_with_history(t, stim, b0, b1, gamma, rng):
    """Simulate spikes whose rate depends on the stimulus AND recent spikes.

    log λ(t) = b0 + b1*stim(t) + gamma * (spike in previous bin)
    A negative gamma is a refractory effect (less likely to fire right after
    firing) — history that a stimulus-only model cannot capture.
    """
    dt = t[1] - t[0]
    T = t.size
    y = np.zeros(T)
    prev = 0.0
    for i in range(T):
        eta = b0 + b1 * stim[i] + gamma * prev
        rate = np.exp(eta)
        p = 1.0 - np.exp(-rate * dt)        # prob of >=1 spike in the bin
        y[i] = 1.0 if rng.random() < p else 0.0
        prev = y[i]
    return y


def fit_model(X, y, dt):
    """Fit a Poisson GLM; return dict with AIC, BIC, KS p, and coef + 95% CI."""
    offset = np.full(y.shape, np.log(dt))
    if X is None:                               # constant-rate model
        fit = fit_poisson_glm(np.zeros((y.size, 1)), y, offset=offset,
                              include_intercept=True, l2=1e-8)
        coefs = np.array([fit.intercept])
        lam = np.exp(fit.intercept + offset)
        k = 1
        Xaug = np.ones((y.size, 1))
    else:
        fit = fit_poisson_glm(X, y, offset=offset, l2=1e-8, max_iter=200)
        coefs = np.concatenate([[fit.intercept], fit.coefficients])
        lam = np.exp(fit.intercept + X @ fit.coefficients + offset)
        k = coefs.size
        Xaug = np.column_stack([np.ones(y.size), X])

    ll = float(np.sum(y * np.log(lam + 1e-300) - lam))     # Poisson log-lik
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(y.size)
    ks_p = population_time_rescale([y], [lam]).ground_ks_pvalue
    # 95% CI from the GLM Fisher information: cov = (Xaug^T diag(lam) Xaug)^-1
    fisher = Xaug.T @ (lam[:, None] * Xaug)
    se = np.sqrt(np.diag(np.linalg.pinv(fisher)))
    return {"aic": aic, "bic": bic, "ks_p": ks_p, "coefs": coefs, "se": se, "k": k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-fig", metavar="PATH", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(1)

    dt = 0.001
    t = np.arange(0.0, 40.0, dt)
    stim = np.sin(2 * np.pi * 1.0 * t)
    y = simulate_with_history(t, stim, b0=3.2, b1=1.0, gamma=-5.0, rng=rng)
    print(f"Simulated {int(y.sum())} spikes over {t[-1]:.0f} s "
          f"(stimulus + refractory history)\n")

    hist = np.concatenate([[0.0], y[:-1]])      # previous-bin spike (history)
    models = {
        "M0: constant": fit_model(None, y, dt),
        "M1: + stimulus": fit_model(stim[:, None], y, dt),
        "M2: + stimulus + history": fit_model(np.column_stack([stim, hist]), y, dt),
    }

    print(f"{'model':<28}{'k':>3}{'AIC':>11}{'BIC':>11}{'KS p':>10}   verdict")
    best_aic = min(m["aic"] for m in models.values())
    for name, m in models.items():
        star = "  <- best AIC" if m["aic"] == best_aic else ""
        verdict = "fits" if m["ks_p"] > 0.05 else "REJECTED"
        print(f"{name:<28}{m['k']:>3}{m['aic']:>11.1f}{m['bic']:>11.1f}"
              f"{m['ks_p']:>10.2g}   {verdict}{star}")

    m2 = models["M2: + stimulus + history"]
    print("\nUncertainty (M2, 95% CI = estimate ± 1.96·SE):")
    labels = ["intercept", "stimulus (true 1.0)", "history (true -5.0)"]
    for lab, c, s in zip(labels, m2["coefs"], m2["se"]):
        print(f"  {lab:<22} {c:+.3f}  ±{1.96*s:.3f}")
    print("\nLesson: adding the stimulus helps, but only the model WITH spike")
    print("history passes goodness-of-fit. Lowest AIC alone is not enough —")
    print("confirm with the KS test, and report coefficients with their CIs.")

    if args.save_fig:
        _save_figure(args.save_fig, models)
        print(f"\nSaved figure to {args.save_fig}")


def _save_figure(path, models) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(models)
    aic = [models[n]["aic"] for n in names]
    ksp = [models[n]["ks_p"] for n in names]
    short = ["M0\nconstant", "M1\n+stimulus", "M2\n+stim+history"]
    colors = ["#a0aec0" if p <= 0.05 else "#2f855a" for p in ksp]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(short, aic, color=colors)
    ax.set_ylabel("AIC (lower is better)")
    ax.set_title("Model comparison: AIC bars; green = passes KS goodness-of-fit")
    ax.set_ylim(min(aic) * 0.999, max(aic) * 1.001)
    for b, p in zip(bars, ksp):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"KS p={p:.2g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
