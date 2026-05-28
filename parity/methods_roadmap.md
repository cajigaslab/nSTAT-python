# Methods incorporation roadmap (2026 review)

> Companion to [`integration_opportunities.md`](integration_opportunities.md).
> That document surveyed the **Python library ecosystem** (which libraries
> to bridge).  This one is a prioritized **methods/algorithms** plan: what
> capabilities to add next, drawn from a 2026 deep-dive of the toolbox plus
> a literature scan (2022–2026).  Every item respects the
> [CLAUDE.md](../CLAUDE.md) independence rule (Python-side only, no MATLAB-repo
> coupling) and the core-vs-`extras` decision rule.
>
> Generated 2026-05-28.

---

## How to read this

Each item lists: **what** it is, **the gap** it fills in nSTAT, **core vs
extras** placement, **difficulty** (thin wrapper / moderate port / large
research effort), and **license/dep** notes.  Items are grouped by tier;
within a tier they are ordered by (value × feasibility).

The independence + core/extras rules from CLAUDE.md are binding:
- A method that exists in MATLAB nSTAT or has a `parity/manifest.yml`
  entry → **core** (`nstat/`).
- A Python-only method, or one needing deps outside the core set
  (PyTorch/JAX/GP libs) → **`nstat.extras`**.
- New optional deps go in `pyproject.toml [project.optional-dependencies]`
  with a backing-group entry recognized by
  `tests/test_pyproject_consistency.py`.

---

## Tier 0 — Finish the in-flight EM work (highest priority)

These are not new methods; they close the gap surfaced by the 2026
deep-dive of `nstat.extras.em.dynamax_bridge` (the PP_EM / mPPCO_EM
trainers).  The current state after that pass: the trainers fit the
**observation model** correctly (firing rates; hybrid Gaussian `R`) and
are numerically stable, but the **latent parameters `A`/`C` are not
uniquely identified** (PLDS gauge freedom — only the scale part is pinned).

### 0.1 Full PLDS identifiability constraints (`PP_EMCreateConstraints` equivalent)

- **What:** Pin the *rotational* part of the PLDS gauge so `A`/`C` are
  uniquely determined and comparable to MATLAB.  Standard canonical
  forms: orthonormal-columns `C` (via per-iteration QR/SVD folded into
  the state transform), or a companion/Schur form for `A`.  MATLAB nSTAT
  does this in `PP_EMCreateConstraints` / `mPPCO_EMCreateConstraints`.
- **Gap:** Today only a diagonal unit-RMS scale normalization is applied
  (`_canonical_scale`), leaving a rotational gauge that lets `|C|` vary
  with seed.  This is the difference between "fits the rates" and
  "recovers interpretable, MATLAB-comparable parameters."
- **Placement:** `extras` (`em.dynamax_bridge`).
- **Difficulty:** Moderate (~1 day): a canonicalization step + tests that
  assert parameter recovery (now meaningful once the gauge is pinned).
- **Validation:** with the gauge fixed, switch the EM tests from
  "rate-tracking" back to true `A`/`C` recovery within tolerance.

### 0.2 Held-out predictive-likelihood diagnostic

- **What:** A proper convergence/quality metric for the EM trainers: the
  true Poisson (and Gaussian) held-out log-likelihood under the fitted
  parameters, replacing the surrogate Gaussian-smoother log-likelihood
  that is currently reported (and which is not a valid objective because
  the IRLS pseudo-observations are re-linearized each iteration).
- **Gap:** Users currently have no trustworthy convergence diagnostic.
- **Placement:** `extras`. **Difficulty:** Thin (~2 h).

---

## Tier 1 — Core goodness-of-fit extensions (cheap, parity-adjacent)

### 1.1 Multivariate & marked time-rescaling KS test — SHIPPED (multivariate)

- **What:** Multivariate (Tao, Weber, Arai, Eden 2018) and discrete-time
  (Haslinger, Pipa, Brown 2010) extensions of the time-rescaling theorem.
  Univariate rescaling KS plots can pass while a model misses inter-neuron
  coupling; the multivariate version catches exactly those failures.
- **Done:**
  - *Discrete-time (Haslinger 2010)* was **already present** — the
    `_ksdiscrete` random-rescaling helper + `computeKSStats(..., dt_correction=1)`,
    with the `ksdiscrete_exactness.mat` gold fixture.
  - *Multivariate / marked (Tao 2018)* is **new**:
    `nstat.population_time_rescale` → `PopulationTimeRescaleResult`
    (core, `nstat/fit.py`).  A ground-process KS (pooled
    `λ_• = Σ λ_k`) plus a marked-region Pearson `χ²`.  Validated: the
    ground KS rejects a synchronous-pair model whose *per-neuron*
    univariate KS both pass (the canonical coupling miss); the `χ²`
    independently catches relative-allocation misfit.  Pure NumPy/SciPy.
- **Placement:** **core** (`nstat/fit.py`).  No `parity/manifest.yml`
  entry: the 2018 method postdates the 2012 MATLAB toolbox, so there is
  no MATLAB counterpart / `matlab_gold` fixture (documented as a
  Python-only extension in the docstring + ClassDefinitions).
- **Difficulty:** Thin (~0.5 day); pure NumPy/SciPy.
- **References:** [Multivariate time-rescaling (PMC3090500)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3090500/),
  [Discrete-time rescaling (PMID 20608868)](https://pubmed.ncbi.nlm.nih.gov/20608868/),
  [Common GOF framework via marked PP rescaling (PMC6208891)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6208891/).

---

## Tier 2 — Flagship `extras` bridge (shares nSTAT's point-process-filter math)

### 2.1 Clusterless / marked point-process state-space decoding + trajectory classification

- **What:** Bayesian state-space decoders that (a) use a **marked
  point-process** observation model — spike-waveform features replace
  spike sorting (clusterless) — and (b) classify the *trajectory type*
  (e.g., replay vs. local) via a discrete latent state on top of the
  continuous decode.  This is the modern descendant of the exact PPAF/PPHF
  point-process filters nSTAT already implements.
- **Gap:** nSTAT has PPAF/PPHF but no clusterless observation model and no
  discrete-state trajectory classifier.  This is the single highest-fit
  extension because it shares nSTAT's point-process-filter mathematics.
- **Placement:** `extras` (new `nstat.extras.decoding` or
  `interop`-style bridge).
- **Difficulty:** Thin-to-moderate bridge to
  [`replay_trajectory_classification`](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification)
  (Denovellis et al. 2021, eLife). **Verify the LICENSE** (likely MIT)
  before adding the dep, per the independence rule.
- **References:** [Denovellis 2021 eLife](https://elifesciences.org/articles/64505),
  [clusterless marked-PP filter (PMC4805376)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4805376/).

---

## Tier 3 — Deeper state-space / encoding methods (moderate ports)

### 3.1 Conditionally Linear Dynamical Systems (CLDS)

- **What:** Linear dynamics *conditioned* on task/behavioral covariates via
  Gaussian-process priors — interpretable, Kalman-tractable, but captures
  condition-dependent/nonlinear dynamics; works in data-limited regimes.
- **Gap:** An interpretable upgrade to nSTAT's linear Kalman / PLDS-EM
  decoders that stays faithful to the "interpretable state-space" ethos
  (vs. black-box LFADS).
- **Placement:** `extras`. **Difficulty:** Moderate port (recent paper;
  check for released code first). Builds on the LDS-EM machinery already
  in `em.dynamax_bridge`.
- **Reference:** [Geadah et al., NeurIPS 2025, arXiv:2502.18347](https://arxiv.org/abs/2502.18347).

### 3.2 Bayesian nonparametric (non-)renewal point processes (NPNR)

- **What:** Sparse-variational-GP generalization of modulated renewal
  processes; nonparametric prior over the conditional ISI distribution,
  with automatic relevance determination for history dependence beyond
  renewal order.
- **Gap:** nSTAT GLMs assume parametric history filters + Poisson/exponential
  ISI structure.  NPNR models non-renewal ISI variability nonparametrically
  — it directly *characterizes* the rescaled-ISI structure nSTAT currently
  only *tests* (KS).  Strong complement to Tier 1.1.
- **Placement:** `extras` (PyTorch/JAX-backed). **Difficulty:**
  Moderate-to-large; best as a wrapper of the authors' code if released.
- **Reference:** [NeurIPS 2023 / bioRxiv 2023.10.15.562381](https://www.biorxiv.org/content/10.1101/2023.10.15.562381v1.full).

### 3.3 Nonparametric GP coupling/history filters for GLMs

- **What:** Infer GLM stimulus/history/coupling filters as continuous-time
  GP functions via sparse variational inference — auto-learns the filter
  temporal span; no hand-chosen raised-cosine/spline basis.
- **Gap:** nSTAT (and NeMoS) use fixed basis expansions.  This removes the
  basis-choice burden and complements, rather than duplicates, the NeMoS
  bridge.
- **Placement:** `extras`. **Difficulty:** Moderate (variational GP).
- **Reference:** [Dowling, Zhao, Park, arXiv:2009.01362](https://arxiv.org/abs/2009.01362).

---

## Tier 4 — Niche / lower priority

- **GLM-Transformer (low-rank population coupling)** — scalable, partly
  interpretable coupling inference for large populations.
  [arXiv:2506.02263](https://arxiv.org/pdf/2506.02263).  `extras`; wait
  for code release.
- **Submillisecond functional-connectivity GLM (Laguerre basis)** —
  sub-bin-resolution coupling estimation.
  [arXiv:2510.20966](https://arxiv.org/abs/2510.20966).  `extras`; niche.

---

## Explicitly NOT recommended

- **LFADS / `lfads-torch`** — large research effort, black-box autoencoder;
  conflicts with nSTAT's interpretability posture and overlaps the
  Dynamax direction.  Point users at
  [`lfads-torch`](https://github.com/arsedler9/lfads-torch) in docs
  instead of bridging.
- **Plain PLDS-EM** — already covered by the Dynamax bridge
  (`fit_linear_gaussian_em` for KF_EM; the PP/hybrid trainers for the
  point-process variants).

---

## Suggested sequencing

1. **Tier 0.1 + 0.2** — finish the EM trainers (identifiability + a real
   diagnostic).  Closes the open deep-dive item; makes the shipped
   PP_EM/mPPCO_EM trustworthy at the parameter level.
2. **Tier 1.1** — multivariate time-rescaling GOF.  Cheapest, core,
   parity-aligned; pairs naturally with the time_rescale oracle already
   in the test tree.
3. **Tier 2.1** — clusterless decoding bridge.  Flagship `extras`
   addition; shares the PPAF/PPHF mathematics nSTAT already owns.
4. **Tier 3** — CLDS, then NPNR / GP-GLM, as `extras` once upstream code
   is confirmed available + license-compatible.

Re-run this review annually or whenever a Tier-1/2 candidate ships a
maintained, license-clear Python implementation.

*All licenses must be verified against the LICENSE file before adding a
dependency — search-time metadata is not authoritative.*
