# `nstat.extras.spatial` — Spatial & spatiotemporal point processes

The Python-only companion to the bci-curriculum's two point-process
chapters — *Spatial Point Processes* (Ch. 5) and *Spatiotemporal Point
Processes* (Ch. 6).  It turns a spatial point pattern into a **posterior
rate map with credible bands** (log-Gaussian Cox process by the Laplace
approximation), provides the **inhomogeneous second-order
goodness-of-fit** suite a homogeneous `K`-function cannot give for a
non-stationary neural field, and implements the **discrete-time-rescaling
KS correction** that fixes a real bug in the naive time-rescaling test at
finite bin width.

It has **no MATLAB counterpart** and therefore **no `parity/manifest.yml`
entry** — it lives in the opt-in `extras/` namespace precisely so the core
`nstat` MATLAB-parity contract is preserved.  See the
[methods roadmap](../../parity/methods_roadmap.md) "Spatial point
processes" tier.

## Install

The **core** — `lgcp`, `spatial_gof`, `marked_gof` — needs only the
already-present `numpy` / `scipy`.  **No extra to install.**

The optional heavier bridges each have their own group (deliberately
**not** rolled into `[all-extras]`, like `[dynamax]`/`[clusterless]`):

```bash
pip install nstat-toolbox[spatial-gp]   # heavier GP path: lgcp_fit(backend="gpflow")
pip install nstat-toolbox[hawkes]       # multivariate Hawkes via tick
pip install nstat-toolbox[dpp]          # DPP sampling via DPPy
```

The DPP eigen-sampler has a **dependency-free** NumPy fallback
(`dpp_bridge.sample_l_ensemble`) — the `[dpp]` group is only for the
broader DPPy sampler catalogue.

## API

### LGCP rate map (pure NumPy/SciPy)

| Symbol | Notes |
|---|---|
| `lgcp_fit(points, domain, *, grid=20, kernel="matern52", length_scale=0.12, variance=1.0, prior_mean=None, max_iter=50, tol=1e-8, jitter=1e-6, backend="numpy")` | Bins events, places a Matern GP prior on `log Λ`, finds the posterior mode by Newton/IRLS (Rasmussen-Williams Alg. 3.1) → `LGCPResult` |
| `LGCPResult.rate_map(level=0.90)` | `(mean, lo, hi)` log-normal credible band: `mean=exp(f̂+v/2)`, `lo/hi=exp(f̂∓z√v)`. Band is **wider in data-sparse cells** (where `Ŵ→0`, `v→` GP prior variance) |
| `LGCPResult.rate_mean()` / `.intensity_fn()` | the posterior-mean rate / a callable `X→rate` for use as `lambda_hat` (mind the plug-in caveat) |

### Inhomogeneous second-order goodness-of-fit (pure NumPy/SciPy)

| Symbol | Notes |
|---|---|
| `pair_correlation(points, lambda_hat, r_grid, *, bw=None, domain=None)` | SOIRS-reweighted `g(r)`; `>1` clustering, `<1` repulsion, `=1` Poisson null |
| `k_inhom(points, lambda_hat, r_grid, *, domain=None)` | inhomogeneous `K` (Baddeley-Møller-Waagepetersen 2000); `=πr²` for inhomogeneous Poisson (2-D) |
| `l_function(points, lambda_hat, r_grid, *, domain=None)` | variance-stabilized `L(r)=√(K/π)`; `L(r)−r=0` under the null |
| `nearest_neighbour_FGJ(points, r_grid, *, domain=None, ...)` | empty-space `F`, nearest-neighbour `G`, `J=(1−G)/(1−F)` |
| `global_envelope(points, lambda_hat, r_grid, *, n_sim=199, domain=None, statistic="pcf", bw=None, alpha=0.05, ...)` | Monte-Carlo global-rank envelope (Myllymäki et al. 2017) → `EnvelopeResult` (`observed`, `lo`, `hi`, `inside`, `p_interval`) |

### Marked / discrete-time-rescaling goodness-of-fit (pure NumPy/SciPy)

| Symbol | Notes |
|---|---|
| `marked_time_rescaling(spike_bins, marks, p_k, mark_cdf=None, *, decoded=None, n_draws=25, alpha=0.05, rng=None)` | runs the time axis BOTH uncorrected and discrete-time-corrected, plus the mark axis via `F(m\|·)` → `MarkedGOFResult` |
| `uncorrected_rescaled(spike_bins, p_k)` | naive `1−exp(−Σp_k)` variates — **false-rejects** at finite bin width |
| `corrected_rescaled(spike_bins, p_k, rng=None)` | `u_j=[∏(1−p_k)]·(1−r_j·p_{k_j})` — exactly Unif(0,1) (Haslinger-Pipa-Brown 2010) |
| `multivariate_time_rescaling(spike_bins_per_channel, p_k_per_channel, ...)` | per-channel rescaling for a finite (channel) mark space (Gerhard-Haslinger-Pipa 2011) |

### Optional bridges (lazy import; raise an install hint if the dep is absent)

| Symbol | Backend | Group |
|---|---|---|
| `hawkes_bridge.fit_hawkes_exp(event_times, decay=1.0, ...)` → `HawkesFit` | `tick` | `[hawkes]` |
| `dpp_bridge.sample_l_ensemble(L, rng=None)` | **NumPy (no dep)** | — |
| `dpp_bridge.sample_dpp(L, rng=None, *, backend="auto")` | `DPPy` → NumPy fallback | `[dpp]` |
| `lgcp_fit(..., backend="gpflow")` | `gpflow` | `[spatial-gp]` |

## Gotchas

- **Plug-in bias (read this).** The reweighted `g`/`K`/`global_envelope`
  estimators are honest only when `lambda_hat` is **held out** — an
  intensity fit to the *same* pattern has already absorbed some
  clustering, which deflates the statistic and shrinks the envelope below
  nominal coverage (Shaw, Møller & Waagepetersen 2021). Use a smoother
  fit to a disjoint fold (on synthetic data, the known field plays this
  role). `LGCPResult.intensity_fn()` is convenient but *is* a plug-in.
- **Discrete-time correction is mandatory at finite Δ.** A KS test on the
  bin-summed compensator (`uncorrected_rescaled`) false-rejects a correct
  model with bias `O(λ̄Δ)`. Always use `corrected_rescaled` /
  `marked_time_rescaling` (average several `rng` draws, or use a
  Monte-Carlo reference band).
- **Per-channel passing is necessary, not sufficient.** A coupling-blind
  fit can pass every per-channel KS while the joint model is wrong (the
  multivariate time-rescaling theorem).
- **`pair_correlation` is planar (`d=2`).** `lgcp_fit` / `k_inhom` work in
  general `d`, but the `g(r)` ring normalization assumes the plane.

## Recipe

```python
import numpy as np
from nstat.extras.spatial import lgcp_fit, pair_correlation, global_envelope

rng = np.random.default_rng(0)

# Synthetic place field on the unit square (thinned inhomogeneous Poisson).
mu, peak = np.array([0.45, 0.55]), 900.0
Sinv = np.linalg.inv(np.array([[0.045, 0.008], [0.008, 0.035]]))
def loglam(X):
    d = X - mu
    return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)
n = rng.poisson(peak); P = rng.uniform(0, 1, (n, 2))
pts = P[rng.uniform(0, 1, n) < np.exp(loglam(P)) / peak]

# LGCP rate map with a 90% credible band (wider where data thin out).
res = lgcp_fit(pts, ((0, 1), (0, 1)), grid=20)
mean, lo, hi = res.rate_map(level=0.90)

# Held-out SOIRS-reweighted g(r) + global-rank envelope (use the KNOWN
# field as the held-out intensity; on real data use a disjoint-fold smoother).
def lam_at(X): return np.exp(loglam(X))
r = np.linspace(0.03, 0.18, 12)
g = pair_correlation(pts, lam_at, r, domain=((0, 1), (0, 1)), bw=0.04)
env = global_envelope(pts, lam_at, r, n_sim=199, domain=((0, 1), (0, 1)))
print("g(r) ~ 1:", round(float(np.nanmean(g)), 2), "| inside envelope:", env.inside)
```

Discrete-time-rescaling KS (Ch. 6):

```python
import numpy as np
from nstat.extras.spatial import marked_time_rescaling

rng = np.random.default_rng(1)
n_bins, Delta = 6000, 0.020
p_k = np.clip(6.0 * Delta * (1 + 0.5 * np.sin(np.linspace(0, 8*np.pi, n_bins))), 0, 1)
spike_bins = np.flatnonzero(rng.uniform(size=n_bins) < p_k)
res = marked_time_rescaling(spike_bins, None, p_k, n_draws=25, rng=rng)
print("uncorrected rejects:", not res.inside_uncorrected,
      "| corrected passes:", res.inside_corrected)
```

## Scope

| Feature | Status |
|---|---|
| LGCP Laplace rate map + log-normal credible band | shipped (NumPy) |
| Inhomogeneous `g`/`K`/`L`, F/G/J, global-rank envelope | shipped (NumPy) |
| Discrete-time-rescaling correction + marked / multivariate KS | shipped (NumPy) |
| DPP eigen-sampler (`L`-ensemble) | shipped (NumPy fallback) + DPPy bridge |
| Multivariate Hawkes | `tick` bridge (`[hawkes]`) |
| Heavier variational GP for the LGCP | optional `gpflow` path (`[spatial-gp]`) |
| Auto-Poisson / SPDE-GMRF estimators | not in scope (theory-only in Ch. 5) |

## References

- Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
  Learning*, Algorithm 3.1.
- Baddeley AJ, Møller J, Waagepetersen R (2000). *Non- and semi-parametric
  estimation of interaction in inhomogeneous point patterns.* Statistica
  Neerlandica 54(3):329.
- Myllymäki M, Mrkvička T, Grabarnik P, Seijo H, Hahn U (2017). *Global
  envelope tests for spatial processes.* JRSS-B 79(2):381.
- Shaw T, Møller J, Waagepetersen R (2021). *Globally intensity-reweighted
  estimators for K- and pair correlation functions.* ANZJS 63(1):93.
- Haslinger R, Pipa G, Brown E (2010). *Discrete time rescaling theorem.*
  Neural Computation 22(10):2477.
- Gerhard F, Haslinger R, Pipa G (2011). *Applying the multivariate
  time-rescaling theorem to neural population models.* Neural Computation
  23(6):1452.
- Kulesza A, Taskar B (2012). *Determinantal Point Processes for Machine
  Learning.* FnT in ML 5(2-3):123.
- Bacry E, Bompaire M, Gaïffas S, Poulsen S (2018). *tick: a Python library
  for statistical learning.* JMLR 18(214):1.
