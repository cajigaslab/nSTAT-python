# `nstat.extras.spatial` — Spatial & spatiotemporal point processes

The Python-only spatial / spatiotemporal point-process companion to
`nstat`.  It turns a spatial point pattern into a **posterior rate map
with credible bands** (log-Gaussian Cox process by the Laplace
approximation), provides a **tensor-product B-spline log-rate basis**
that drops into `nstat.glm.fit_poisson_glm`, exposes the **inhomogeneous
second-order goodness-of-fit** suite a homogeneous `K`-function cannot
give for a non-stationary neural field (with four published
edge-correction modes), and implements the **discrete-time-rescaling KS
correction** that fixes a real bug in the naive time-rescaling test at
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
| `lgcp_fit_glm(points, domain, basis, prior, *, grid=32, prior_mean=None, max_iter=50, tol=1e-6)` | Basis-projected LGCP: penalized Poisson IRLS on the B-spline coefficient vector with a `MaternPrior` evaluated at the Greville abscissae (Diggle-Moraga-Rowlingson-Taylor 2013; Wood 2017). The cubic cost scales with the basis dimension `K` (not the cell count `G*G`), so this is the routine to use at `G >= 50`. Returns the same `LGCPResult` as `lgcp_fit`. |
| `MaternPrior(nu, length_scale, marginal_var=1.0, jitter=1e-6)` | Matern GP prior (`nu ∈ {0.5, 1.5, 2.5}`) used by `lgcp_fit_glm`; caches `K`, its Cholesky factor, `K_inv`, and `log_det` per `coords` array. On a Cholesky failure the jitter is bumped 10x and retried once. |
| `LGCPResult.rate_map(level=0.90)` | `(mean, lo, hi)` log-normal credible band: `mean=exp(f̂+v/2)`, `lo/hi=exp(f̂∓z√v)`. Band is **wider in data-sparse cells** (where `Ŵ→0`, `v→` GP prior variance) |
| `LGCPResult.rate_mean()` / `.intensity_fn()` | the posterior-mean rate / a callable `X→rate` for use as `lambda_hat` (mind the plug-in caveat) |

### Inhomogeneous second-order goodness-of-fit (pure NumPy/SciPy)

| Symbol | Notes |
|---|---|
| `pair_correlation(points, lambda_hat, r_grid, *, bw=None, domain=None, edge_correction="epanechnikov")` | SOIRS-reweighted `g(r)`; `>1` clustering, `<1` repulsion, `=1` Poisson null |
| `k_inhom(points, lambda_hat, r_grid, *, domain=None, edge_correction="epanechnikov")` | inhomogeneous `K` (Baddeley-Møller-Waagepetersen 2000); `=πr²` for inhomogeneous Poisson (2-D) |
| `l_function(points, lambda_hat, r_grid, *, domain=None, edge_correction="epanechnikov")` | variance-stabilized `L(r)=√(K/π)`; `L(r)−r=0` under the null |
| `nearest_neighbour_FGJ(points, r_grid, *, domain=None, ...)` | empty-space `F`, nearest-neighbour `G`, `J=(1−G)/(1−F)` |
| `global_envelope(points, lambda_hat, r_grid, *, n_sim=199, domain=None, statistic="pcf", bw=None, alpha=0.05, ...)` | Monte-Carlo global-rank envelope (Myllymäki et al. 2017) → `EnvelopeResult` (`observed`, `lo`, `hi`, `inside`, `p_interval`) |

#### Edge corrections

`pair_correlation`, `k_inhom`, and `l_function` accept an `edge_correction`
keyword selecting one of four published modes.  The default
(`"epanechnikov"`) is the original SOIRS estimator and is bit-identical
to the pre-keyword behaviour.  The other three require a rectangular
`domain = ((xlo, xhi), (ylo, yhi))`.

| Mode | Reference | Notes |
|---|---|---|
| `"epanechnikov"` | Stoyan & Stoyan 1994 | Default; the SOIRS Epanechnikov-kernel estimator already shipped. **Output is bit-identical to omitting the kwarg.** |
| `"isotropic"` | Ripley 1976, 1977 | Per pair `(i, j)` at distance `r`, weight by the symmetric average `0.5 * (1/frac_disc(p_i, r) + 1/frac_disc(p_j, r))` of the inverse fraction of the disc of radius `r` inside the rectangle (Baddeley-Rubak-Turner 2015 eq. 7.6; matches `spatstat::Kinhom`'s `correction="isotropic"`). |
| `"translation"` | Ohser 1983 | Weight each pair by `|W| / |W ∩ W_h|` where `h` is the inter-event offset; symmetric in `i↔j` because the intersection depends only on `|h|`. |
| `"border"` | Baddeley-Rubak-Turner 2015 §7.4 | Restrict the focal event set per radius to events with boundary-distance ≥ `r`; if no event qualifies at that radius, returns `NaN` (not a silent zero). |

### B-spline log-rate basis (pure NumPy/SciPy)

| Symbol | Notes |
|---|---|
| `bspline_basis_1d(grid, n_knots, degree=3, clamped=True)` | `(N, n_knots)` design matrix on a 1-D grid; rows sum to 1 (partition of unity) when `clamped=True`; de Boor 1978 |
| `bspline_basis_2d(grid_x, grid_y, n_knots, degree=3, domain="rect", clamped=True)` | tensor-product `(Nx*Ny, nx*ny)` design matrix; **row layout is `indexing="ij"`** — row `i*Ny + j` evaluates at `(grid_x[i], grid_y[j])`; reshape with `pred.reshape(len(grid_x), len(grid_y))`; `domain="circular"` raises `NotImplementedError` |
| `BSplineBasis2D.from_grid(grid_x, grid_y, n_knots, degree=3, clamped=True)` → `BSplineBasis2D` | frozen dataclass; `.design_matrix()` returns the cached design matrix, `.gram()` returns the P-spline second-difference penalty `Dx.T Dx ⊗ Iy + Ix ⊗ Dy.T Dy` (Eilers-Marx 1996) — symmetric PSD by construction; `.coefficient_coords()` returns the `(K, 2)` Greville-abscissa anchor points (de Boor 1978) of the basis coefficients in ij flattening — feed to `MaternPrior` for `lgcp_fit_glm`. |

The 2-D design matrix is a valid `x` argument to
`nstat.glm.fit_poisson_glm`; a smooth penalty can be added later by
augmenting the IRLS Hessian with `rho * basis.gram()`.

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

Basis-projected LGCP (cheap at large grids):

```python
import numpy as np
from nstat.extras.spatial import lgcp_fit_glm, MaternPrior
from nstat.extras.spatial.basis import BSplineBasis2D

rng = np.random.default_rng(2)
# Same single-bump pattern as above.
mu, peak = np.array([0.45, 0.55]), 900.0
Sinv = np.linalg.inv(np.array([[0.045, 0.008], [0.008, 0.035]]))
def loglam(X):
    d = X - mu
    return np.log(peak) - 0.5 * np.einsum("ni,ij,nj->n", d, Sinv, d)
n = rng.poisson(peak); P = rng.uniform(0, 1, (n, 2))
pts = P[rng.uniform(0, 1, n) < np.exp(loglam(P)) / peak]

# 64x64 cell grid, 10x10 cubic B-spline basis with a Matern-5/2 prior on
# the coefficient vector.  Dominant solve is O(K^3) = O(100^3), not O(M^3).
G = 64
gx = np.linspace(0, 1, G); gy = np.linspace(0, 1, G)
basis = BSplineBasis2D.from_grid(gx, gy, n_knots=10)
prior = MaternPrior(nu=2.5, length_scale=0.18, marginal_var=1.0)
res = lgcp_fit_glm(pts, ((0, 1), (0, 1)), basis, prior, grid=G)
mean, lo, hi = res.rate_map(level=0.90)
```

Discrete-time-rescaling KS:

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
| Basis-projected LGCP (`lgcp_fit_glm` + `MaternPrior`) | shipped (NumPy) |
| Inhomogeneous `g`/`K`/`L`, F/G/J, global-rank envelope | shipped (NumPy) |
| Discrete-time-rescaling correction + marked / multivariate KS | shipped (NumPy) |
| DPP eigen-sampler (`L`-ensemble) | shipped (NumPy fallback) + DPPy bridge |
| Multivariate Hawkes | `tick` bridge (`[hawkes]`) |
| Heavier variational GP for the LGCP | optional `gpflow` path (`[spatial-gp]`) |
| Auto-Poisson / SPDE-GMRF estimators | not in scope |

## References

- Møller J, Syversveen AR, Waagepetersen RP (1998). *Log Gaussian Cox
  processes.* Scandinavian Journal of Statistics 25(3):451-482.
- Rasmussen CE, Williams CKI (2006). *Gaussian Processes for Machine
  Learning*, Algorithm 3.1.
- Diggle PJ, Moraga P, Rowlingson B, Taylor BM (2013). *Spatial and
  spatio-temporal log-Gaussian Cox processes.* Statistical Science
  28(4):542-563.
- Wood SN (2017). *Generalized Additive Models: An Introduction with R.*
  Chapman & Hall/CRC, 2nd ed.
- de Boor C (1978). *A Practical Guide to Splines.* Springer.
- Eilers PHC, Marx BD (1996). *Flexible Smoothing with B-splines and
  Penalties.* Statistical Science 11(2):89-121.
- Ripley BD (1976). *The second-order analysis of stationary point
  processes.* J. Appl. Probab. 13(2):255-266.
- Ripley BD (1977). *Modelling spatial patterns.* JRSS-B 39(2):172-212.
- Ohser J (1983). *On estimators for the reduced second moment measure
  of point processes.* Math. Operationsforsch. Statist., Ser. Statist.
  14(1):63-71.
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns:
  Methodology and Applications with R.* CRC Press.
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
