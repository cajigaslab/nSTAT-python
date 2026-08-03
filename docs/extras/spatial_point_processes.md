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
[methods roadmap](https://github.com/cajigaslab/nSTAT-python/blob/main/parity/methods_roadmap.md)
"Spatial point processes" tier.

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
| `global_envelope(points, lambda_hat, r_grid, *, n_sim=199, domain=None, statistic="pcf", bw=None, alpha=0.05, edge_correction="epanechnikov", ...)` | Monte-Carlo global-rank envelope (Myllymäki et al. 2017) → `EnvelopeResult` (`observed`, `lo`, `hi`, `inside`, `p_interval`); the `edge_correction` keyword is forwarded to the per-curve summary statistic |
| `cross_k_inhom(points_A, points_B, lambda_A, lambda_B, r_grid, *, domain=None, edge_correction="epanechnikov")` | inhomogeneous cross `K_{AB}(r)` (Baddeley-Møller-Waagepetersen 2000) for two disjoint label classes; `=πr²` under independent inhomogeneous Poisson labels |
| `cross_pair_correlation(points_A, points_B, lambda_A, lambda_B, r_grid, *, bw=None, domain=None, edge_correction="epanechnikov")` | cross pair correlation `g_{AB}(r)` — `>1` cross-attraction, `<1` cross-repulsion, `=1` independent labels |

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
| `multivariate_gof_with_coupling(spike_bins_per_channel, p_k_per_channel, *, n_tau_bins=4, ...)` | runs the per-channel test *and* the population coupling test (`nstat.population_time_rescale`, Tao et al. 2018) on the same data → `CoupledMarkedGOFResult` (`per_channel`, `population`). Per-channel passing is necessary but not sufficient; this wrapper closes that gap |

### Marked-pattern second-order diagnostics (pure NumPy/SciPy)

For a *single* labelled point pattern (one mark per event), the mark
correlation function and mark variogram test whether the marks are
independent of the geometry of the pattern.

| Symbol | Notes |
|---|---|
| `mark_correlation(points, marks, r_grid, *, kernel="schlather", bw=None)` | Kernel mark correlation `k_f(r)` (Schlather 2001 product kernel by default; Stoyan-Stoyan 1994 §13).  `>1` mark clustering, `<1` mark repulsion, `=1` independent marks.  `kernel="isham"` gives the centred-product (mark covariance) form; `kernel="none"` skips the global normalisation. |
| `mark_variogram(points, marks, r_grid, *, bw=None)` | Kernel mark variogram `γ_m(r) = ½ E[(m_i − m_j)² | ‖x_i − x_j‖ = r]` (Cressie-Hawkins 1980; Stoyan-Stoyan 1994 §13).  Identically zero for a constant mark field; approaches `Var(m)` under independent marks. |

Both estimators return `NaN` at lags where the Epanechnikov kernel
weight sums to zero (no pairs in support) — never a silent zero.

### Rescaled-time autocorrelation (independence diagnostic)

The discrete-time-rescaling KS test checks the *marginal* distribution
of the rescaled variates (Unif(0,1)) but is blind to serial dependence
(Brown et al. 2002).  `rescaled_acf` returns the lag autocorrelation
of the normal-score-transformed uniforms `z_j = Φ⁻¹(u_j)` and an
asymptotic Bartlett band (Andersen 1997; Truccolo et al. 2005) — a
complement to, not a replacement for, the marginal KS test.

| Symbol | Notes |
|---|---|
| `rescaled_acf(u_rescaled, *, n_lags=20)` → `RescaledACFResult` | `acf` at lags `1..n_lags`, with the two-sided `±1.96/√n` Bartlett band and a per-lag `inside_band` mask |

### Smoothed point-process residuals

| Symbol | Notes |
|---|---|
| `pp_residuals_smoothed(spike_bins, lam_per_bin, bandwidth, *, dt=1.0)` → `(t_grid, residuals)` | Convolves the per-bin residual `e_k = N_k − λ̂_k Δ` with a normalised Gaussian kernel of standard deviation `bandwidth` *bins*.  Centred at zero under the true model (Brown et al. 2002); a sustained drift flags a time-localised mis-fit (Andersen 1997; Truccolo et al. 2005) |

### LGCP Bartlett density from pair correlation

| Symbol | Notes |
|---|---|
| `bartlett_density_from_pcf(r_grid, g_of_r, k_grid=None)` → `(k_grid, S)` | Hankel-zero transform of `(g(r) − 1)` — the 2-D spatial Bartlett spectral density (Bartlett 1964; Stein 1999 §3).  Default `k_grid`: 64 log-spaced wavenumbers from `π / r_max` to `π / Δr_min`.  Accurate over the body of the wavenumber grid; the top decile near `k_max` is sensitive to the lag-grid truncation (Stein 1999 §3) and is best excluded from any closed-form comparison. |

For an LGCP driven by a Gaussian log-rate field of covariance `C(r)`,
the population pair correlation is `g(r) = exp(C(r))` (Møller,
Syversveen & Waagepetersen 1998), so the Bartlett density of the
empirical `g(r)` can be compared to the closed-form transform of
`exp(C(r)) − 1` for parametric covariance families (Matérn, Gaussian).

### Cluster Cox processes (pure NumPy/SciPy)

Three canonical cluster Cox processes — homogeneous Poisson parents
with `Poisson(μ)` offspring counts scattered by an offspring kernel.
They are the discrete-burst counterpart to the smoothly-varying
log-Gaussian Cox process in [`lgcp`](#lgcp-rate-map-pure-numpyscipy)
and the natural targets for the minimum-contrast estimator below.

| Symbol | Notes |
|---|---|
| `ThomasProcess(intensity_parent, mu_offspring, sigma)` | Thomas (1949): isotropic Gaussian offspring displacement of standard deviation `σ`.  Frozen dataclass; rejects non-positive parameters at construction. |
| `MaternClusterProcess(intensity_parent, mu_offspring, radius)` | Matérn (1986): offspring uniform on the disc of radius `R`. |
| `NeymanScottCox(intensity_parent, mu_offspring, offspring_kernel, pad=0.0)` | Neyman & Scott (1958): generic cluster Cox with a user-supplied `offspring_kernel(n, rng) -> (n, 2)` displacement.  Emits a warning at simulation time if `pad == 0` (the parent window is unbuffered and the in-window pattern is edge-biased). |
| `thomas_pair_correlation(r, sigma, intensity_parent, mu_offspring)` → `g(r)` | Closed form `1 + exp(-r²/(4σ²)) / (4π σ² λ_p)` (Møller-Waagepetersen 2003 §5.3).  `mu_offspring` is accepted for API symmetry but does not enter `g(r)`. |
| `matern_cluster_pair_correlation(r, radius, intensity_parent, mu_offspring)` → `g(r)` | `g(r) = 1 + h(r;R)/(π R² λ_p)` on `r ≤ 2R`, `g(r) = 1` thereafter, with `h(r;R) = (1/π)·[2 arccos(r/(2R)) − (r/R)·√(1 − r²/(4R²))]` (Diggle 2013 §6.2.1). |
| `simulate_thomas(intensity_parent, mu_offspring, sigma, window, *, rng)` → `(n, 2)` float64 | Parent buffer `pad = 3σ`; offspring cropped to `window = (xmin, ymin, xmax, ymax)`. |
| `simulate_matern_cluster(intensity_parent, mu_offspring, radius, window, *, rng)` → `(n, 2)` | Parent buffer `pad = R` (exact kernel support). |
| `simulate_neyman_scott(process, window, *, rng, return_parents=False)` | Generic dispatcher; `return_parents=True` additionally returns the (un-cropped) parent locations. |

### Minimum-contrast estimation (Diggle 2013 §6.2.1)

| Symbol | Notes |
|---|---|
| `min_contrast_estimator(g_emp, g_model_fn, r_grid, theta0, *, bounds=None, q=0.25)` → `MinContrastResult` | Minimises `S(θ) = ∫ [g_emp(r)^q − g_model(r;θ)^q]² dr` by L-BFGS-B (Simpson integration on the lag grid).  NaN samples (typical of the `"border"` edge correction at small `r`) are silently dropped before integration; the call NEVER raises on a non-converging optimiser — `success=False` propagates instead.  Default `q = 0.25` (Møller-Waagepetersen 2003 §4.2). |
| `fit_thomas(points, domain, r_grid, theta0=None)` → `MinContrastResult` | Fits `(σ, λ_p)`.  Default `theta0 = (0.1·diam, 10.0)`; bounds `[(1e-6, None), (1e-3, None)]`.  Recover `μ̂ = n / (λ̂_p · |W|)` post-hoc — second-order statistics do NOT identify `μ`. |
| `fit_matern_cluster(points, domain, r_grid, theta0=None)` → `MinContrastResult` | Same shape for `(R, λ_p)`. |
| `MinContrastResult` | Frozen dataclass: `theta_hat`, `objective_value`, `g_model_at_theta`, `n_iter`, `success`, `message`. |

`fit_thomas` / `fit_matern_cluster` build the empirical `g(r)` with
`pair_correlation(..., edge_correction="border")` (Baddeley-Rubak-Turner
2015 §7.4), so they handle the small-`r` `NaN` band correctly without
extra plumbing.

### Gibbs interaction processes (pure NumPy/SciPy)

Inhibitory or interactive point patterns whose Papangelou conditional
intensity factors into a log-linear form — fit via the Berman-Turner
device (Baddeley-Rubak-Turner 2015 §13.5) by reformulating
pseudo-likelihood as a weighted Poisson GLM and delegating to
`nstat.fit_poisson_glm`.
Companion to the cluster-Cox attractive models above.

| Symbol | Notes |
|---|---|
| `GibbsStrauss(beta, gamma, interaction_radius)` | Strauss (1975): pairwise interaction `γ ∈ (0, 1]` on pairs within `R`.  `γ = 1` recovers Poisson; `γ < 1` repels.  Frozen dataclass; rejects `γ > 1` at construction (use `fit_thomas` for clustered data). |
| `HardcoreProcess(beta, hardcore_radius)` | The `γ → 0` limit of Strauss — forbids pairs closer than `R`. |
| `AreaInteractionProcess(beta, eta, interaction_radius)` | Baddeley-van Lieshout (1995): higher-order Widom-Rowlinson interaction on the union of `R`-discs. |
| `simulate_strauss_birth_death(process, window, *, n_steps=5000, pixel_resolution=256, rng)` → `(n, 2)` | Metropolis birth-death sampler with conditional-intensity acceptance.  Defaults match Møller-Waagepetersen 2003 §10.4 mixing recommendations. |
| `simulate_hardcore_rejection(process, window, *, max_attempts=1_000_000, rng)` → `(n, 2)` | Dart-throwing rejection; raises `RuntimeError("…; consider simulate_strauss_birth_death with γ ≈ 0…")` if it cannot place a target-Poisson(β·\|W\|) count after `max_attempts`. |
| `pseudo_likelihood_fit(points, process, domain, *, n_dummies=None)` → `GibbsFitResult` | Berman-Turner fit.  Uses the equivalent `Y = z` (indicator) form with `offset = log(w)` — algebraically identical to the literal `Y = y/w` weighted formulation but numerically stable (Baddeley-Rubak-Turner 2015 §13.5 eq. 13.21).  Default `n_dummies = max(4·len(points), 1000)`. |
| `GibbsFitResult` | Frozen dataclass: `theta_hat`, `glm_result` (the underlying `PoissonGLMResult`), `process_kind`, `interaction_radius`, `n_data`, `n_dummies`. |

`pseudo_likelihood_fit` emits `UserWarning("data appears clustered;
consider fit_thomas")` when the unclipped Strauss `γ` estimate exceeds
1, then clips to the valid `(0, 1]` interval.  For `AreaInteractionProcess`,
both the simulator and the fitter require `R ≥ 2 · pixel_size` to keep
the discretised disc-union integrals well-defined.

The hardcore intensity estimator is upward-biased (median ≈ 40% at
small `R` for the dart-throwing simulator): the intercept-only GLM
over-attributes activity to unexcluded quadrature area.  The
Baddeley-Rubak-Turner 2015 §13.4 analytical correction
`β̂ / (1 − π R² λ̂)` closes the gap but is not applied automatically
— check the test docstring in
`tests/extras/test_spatial_pseudo_likelihood.py` for the bias
characterisation if you need calibrated intensity recovery.

### Hawkes EM declustering (pure NumPy/SciPy)

Pure-NumPy single-realisation EM for the univariate Hawkes process with
an exponential triggering kernel (Veen-Schoenberg 2008; Marsan-Lengliné
2008).  The algorithm exploits the **branching-structure** interpretation
of the Hawkes process: every event is either spontaneous (parent = self,
rate `mu`) or triggered by an earlier event `j` (kernel
`alpha * exp(-beta * (t_i - t_j))`).  The E-step assigns soft
responsibilities `p_ii` and `p_ij` from those rate weights; the M-step
maximises the resulting `Q(theta)` in closed form for `mu`, `alpha`, and
`beta` — no inner numerical optimisation.  This is the methodological
opposite of the optional tick-backed `hawkes_bridge.fit_hawkes_exp`,
which fits a multivariate Hawkes by SGD-style maximum-likelihood without
recovering the latent parent assignments.

| Symbol | Notes |
|---|---|
| `HawkesEMSpec(mu0=None, alpha0=0.5, beta0=1.0, max_iter=100, tol=1e-6)` | Frozen dataclass.  `mu0=None` auto-infers `N/T` at fit time.  Validates that `alpha0, beta0 > 0`, `mu0 > 0` (or `None`), `max_iter >= 1`, `tol > 0`; warns (`UserWarning`) when the initialisation is super-critical (`alpha0/beta0 >= 1`). |
| `em_hawkes_exponential(event_times, T, *, spec=None, return_responsibilities=False)` → `HawkesEMResult` | Fits the Hawkes via Veen-Schoenberg EM.  Raises on empty input, unsorted event times, or `T <= event_times[-1]`.  Single-event input returns a degraded Poisson(`1/T`) result.  `return_responsibilities=True` lazy-imports `scipy.sparse` and attaches a lower-triangular CSR matrix where row `i` carries event `i`'s parent distribution. |
| `HawkesEMResult` | Frozen dataclass: `mu_hat`, `alpha_hat`, `beta_hat`, `log_likelihood_trace` (per-iteration LL), `n_iter`, `converged`, `responsibilities` (CSR or `None`), and the `.branching_ratio` property (`alpha_hat / beta_hat`). |
| `simulate_hawkes_exponential(mu, alpha, beta, T, *, rng)` → `(M,)` float64 | Ogata-thinning Hawkes simulator on `[0, T]`.  Raises `ValueError` on super-critical `alpha / beta >= 1` (expected event count is infinite). |

**Memory caveat.**  The implementation materialises the full
`(N, N)` time-difference matrix, so memory cost is `O(N^2)` — fine up
to `N ≈ 5000` in dense NumPy, beyond which the tick-backed
`hawkes_bridge.fit_hawkes_exp` (multivariate, stochastic-gradient inner
loop) is the better choice even for the univariate case.

**Identifiability caveat.**  The Hawkes log-likelihood is only weakly
identified between the kernel amplitude `alpha` and decay `beta` on a
single realisation — `alpha` and `beta` trade off along an iso-`alpha/beta`
ridge.  The **branching ratio** `alpha/beta` exposed via
`HawkesEMResult.branching_ratio` is by far the tightest summary; prefer
it to the individual parameters when reporting recovered structure.

### SBM-prior multivariate Hawkes EM (pure NumPy)

Latent-block extension of the Veen-Schoenberg branching-responsibilities
EM (Linderman, Adams & Pillow 2014).  Where `em_hawkes_exponential`
fits a *single* univariate Hawkes with `(mu, alpha, beta)`, the SBM
prior pools `N` processes into `K` latent communities and fits the
`K x K` block-coupling matrix `A` plus a shared exponential decay
`beta` and per-process baseline rates `mu`.  Pure NumPy; no `[hawkes]`
dep; built for `N` up to ~30 and total events up to ~5000.

The block-coupling matrix `A` is per-(sender, receiver) pair, so for
the multivariate process to be sub-critical the spectral radius of
`A * diag(K_counts) / beta` (or equivalently `A_pair / beta` where
`A_pair[m, n] = A[z_m, z_n]`) must be below 1.  The simulator enforces
this; configurations failing it raise `ValueError("super-critical:
spectral radius of A/beta = ...")`.

| Symbol | Notes |
|---|---|
| `SBMHawkesSpec(n_blocks, alpha0=0.3, alpha0_off=0.05, beta0=1.0, max_iter=100, tol=1e-6, z_init="kmeans-rate")` | Frozen dataclass.  Validates `n_blocks >= 1`, `alpha0 > 0`, `alpha0_off >= 0`, `beta0 > 0`, `max_iter >= 1`, `tol > 0`, `z_init in {"random", "kmeans-rate"}`; warns when `alpha0/beta0 >= 1` (intra-block super-critical at init). |
| `em_sbm_hawkes(event_times_per_process, T, *, spec, seed=None)` → `SBMHawkesResult` | Hard-EM with closed-form `(mu, beta, A)` M-step and a greedy accept-if-LL-up update for `z`.  Raises on empty list / silent process / unsorted events / `T <= max event` / `n_blocks > N`.  Algorithm builds a kernel-response cache `R[n][i, m] = sum_{j: t_j^m < t_i^n} exp(-beta(t_i^n - t_j^m))` once per beta to amortise the greedy z update. |
| `SBMHawkesResult` | Frozen dataclass: `z_hat (N,) int`, `A_hat (K, K)`, `beta_hat`, `mu_hat (N,)`, `log_likelihood_trace`, `n_iter`, `converged`, `n_processes`, `n_blocks`. |
| `simulate_sbm_hawkes(z, A, beta, mu, T, *, rng)` → `list[(M_n,) float64]` | Multivariate Ogata thinning with multiplicative-decay state updates.  Raises on super-critical configurations. |

**Permutation caveat.**  Block labels are arbitrary; only the
partition matters.  Recovery tests should compare `z_hat` to ground
truth up to label permutation — `scipy.optimize.linear_sum_assignment`
on the (negative) confusion matrix is the convenient Hungarian
alignment.

**Cost.**  Per EM iteration the dominant work is the `O(N^2 *
avg_events^2)` build of `R` (re-done only when `beta` changes), plus
`O(N * total_events * K)` per greedy z sweep.  For `N <= 30` and total
events `<= 5000` the whole fit finishes in seconds.  For larger
problems prefer the tick-backed multivariate
`hawkes_bridge.fit_hawkes_exp`.

### Optional bridges (lazy import; raise an install hint if the dep is absent)

| Symbol | Backend | Group |
|---|---|---|
| `hawkes_bridge.fit_hawkes_exp(event_times, decay=1.0, ...)` → `HawkesFit` | `tick` | `[hawkes]` |
| `dpp_bridge.sample_l_ensemble(L, rng=None)` | **NumPy (no dep)** | — |
| `dpp_bridge.sample_dpp(L, rng=None, *, backend="auto")` | `DPPy` → NumPy fallback | `[dpp]` |
| `lgcp_fit(..., backend="gpflow")` | `gpflow` | `[spatial-gp]` |

### Spatiotemporal wave analysis (pure NumPy/SciPy)

Bartlett (frequency × wave-vector) spectrum of a Hawkes triggering
matrix and a top-N wave-peak detector — Python-only, no `[hawkes]`
needed (despite living alongside `fit_hawkes_exp`).  Companion to
Daley & Vere-Jones (2003) §8.4 / Bacry-Mastromatteo-Muzy (2015).

| Symbol | Notes |
|---|---|
| `bartlett_spectrum(triggering_matrix, electrode_positions, freq_grid, wave_vector_grid, *, decay=1.0, return_complex=False)` | `(Nf, Nk)` real power by default; `return_complex=True` returns the complex spectrum.  Warns when the adjacency's spectral radius is `>= 1` (Hawkes stationarity violated). |
| `reconstruct_kernel(adjacency, decays, tau_grid)` | `(C, C, Nt)` exponential-family kernel `A[c1, c2] * exp(-decays * tau)` for `tau >= 0`.  Non-exponential families are out of scope. |
| `detect_wave_peaks(spectrum, freq_grid, wave_vector_grid, *, n_peaks=3, min_separation_bins=1)` → `WaveAnalysisResult` | Greedy descending-power sort + Chebyshev non-max suppression; masks DC `|k|=0` rows; returns `(freq, kx, ky, power, speed, direction)`. |
| `WaveAnalysisResult` | Frozen dataclass; `speed = 2*pi*freq / |k|` in `position-unit / s`, `direction = atan2(ky, kx)` in radians. |

### Spatiotemporal point processes (v0.6.0, pure NumPy/SciPy)

Six modules extend the static-spatial toolkit above to full space-time
`(x, y, t)` point processes — kernel intensity estimation, second-order
goodness-of-fit, two self-exciting (Hawkes/ETAS) families, a
spatiotemporal log-Gaussian Cox process, and a rate-modulated renewal
(conditional-ISI) model.  Every one is an original implementation
directly from the published equations cited per module below — none is
a port of an external package — and, like the rest of this module,
none has a MATLAB nSTAT counterpart, so there is no
`parity/manifest.yml` entry for any of them.

#### Space-time kernel intensity (`st_intensity`)

| Symbol | Notes |
|---|---|
| `intensity_st_kde(points, times, *, domain, period=None, bw_space=None, bw_time=None, separable=False, grid=(40,40,40), kernel="gaussian")` | Boundary-corrected product-kernel space-time intensity λ̂(x,t) (Diggle 2013, Ch. 7). The edge correction is a query-point-evaluated **uniform/local edge correction in the spirit of Diggle (1985)** — not his original per-event correction — so `∫∫ λ̂ ≈ n` only *approximately* (exact well inside the window, approximate near the boundary). `separable=True` fits the marginal `m(x)`/`mu(t)` independently and combines as `m(x)*mu(t)/N`; the default (`False`) fits the full non-separable product kernel. `period` defaults to `(times.min(), times.max())` and, despite the parameter name, is an *observation interval* used for the boundary correction — **not** a periodic/wrap-around condition. `bw_space`/`bw_time` default to Silverman's (1986) normal-reference rule of thumb. → `STIntensityResult` |
| `STIntensityResult` | Frozen dataclass: `grid_x` (`(Gx*Gy, 2)`), `grid_t` (`(Gt,)`), `intensity` (`(Gt, Gx*Gy)`), `bw_space`, `bw_time`, `separable`, `domain`, `period`, `kernel`, `points`, `times`. `.evaluate(x, t)` re-evaluates the estimator formula at arbitrary query points (`x` is `(m, 2)` or `(2,)`, `t` is `(m,)` or scalar; broadcasts when one side has length 1) rather than interpolating the grid — this is the exact callable the space-time `K`/pair-correlation estimators below expect as `lambda_hat`. |

#### Space-time inhomogeneous second-order goodness-of-fit (`spatiotemporal_gof`)

| Symbol | Notes |
|---|---|
| `k_st_inhom(points, times, lambda_hat, r_grid, t_grid, *, domain, period, edge_correction="translation")` | SOIRS-reweighted inhomogeneous space-time `K`-function (Diggle-Chetwynd-Haggkvist-Morris 1995; Gabriel-Diggle 2009). **Poisson null:** `K_st(r, t) = pi * r**2 * 2*t`. `edge_correction` is `"isotropic"` (Ripley spatial factor × 1-D translation temporal factor), `"translation"` (Ohser 1983 spatial × 1-D translation temporal — the default), or `"border"` (Baddeley-Rubak-Turner 2015 spatial × 1-D border temporal; `NaN` where no event/interval qualifies as a usable focal point). → `STKResult`, with a `.l_st()` variance-stabilising transform `sqrt(K_st / (2*pi))`. |
| `pair_correlation_st(points, times, lambda_hat, r_grid, t_grid, *, bw_r=None, bw_t=None, domain, period)` | SOIRS-reweighted space-time pair correlation `g(r, t)` (Møller-Ghorbani 2012). **Poisson null:** `g(r, t) ≡ 1`; `>1` clustering, `<1` inhibition. No geometric edge correction beyond the SOIRS reweighting (matches the static `pair_correlation`'s uncorrected default) — use `k_st_inhom`'s `edge_correction` instead when the boundary matters for a cumulative statistic. Returns a plain `(len(r_grid), len(t_grid))` array, not a result dataclass. |
| `global_envelope_st(points, times, lambda_hat, r_grid, t_grid, *, n_sim=199, statistic="kst", alpha=0.05, domain, period, rng=None, edge_correction="translation")` | Monte-Carlo **global-rank envelope** test (Myllymäki et al. 2017) against the inhomogeneous space-time Poisson null — the joint extreme-rank envelope over the flattened `(r, t)` surface, not per-cell order statistics. Null realisations are drawn by Lewis-Shedler thinning of the fitted `lambda_hat` over `domain x period`. `statistic` is `"kst"` (default, `K_st`), `"lst"` (the `.l_st()` transform), or `"gst"` (`pair_correlation_st`, always uncorrected regardless of `edge_correction`). → `STEnvelopeResult` with `.inside` (`True` ⇒ fails to reject the null) and `.p_interval` (conservative/liberal p-value bounds). |
| `STKResult` | Frozen dataclass: `r_grid`, `t_grid`, `k_st` (`(nr, nt)`), `edge_correction`; carries the `.l_st()` method above. |
| `STEnvelopeResult` | Frozen dataclass: `r_grid`, `t_grid`, `observed`, `lo`, `hi` (each `(nr, nt)`), `inside`, `p_interval`. |

Same **plug-in-bias** caveat as the static-spatial module applies here:
pass a **held-out** `lambda_hat` (e.g. `intensity_st_kde` fit to a
disjoint calibration epoch), never a fit to the same pattern being
tested.

#### Space-time Hawkes / ETAS via branching EM (`spatial_hawkes`)

| Symbol | Notes |
|---|---|
| `em_spatial_hawkes(points, times, *, domain, T, spec=None, return_responsibilities=False)` | Branching EM (Veen-Schoenberg 2008) for a separable space-time self-exciting process with a **homogeneous** background: `lambda(x, t) = mu/|W| + sum_{t_j<t} K * c*exp(-c*(t-t_j)) * N(x-x_j; sigma^2 I)`. Both the temporal kernel `g(t) = c*exp(-c*t)` and the spatial kernel `h(r)` are individually normalised to integrate to 1, so **`K` IS the branching ratio directly** — no `alpha/beta`-style division needed (contrast the static `hawkes_em.HawkesEMResult.branching_ratio` property). Dense `O(N^2)` implementation, same memory caveat as the static `hawkes_em`. → `SpatialHawkesResult` |
| `simulate_spatial_hawkes(mu, K_branch, c, sigma_space, *, domain, T, rng)` → `(points, times)` | Branching/Poisson-cluster simulator (Møller-Rasmussen 2005): homogeneous background immigrants on `W x [0, T]` plus `Poisson(K_branch)` direct offspring per event with an `Exp(c)` temporal delay and an `N(0, sigma_space^2 I)` spatial offset. Raises `ValueError` on `K_branch >= 1` (super-critical: infinite expected event count). |
| `SpatialHawkesSpec(mu0=None, K0=0.5, c0=1.0, sigma0=0.1, max_iter=200, tol=1e-6)` | Frozen dataclass of initial guesses + convergence config. `mu0=None` auto-infers `N/T` at fit time. Warns (`UserWarning`) when `K0 >= 1` (super-critical initialisation). |
| `SpatialHawkesResult` | Frozen dataclass: `mu_hat` (**total** background rate over the whole window `W`, so the background density is `mu_hat/|W|`), `K_branch_hat`, `c_hat`, `sigma_space_hat`, `log_likelihood_trace`, `n_iter`, `converged`, `responsibilities` (lower-triangular CSR or `None`; row `i`'s diagonal cell is P(background), cell `[i, j]` with `j < i` is P(`j` triggered `i`); rows sum to 1). |

#### Spatiotemporal LGCP by Kronecker Laplace approximation (`lgcp_st`)

| Symbol | Notes |
|---|---|
| `lgcp_st_fit(points, times, *, domain, period, grid=(24,24,24), length_scale_space=0.12, length_scale_time=0.1, nu=1.5, variance=1.0, prior_mean=None, max_iter=50, tol=1e-8, jitter=1e-6)` | Bins events on a 3-D `(x, y, t)` grid; places a **separable Matérn GP prior** on the log-rate with `K = Kx ⊗ Ky ⊗ Kt` (Kronecker product of three 1-D Matérn factors); finds the posterior mode by Newton/IRLS (Rasmussen-Williams 2006 Alg. 3.1) using a **matrix-free conjugate-gradient** solve (`scipy.sparse.linalg.cg` against a `LinearOperator`) that never forms the dense `(Gx*Gy*Gt)^2` covariance, and estimates the posterior-variance diagonal by a **Hutchinson stochastic estimator** (64 Rademacher probes; Hutchinson 1990). `nu` (shared Matérn smoothness for all three axes) is one of `{0.5, 1.5, 2.5}`. Higher `variance` gives a heavier-tailed / less-informative prior; **`variance -> inf` is the Poisson-MLE limit** (flat prior everywhere, mode collapses to `log(count / cell_volume)` per cell) — **not** `variance -> 0`. → `LGCPSTResult` |
| `LGCPSTResult` | Frozen dataclass: `grid_x`, `grid_t`, `counts`, `f_mode`, `f_var` (all `(Gt, Gx*Gy)` except `grid_x`/`grid_t`), `cell_volume`, `n_iter`, `converged`. `.rate_map(t, level=0.90)` snaps `t` to the nearest `grid_t` entry and returns the `(mean, lo, hi)` **log-normal credible band** at that time slice: `mean = exp(f_mode + f_var/2)`, `lo/hi = exp(f_mode ∓ z*sqrt(f_var))`. `.intensity_fn()` returns a callable `(X, t) -> rate` (nearest-cell lookup of the posterior-mean rate) — a convenient Cox-process background function, e.g. for `simulate_cox_hawkes` below. |

#### Cox-Hawkes: LGCP background × spatial-Hawkes excitation (`cox_hawkes`)

| Symbol | Notes |
|---|---|
| `fit_cox_hawkes(points, times, *, domain, period, grid=(24,24,24), length_scale_space=0.12, length_scale_time=0.1, max_outer=20, tol=1e-4, hawkes_spec=None)` | Alternating estimator for the doubly-stochastic model of Miscouridou et al. (2022) — an inhomogeneous LGCP background plus spatial-Hawkes excitation — fit by a declustering EM built from the two modules above (not that paper's own joint-MCMC inference): init `lgcp_st_fit` treating every event as background → E-step soft background/triggered responsibilities → **deterministic weighted-histogram** LGCP background refit (bins each event's background-responsibility *mass* rather than a stochastic thinning draw, so `fit_cox_hawkes` has **no RNG parameter**; repeated calls on the same input are bit-identical) → closed-form excitation M-step (`K`, `c`, `sigma_space`) → iterate to `tol`/`max_outer`. `length_scale_space`/`length_scale_time` forward unchanged to *every* background refit; the defaults (0.12, 0.1) are calibrated for a **unit-square domain** — for a physically-sized domain (e.g. an 8mm microelectrode array), scale them to the domain extent or fit on a normalised `((0, 1), (0, 1))` domain and rescale the rate afterward. → `CoxHawkesResult` |
| `simulate_cox_hawkes(background_intensity_fn, K_branch, c, sigma_space, *, domain, T, rng, bg_max=None)` → `(points, times)` | Two-stage simulator: inhomogeneous background immigrants via Lewis-Shedler (1979) dominating-rate thinning of `background_intensity_fn(X, t)`, then the same branching offspring cascade as `simulate_spatial_hawkes`. `bg_max=None` auto-estimates a dominating rate by probing 5000 random points and padding by 1.25x (a heuristic, not a guarantee — pass an explicit `bg_max` if the true peak of `background_intensity_fn` is known). |
| `CoxHawkesResult` | Frozen dataclass: `background` (an `LGCPSTResult`), `K_branch_hat`, `c_hat`, `sigma_space_hat`, `background_fraction` (mean posterior P(background) across all events), `log_likelihood_trace`, `n_outer`, `converged`, `event_points`, `event_times`. `.intensity_fn()` returns the **exact** fitted `(X, t) -> rate` callable (background + excitation summed over the observed event history, not a mean-field `1/(1-K)` surrogate). |

#### Modulated renewal / conditional-ISI point process (`modulated_renewal`)

| Symbol | Notes |
|---|---|
| `fit_modulated_renewal(spike_times, covariates, *, renewal="inverse_gaussian", basis=None, dt=None, penalty=0.0, max_iter=100, tol=1e-6, n_inner=12)` | Penalized-ML fit of a rate-modulated renewal CIF (Barbieri-Quirk-Frank-Wilson-Brown 2001; the Cox 1955 time-change construction): `lambda(t | H_t) = lambda_0(t) * r(s(t); theta)`, where `lambda_0(t) = exp(x(t)'beta)` is a log-linear Poisson-GLM rate and `s(t)` is elapsed time since the last spike measured in `lambda_0`-rescaled ("operational") time. `renewal` is `"gamma"` or `"inverse_gaussian"` (default); both are parametrised with mean 1 so `theta` alone sets `CV = 1/sqrt(theta)` — gamma `theta=1` is exactly Exponential(1), the exact Poisson limit. Outer alternation of a beta-step (`nstat.glm.fit_poisson_glm` on the modulated design with `offset = log(hazard) + log(dt)`, itself an inner **damped Picard** loop of up to `n_inner` steps because the offset depends on the current `lambda_0` estimate) and a theta-step (1-D MLE of the renewal shape on the current operational-time ISIs). `basis` (e.g. a B-spline design from `bspline_basis_1d`) overrides `covariates` as the GLM design matrix when given. → `ModulatedRenewalResult` |
| `simulate_modulated_renewal(rate_fn_or_beta, shape_param, *, T, renewal="inverse_gaussian", rng, dt=0.001)` → spike times | Time-rescaling inverse-method simulator (Ogata 1988; Brown et al. 2002), generalised from the Poisson `Exp(1)` case to a general renewal density: draw i.i.d. operational-time increments (mean 1), partial-sum them, and invert through `Lambda_0^{-1}` (linear interpolation on a dense `dt` grid) to recover real spike times. |
| `renewal_hazard(tau, shape_param, renewal="inverse_gaussian")` | Renewal hazard `r(tau; theta) = f(tau; theta) / S(tau; theta)` — the multiplicative history term, evaluated at elapsed operational time `tau`. |
| `renewal_cdf(u, shape_param, renewal="inverse_gaussian")` | Renewal CDF `F(u; theta)` — the probability-integral transform used for the goodness-of-fit tie: under the true model, `renewal_cdf(rescaled_isis, shape_param, renewal)` is Uniform(0,1) (the classical time-rescaling-theorem check of Brown et al. 2002). |
| `ModulatedRenewalResult` | Frozen dataclass: `beta`, `shape_param`, `renewal`, `rescaled_isis` (operational-time ISIs, length `len(spike_times) - 1`), `cv` (`= 1/sqrt(shape_param)`; `~1` Poisson-like, `<1` regular), `log_likelihood`, `n_iter`, `converged`, `spike_times`, `dt`. `.rate_fn()` returns a callable `t -> lambda(t | H_t)` (the fitted CIF, nearest-bin covariate lookup) — feed its output through `dt` to build per-bin probabilities for `marked_time_rescaling` / the rest of `nstat.extras.spatial.marked_gof`. |

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
- **`fit_cox_hawkes`'s length scales assume a unit-square domain.**
  `length_scale_space=0.12`/`length_scale_time=0.1` forward unchanged to
  *every* background refit; on a physically-sized domain (mm/cm) they are
  far shorter than one grid cell and produce an under-smoothed, speckled
  background. Rescale the length scales to the domain extent, or simulate
  and fit on a normalised `((0, 1), (0, 1))` domain and rescale the
  recovered rate/positions back afterward (see the ECoG-grid demo).
- **`lgcp_st_fit`'s `variance -> inf` is the Poisson-MLE limit, not
  `variance -> 0`.** A larger `variance` widens (flattens) the Matérn GP
  prior on the log-rate, so the Newton/IRLS mode drifts toward the
  unregularised per-cell Poisson MLE `log(count / cell_volume)`; a
  *smaller* `variance` pulls the fit toward the (informative) prior mean
  instead.
- **`em_spatial_hawkes`'s `K_branch_hat` IS the branching ratio,
  directly.** Unlike the static `hawkes_em.HawkesEMResult`, whose
  `branching_ratio` property divides an unnormalised amplitude `alpha` by
  a decay `beta`, the space-time triggering kernel here is individually
  normalised in both space and time, so `K` itself already equals the
  expected number of direct offspring per event — do not divide it by
  anything.
- **`global_envelope_st` needs a held-out `lambda_hat`, same caveat as
  the static `global_envelope`.** Reweighting by an intensity fit to the
  *same* pattern under test deflates the statistic's variance and shrinks
  the envelope below nominal coverage; fit `intensity_st_kde` (or
  `lgcp_st_fit`) on a disjoint calibration epoch and pass that estimate's
  `.evaluate` (or `.intensity_fn()`) as `lambda_hat` instead.

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

Bartlett wave-vector spectrum of a Hawkes triggering matrix:

```python
import numpy as np
from nstat.extras.spatial import bartlett_spectrum, detect_wave_peaks

# 4 electrodes on a unit grid; toy positive-excitation adjacency.
pos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
adj = 0.05 * np.ones((4, 4))
f = np.linspace(0.5, 5.0, 16)
kx = np.linspace(-2.0, 2.0, 9)
ky = np.linspace(-2.0, 2.0, 9)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
k = np.stack([KX.ravel(), KY.ravel()], axis=1)

S = bartlett_spectrum(adj, pos, f, k, decay=1.0)              # (Nf, Nk)
peaks = detect_wave_peaks(S, f, k, n_peaks=3)
print("speeds (pos-unit / s):", peaks.speed)
print("directions (rad):", peaks.direction)
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

Spatiotemporal: simulate a sub-critical space-time Hawkes/ETAS catalogue
and recover its parameters by branching EM (`K_branch_hat` is the
branching ratio directly — no `alpha/beta` division needed):

```python
import numpy as np
from nstat.extras.spatial import (
    SpatialHawkesSpec,
    em_spatial_hawkes,
    simulate_spatial_hawkes,
)

rng = np.random.default_rng(3)
domain = ((0.0, 1.0), (0.0, 1.0))
T = 300.0

points, times = simulate_spatial_hawkes(
    mu=0.5, K_branch=0.35, c=1.0, sigma_space=0.05,
    domain=domain, T=T, rng=rng,
)

fit = em_spatial_hawkes(
    points, times, domain=domain, T=T,
    spec=SpatialHawkesSpec(K0=0.3, c0=1.0, sigma0=0.1),
)
print(
    "K_branch_hat:", round(fit.K_branch_hat, 2),
    "| c_hat:", round(fit.c_hat, 2),
    "| converged:", fit.converged,
)
```

## Examples and notebooks

End-to-end demonstrations of the basis-projected LGCP path live in
[`examples/paper/example06_place_fields_glm_basis.py`](../../examples/paper/example06_place_fields_glm_basis.py)
with a companion walkthrough notebook
[`notebooks/PlaceFieldGLMBasis.ipynb`](../../notebooks/PlaceFieldGLMBasis.ipynb).
The example fits a synthetic place field with `BSplineBasis2D` /
`MaternPrior` / `lgcp_fit_glm`, evaluates the inhomogeneous second-order
diagnostics `pair_correlation` / `k_inhom` / `l_function` under each of
the four `edge_correction` modes, and renders the posterior rate map
with credible bands.

The spatiotemporal wave path is demonstrated by
[`examples/paper/example07_spatiotemporal_hawkes_waves.py`](../../examples/paper/example07_spatiotemporal_hawkes_waves.py)
with the companion notebook
[`notebooks/HawkesWaveAnalysis.ipynb`](../../notebooks/HawkesWaveAnalysis.ipynb).
Both scripts build a small electrode-array Hawkes triggering matrix,
compute its `bartlett_spectrum`, run `detect_wave_peaks` to surface a
`WaveAnalysisResult` (speeds and directions), and use
`reconstruct_kernel` to visualize the parametric exponential kernel
underlying the fit.

The full encoding-then-decoding loop on real spikes lives in
[`examples/paper/example08_real_place_cells.py`](../../examples/paper/example08_real_place_cells.py)
with the companion notebook
[`notebooks/RealPlaceCellDecoding.ipynb`](../../notebooks/RealPlaceCellDecoding.ipynb).
The script loads Animal 1 of the figshare paper dataset, fits a
4-cell B-spline Poisson GLM on the training half, decodes position
with the PPAF on the held-out half, and runs `pair_correlation` plus
`global_envelope` (Ripley isotropic) and the population
`rescaled_acf` (Bartlett band) as a held-out goodness-of-fit suite.
The multitype cross-correlation pipeline is exercised by
[`notebooks/MultitypeCrossK.ipynb`](../../notebooks/MultitypeCrossK.ipynb),
which contrasts an independent two-type Poisson labelling with a
Thomas-clustered shared-parent labelling under `cross_k_inhom` and
`cross_pair_correlation` (Baddeley-Moller-Waagepetersen 2000).

The cluster-Cox + Gibbs catalogue is exercised end-to-end by two
companion synthetic demo scripts and one walkthrough notebook.
[`examples/extras/spatial_cluster_cox_demo.py`](../../examples/extras/spatial_cluster_cox_demo.py)
simulates a Thomas and a Matérn-cluster process on the unit square and
recovers `(sigma, lambda_p)` / `(R, lambda_p)` via `fit_thomas` and
`fit_matern_cluster` — the minimum-contrast (Diggle 2013 §6.2.1)
estimators built on the SOIRS border-corrected pair correlation; the
mean offspring count `mu` is then recovered post-hoc from `n / (lambda_p
* |W|)` because second-order statistics do not identify it.
[`examples/extras/spatial_gibbs_demo.py`](../../examples/extras/spatial_gibbs_demo.py)
simulates a Strauss (`gamma = 0.4`), a hard-core, and an
area-interaction (`eta = 4.0`) process via the Metropolis birth-death
chain (Geyer 1999) and the dart-throwing rejection sampler, then recovers
their parameters with `pseudo_likelihood_fit` — the Berman-Turner (1992)
device routing through `nstat.glm.fit_poisson_glm`.  The hard-core demo
deliberately uses `beta = 60` to land in the numerical envelope where
the intercept-only GLM converges and the documented upward bias on
`beta_hat` (BRT-2015 §13.4) is visible.

The companion walkthrough notebook
[`notebooks/SpatialClusterAndGibbs.ipynb`](../../notebooks/SpatialClusterAndGibbs.ipynb)
runs both halves in one place — six processes, two estimators, a single
recovery table — so the attractive (cluster-Cox) and repulsive (Gibbs)
catalogues can be inspected side-by-side.

The spatiotemporal tier is exercised by four standalone demo scripts,
each grounded in a realistic electrode-array recording geometry (Utah
-style microelectrode array or ECoG surface grid) with fully synthetic
spikes — no figshare dataset access required.
[`examples/extras/spatial_stlgcp_microelectrode_demo.py`](../../examples/extras/spatial_stlgcp_microelectrode_demo.py)
simulates a Gaussian rate "bump" translating across a 10x10 Utah-style
array and recovers it with both `intensity_st_kde` and `lgcp_st_fit`
(including its posterior credible band).
[`examples/extras/spatial_gof_ecog_demo.py`](../../examples/extras/spatial_gof_ecog_demo.py)
compares a baseline (inhomogeneous-Poisson) epoch against a clustered
/traveling-wave epoch on an 8x8 ECoG grid, using a held-out
`intensity_st_kde` fit and `k_st_inhom` / `pair_correlation_st` /
`global_envelope_st` to confirm the baseline fails to reject and the
clustered epoch rejects the space-time Poisson null.
[`examples/extras/spatial_hawkes_ecog_demo.py`](../../examples/extras/spatial_hawkes_ecog_demo.py)
recovers propagating-activity kernels on the same ECoG geometry via
`em_spatial_hawkes` (homogeneous background) and `fit_cox_hawkes`
(inhomogeneous LGCP background), decomposing background from
self-excitation.
[`examples/extras/modulated_renewal_microelectrode_demo.py`](../../examples/extras/modulated_renewal_microelectrode_demo.py)
fits a single regular-spiking microelectrode unit with
`fit_modulated_renewal` (gamma renewal, CV < 1) and shows the continuous
and discrete-time-rescaling KS checks both pass, unlike a naive
Poisson-CV=1 assumption on the same rescaled ISIs.

## Scope

| Feature | Status |
|---|---|
| LGCP Laplace rate map + log-normal credible band | shipped (NumPy) |
| Basis-projected LGCP (`lgcp_fit_glm` + `MaternPrior`) | shipped (NumPy) |
| Inhomogeneous `g`/`K`/`L`, F/G/J, global-rank envelope | shipped (NumPy) |
| Discrete-time-rescaling correction + marked / multivariate KS | shipped (NumPy) |
| DPP eigen-sampler (`L`-ensemble) | shipped (NumPy fallback) + DPPy bridge |
| Multivariate Hawkes | `tick` bridge (`[hawkes]`) |
| Bartlett spectrum + wave-peak detection of a fitted Hawkes adjacency | shipped (NumPy) |
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
- Thomas M (1949). *A generalization of Poisson's binomial limit for use
  in ecology.* Biometrika 36(1/2):18.
- Matérn B (1986). *Spatial Variation* (2nd ed.). Springer Lecture
  Notes in Statistics 36.
- Neyman J, Scott EL (1958). *Statistical approach to problems of
  cosmology.* J. R. Stat. Soc. B 20(1):1.
- Møller J, Waagepetersen RP (2003). *Statistical Inference and
  Simulation for Spatial Point Processes.* Chapman & Hall.
- Diggle PJ (2013). *Statistical Analysis of Spatial and
  Spatio-Temporal Point Patterns* (3rd ed.). CRC.
- Gerhard F, Haslinger R, Pipa G (2011). *Applying the multivariate
  time-rescaling theorem to neural population models.* Neural Computation
  23(6):1452.
- Kulesza A, Taskar B (2012). *Determinantal Point Processes for Machine
  Learning.* FnT in ML 5(2-3):123.
- Bacry E, Bompaire M, Gaïffas S, Poulsen S (2018). *tick: a Python library
  for statistical learning.* JMLR 18(214):1.
- Bacry E, Mastromatteo I, Muzy J-F (2015). *Hawkes processes in finance.*
  Market Microstructure and Liquidity 1(1):1550005.
- Daley DJ, Vere-Jones D (2003). *An Introduction to the Theory of Point
  Processes*, Vol. I, §8.4 — Bartlett spectrum.
- Hansen NR, Reynaud-Bouret P, Rivoirard V (2015). *Lasso and probabilistic
  inequalities for multivariate point processes.* Bernoulli 21(1):83.
- Schlather M (2001). *On the second-order characteristics of marked
  point processes.* Bernoulli 7(1):99-117.
- Stoyan D, Stoyan H (1994). *Fractals, Random Shapes and Point Fields:
  Methods of Geometrical Statistics.* Wiley.
- Cressie N, Hawkins DM (1980). *Robust estimation of the variogram, I.*
  Journal of the IAMG 12(2):115-125.
- Illian J, Penttinen A, Stoyan H, Stoyan D (2008). *Statistical Analysis
  and Modelling of Spatial Point Patterns.* Wiley.
- Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002). *The
  time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2):325-346.
- Andersen PK (1997). *Statistical Models Based on Counting Processes.*
  Springer.
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). *A
  point process framework for relating neural spiking activity to
  spiking history, neural ensemble, and extrinsic covariate effects.*
  Journal of Neurophysiology 93(2):1074-1089.
- Bartlett MS (1964). *The spectral analysis of two-dimensional point
  processes.* Biometrika 51(3-4):299-311.
- Stein ML (1999). *Interpolation of Spatial Data: Some Theory for
  Kriging.* Springer.
- Veen A, Schoenberg FP (2008). *Estimation of space-time branching
  process models in seismology using an EM-type algorithm.* JASA
  103(482):614-624.
- Marsan D, Lengliné O (2008). *Extending earthquakes' reach through
  cascading.* Science 319(5866):1076-1079.
- Linderman SW, Adams RP, Pillow JW (2014). *Discovering latent network
  structure in point process data.* NeurIPS.
- Diggle PJ (1985). *A kernel method for smoothing point process data.*
  Journal of the Royal Statistical Society, Series C 34(2):138-147.
- Diggle PJ, Chetwynd AG, Haggkvist R, Morris SE (1995). *Second-order
  analysis of space-time clustering.* Journal of the Royal Statistical
  Society, Series C 44(1):71-86.
- Gabriel E, Diggle PJ (2009). *Second-order analysis of inhomogeneous
  spatio-temporal point process data.* Statistica Neerlandica
  63(1):43-51.
- Møller J, Ghorbani M (2012). *Aspects of second-order analysis of
  structured inhomogeneous spatio-temporal point processes.* Statistica
  Neerlandica 66(4):472-491.
- Fuentes-Santos I, González-Manteiga W, Mateu J (2018). *A
  nonparametric test for the comparison of first-order structures of
  spatial point processes.* Spatial Statistics 25:44-63.
- Ogata Y (1998). *Space-time point-process models for earthquake
  occurrences.* Annals of the Institute of Statistical Mathematics
  50(2):379-402.
- Zhuang J, Ogata Y, Vere-Jones D (2002). *Stochastic declustering of
  space-time earthquake occurrences.* Journal of the American
  Statistical Association 97(458):369-380.
- Møller J, Rasmussen JG (2005). *Perfect simulation of Hawkes
  processes.* Advances in Applied Probability 37(3):629-646.
- Saatçi Y (2011). *Scalable Inference for Structured Gaussian Process
  Models.* PhD thesis, University of Cambridge.
- Wilson AG, Nickisch H (2015). *Kernel Interpolation for Scalable
  Structured Gaussian Processes (KISS-GP).* ICML.
- Hutchinson MF (1990). *A stochastic estimator of the trace of the
  influence matrix for Laplacian smoothing splines.* Communications in
  Statistics — Simulation and Computation 19(2):433-450.
- Bekas C, Kokiopoulou E, Saad Y (2007). *An estimator for the diagonal
  of a matrix.* Applied Numerical Mathematics 57(11-12):1214-1229.
- Miscouridou X, Bhatt S, Mohler G, Flaxman S, Bhamidi S (2022).
  *Cox-Hawkes: doubly stochastic spatiotemporal Poisson processes.*
  Transactions on Machine Learning Research.
- Celeux G, Diebolt J (1985). *The SEM algorithm: a probabilistic
  teacher algorithm derived from the EM algorithm for the mixture
  problem.* Computational Statistics Quarterly 2:73-82.
- Lewis PAW, Shedler GS (1979). *Simulation of nonhomogeneous Poisson
  processes by thinning.* Naval Research Logistics Quarterly
  26(3):403-413.
- Cox DR (1955). *Some statistical methods connected with series of
  events.* Journal of the Royal Statistical Society, Series B
  17(2):129-164.
- Barbieri R, Quirk MC, Frank LM, Wilson MA, Brown EN (2001).
  *Construction and analysis of non-Poisson stimulus-response models of
  neural spiking activity.* Journal of Neuroscience Methods 105:25-37.
- Kass RE, Ventura V (2001). *A spike-train probability model.* Neural
  Computation 13:1713-1720.
- Chhikara RS, Folks JL (1974). *Estimation of the inverse Gaussian
  distribution function.* Journal of the American Statistical
  Association 69(345):250-254.
- Chhikara RS, Folks JL (1989). *The Inverse Gaussian Distribution:
  Theory, Methodology, and Applications.* Marcel Dekker.
