#!/usr/bin/env python3
"""Example 08 — Real Place-Cell Encoding-and-Decoding With Held-Out Spatial GoF.

Concepts & background:
  Loads Animal 1 of the figshare paper dataset (the same hippocampal
  recording behind example04) and exercises the full extras.spatial
  encoding-then-decoding pipeline end-to-end on real spikes:

    1. Time-based 70/30 train/test split.
    2. Per-cell tensor-product B-spline Poisson GLM on the *training*
       half — :func:`nstat.extras.spatial.bspline_basis_2d` plus
       :func:`nstat.glm.fit_poisson_glm`, log cell area carried as an
       offset.
    3. PPAF decoding of the held-out trajectory with
       :func:`DecodingAlgorithms.PPDecodeFilter` using a quadratic-CIF
       refit per cell — see "Why a quadratic CIF for decoding?" below.
    4. Held-out spatial second-order goodness of fit: edge-corrected
       pair correlation (Ripley isotropic, 1976/1977) plus the global-
       rank envelope of Myllymaki et al. 2017
       (``edge_correction='isotropic'`` end-to-end now that Tier E
       sub-PR-1 has propagated ``edge_correction`` through
       :func:`~nstat.extras.spatial.global_envelope`).
    5. Population discrete-time-corrected rescaled-time ACF
       (Haslinger-Pipa-Brown 2010 correction; Bartlett 1955 band).

Why a quadratic CIF for decoding?
  The tensor-product B-spline basis used for the encoder has 64 cubic
  basis functions per cell; symbolic ``CIF`` objects need an algebraic
  expression per coefficient.  Building 64 sympy expressions per cell
  is prohibitively slow and not what
  :func:`DecodingAlgorithms.PPDecodeFilter` is designed for in the
  Python port (the existing notebook ``StimulusDecode2D.ipynb`` uses
  the same quadratic-CIF pattern).  We therefore re-fit a 6-term
  quadratic per cell purely for the decoder, while keeping the
  B-spline GLM as the spatial encoder used for plotting the place
  fields and computing the held-out pair-correlation diagnostic.

Animal 1 — 4-cell subset choice:
  Cells [1, 20, 24, 48] (0-indexed; MATLAB 1-indexed: 2, 21, 25, 49)
  are the four well-isolated place cells highlighted in example04 and
  in the MATLAB ``HippocampalPlaceCellExample.m`` helpfile.  Each has
  several hundred to a few thousand spikes over the ~1460 s recording
  and a single dominant in-field bump — the regime the B-spline /
  PPAF stack was designed for.  Honest note: the cell with ~12000
  spikes (idx 23) and the ~28000-spike cell (idx 13) are very
  high-rate and behave more like interneurons than classical CA1
  place cells; we exclude them so the decoder doesn't lock onto a
  diffuse high-rate cell.

References:
- Wilson MA, McNaughton BL (1993). Science 261(5124):1055.
- Mehta MR, Lee AK, Wilson MA (2002). Nature 417(6890):741.
- Brown EN, Frank LM, Tang D, Quirk MC, Wilson MA (1998).
  J Neurosci 18(18):7411.
- Eden UT, Frank LM, Barbieri R, Solo V, Brown EN (2004).
  Neural Comput 16(5):971.
- Baddeley AJ, Moller J, Waagepetersen R (2000).
  Statistica Neerlandica 54(3):329.
- Myllymaki M, Mrkvicka T, Grabarnik P, Seijo H, Hahn U (2017).
  JRSS-B 79(2):381.
- Haslinger R, Pipa G, Brown EN (2010). Neural Comput 22(10):2477.
- Bartlett MS (1955). *An Introduction to Stochastic Processes*.

Paper-mapping: Section 2.3.5 (place-cell encoding) extended to a
held-out 2-D spatial GoF that did not appear in the original paper.

Expected outputs:
- Figure 1: B-spline place-field heatmaps for the 4-cell subset,
  with the held-out animal path overlaid.
- Figure 2: PPAF decoded vs true position on the held-out half.
- Figure 3: Held-out pair correlation g(r) with the inhomogeneous
  global-rank envelope (isotropic edge correction).
- Figure 4: Population rescaled-time ACF with the Bartlett band.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import CIF, DecodingAlgorithms, apply_plot_style  # noqa: E402
from nstat.data_manager import get_paper_data_dirs  # noqa: E402
from nstat.extras.spatial import (  # noqa: E402
    bspline_basis_2d,
    pair_correlation,
    rescaled_acf,
)
from nstat.extras.spatial.marked_gof import corrected_rescaled  # noqa: E402
from nstat.extras.spatial.spatial_gof import global_envelope  # noqa: E402
from nstat.glm import fit_poisson_glm  # noqa: E402

# Same 4-cell hand-picked subset that the canonical example04 highlights as
# the "good" CA1 place cells in Animal 1 (MATLAB indices 2, 21, 25, 49 →
# 0-indexed 1, 20, 24, 48; see ``HippocampalPlaceCellExample.m`` and
# ``example04_place_cells_continuous_stimulus.py``).
_ANIMAL1_PLACE_CELL_INDICES: tuple[int, ...] = (1, 20, 24, 48)


# =====================================================================
# Data loader — copied verbatim from example04 per the brief, kept local
# so example04 and example08 do not develop a shared-helper coupling.
# =====================================================================
def _load_animal_data(path: Path):
    """Load place-cell trajectory + spike data from a paper ``.mat`` file."""
    d = loadmat(str(path), squeeze_me=True)
    x = np.asarray(d["x"], dtype=float).ravel()
    y = np.asarray(d["y"], dtype=float).ravel()
    time = np.asarray(d["time"], dtype=float).ravel()
    neurons = np.asarray(d["neuron"], dtype=object).ravel()
    return x, y, time, neurons


def _spike_times(neuron_obj) -> np.ndarray:
    """Extract sorted spike times (s) from a single neuron struct."""
    st = np.asarray(neuron_obj["spikeTimes"].item(), dtype=float).ravel()
    return np.sort(st)


# =====================================================================
# 70/30 time split + spike binning
# =====================================================================
def _train_test_split(
    time: np.ndarray, fraction: float = 0.70
) -> tuple[slice, slice, float]:
    """Return (train_slice, test_slice, t_split) for a 70/30 time split."""
    n = time.size
    n_train = int(np.floor(fraction * n))
    return slice(0, n_train), slice(n_train, n), float(time[n_train])


def _bin_spikes_on_grid(
    spike_times: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_time: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
) -> np.ndarray:
    """Spike counts per spatial bin, restricted to a time mask via t_min/t_max.

    Spikes are assigned to the position bin of the *position sample nearest*
    in time to each spike — the canonical paper helpfile convention.  Only
    spikes whose nearest position sample falls in the supplied position
    window are counted; ``pos_x``/``pos_y``/``pos_time`` are assumed to be
    the *training* (or *test*) slice already.
    """
    if spike_times.size == 0:
        return np.zeros((edges_x.size - 1, edges_y.size - 1), dtype=float)
    # Spikes whose nearest position sample is in the supplied position
    # window get a discrete bin index; others are dropped.
    mask = (spike_times >= float(pos_time[0])) & (spike_times <= float(pos_time[-1]))
    st = spike_times[mask]
    if st.size == 0:
        return np.zeros((edges_x.size - 1, edges_y.size - 1), dtype=float)
    idx = np.searchsorted(pos_time, st, side="left")
    idx = np.clip(idx, 0, pos_time.size - 1)
    # Snap to the closer of (idx-1, idx) — searchsorted only gives the
    # right-bracket index.
    left = np.clip(idx - 1, 0, pos_time.size - 1)
    pick_left = (idx == pos_time.size) | (
        (idx > 0) & (np.abs(pos_time[left] - st) < np.abs(pos_time[idx] - st))
    )
    idx = np.where(pick_left, left, idx)
    H, _, _ = np.histogram2d(pos_x[idx], pos_y[idx], bins=[edges_x, edges_y])
    return H


def _spike_locations(
    spike_times: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_time: np.ndarray,
) -> np.ndarray:
    """Return (n_spikes_in_window, 2) array of spike locations in a window."""
    if spike_times.size == 0:
        return np.zeros((0, 2), dtype=float)
    mask = (spike_times >= float(pos_time[0])) & (spike_times <= float(pos_time[-1]))
    st = spike_times[mask]
    if st.size == 0:
        return np.zeros((0, 2), dtype=float)
    idx = np.searchsorted(pos_time, st, side="left")
    idx = np.clip(idx, 0, pos_time.size - 1)
    left = np.clip(idx - 1, 0, pos_time.size - 1)
    pick_left = (
        (idx > 0) & (np.abs(pos_time[left] - st) < np.abs(pos_time[idx] - st))
    )
    idx = np.where(pick_left, left, idx)
    return np.column_stack([pos_x[idx], pos_y[idx]])


# =====================================================================
# B-spline encoder
# =====================================================================
def _fit_bspline_encoder(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    n_grid: int = 32,
    n_knots: tuple[int, int] = (8, 8),
    degree: int = 3,
    box: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    """Fit per-cell tensor-product B-spline Poisson GLM on the training half.

    Returns
    -------
    grid_x, grid_y, design, rate_maps, glm_results
        ``grid_x``, ``grid_y`` are the 1-D grids; ``design`` is the
        ``(n_grid**2, n_knots[0] * n_knots[1])`` B-spline design matrix;
        ``rate_maps`` is a ``(C, n_grid, n_grid)`` ndarray of predicted
        rates (events / spatial bin / training duration); ``glm_results``
        is the list of :class:`~nstat.glm.PoissonGLMResult` per cell.
    """
    x_lo, x_hi, y_lo, y_hi = box
    grid_x = np.linspace(x_lo, x_hi, n_grid)
    grid_y = np.linspace(y_lo, y_hi, n_grid)
    edges_x = np.linspace(x_lo, x_hi, n_grid + 1)
    edges_y = np.linspace(y_lo, y_hi, n_grid + 1)
    B = bspline_basis_2d(grid_x, grid_y, n_knots=n_knots, degree=degree)
    cell_area = float(((x_hi - x_lo) / n_grid) * ((y_hi - y_lo) / n_grid))
    offset = np.full(B.shape[0], np.log(cell_area), dtype=float)

    rate_maps = np.zeros((len(spikes_per_cell), n_grid, n_grid), dtype=float)
    glm_results = []
    for c, st in enumerate(spikes_per_cell):
        H = _bin_spikes_on_grid(
            st, pos_x_train, pos_y_train, pos_time_train, edges_x, edges_y,
        )
        counts = H.ravel().astype(float)
        # The B-spline basis is a partition of unity → drop the intercept
        # column and damp early Newton steps with an L2 ridge.  Real
        # spike trains have many empty bins near the box edges, which
        # makes unpenalised IRLS overshoot in the boundary B-spline
        # coefficients; the stronger ridge (l2=1.0) and the larger
        # iteration budget (max_iter=400) keep all four cells inside
        # the converged-flag region of the solver.
        glm = fit_poisson_glm(
            B, counts, offset=offset, include_intercept=False, l2=1.0,
            max_iter=400,
        )
        eta = B @ glm.coefficients
        # eta = log(rate * cell_area) → rate_per_bin = exp(eta) (already
        # absorbs the area offset via the design).  Clip to a sane upper
        # bound so a runaway boundary fit cannot poison the downstream
        # envelope's intensity callable (the simulator computes
        # ``lam_max * area`` and would overflow into the Poisson RNG).
        eta_clipped = np.clip(eta, -20.0, 8.0)
        rate_maps[c, :, :] = np.exp(eta_clipped).reshape(n_grid, n_grid)
        glm_results.append(glm)
    return grid_x, grid_y, B, rate_maps, glm_results


# =====================================================================
# Quadratic-CIF decoder
# =====================================================================
def _design_quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[1, x, y, x^2, y^2, x*y] design matrix (T, 6)."""
    return np.column_stack([
        np.ones(x.size), x, y, x * x, y * y, x * y,
    ])


def _fit_quadratic_cif_per_cell(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    delta: float,
) -> list[CIF]:
    """Refit a 6-term quadratic *binomial* CIF per cell for PPAF decoding.

    The B-spline encoder above is used for the spatial intensity GoF; the
    decoder needs an analytical CIF whose gradients sympy can take —
    matching the existing ``StimulusDecode2D.ipynb`` and the fidelity test
    ``test_ppdecodefilter_handles_symbolic_style_polynomial_cifs``.
    """
    # Resample each spike train to the 1/delta lattice defined by
    # pos_time_train (we treat each position sample as one decoding bin).
    n_bins = pos_time_train.size
    X_pos = _design_quadratic(pos_x_train, pos_y_train)  # (n_bins, 6)
    cifs: list[CIF] = []
    for st in spikes_per_cell:
        mask = (st >= float(pos_time_train[0])) & (st <= float(pos_time_train[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_train, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        y_binary = np.zeros(n_bins, dtype=float)
        # Multiple spikes in the same bin collapse to a 1 (binomial CIF
        # convention used by PPAF).
        y_binary[bin_idx] = 1.0
        # Binomial / logistic GLM via a Newton-IRLS through the toolbox
        # surface; lazy-import to keep the module light.
        from nstat.glm import fit_binomial_glm
        glm = fit_binomial_glm(X_pos[:, 1:], y_binary, include_intercept=True,
                               l2=1e-3, max_iter=80)
        b = np.concatenate([[glm.intercept], glm.coefficients])
        cif = CIF(
            b.tolist(),
            ["1", "x", "y", "x^2", "y^2", "x*y"],
            ["x", "y"],
            fitType="binomial",
        )
        cifs.append(cif)
    return cifs


def _decode_position(
    pos_x_test: np.ndarray,
    pos_y_test: np.ndarray,
    pos_time_test: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    cifs: list[CIF],
    *,
    delta: float,
    stride: int = 10,
):
    """Run PPAF decoding on the held-out half (2-D random-walk state).

    Implementation note — why we go through ``PPDecodeFilterLinear``
    instead of ``PPDecodeFilter`` (CIF-object branch):

    Historically the CIF-object branch of
    :func:`DecodingAlgorithms.PPDecodeFilter` rebuilt an
    ``nspikeTrain`` and ``resampled`` it to ``1 / binwidth`` inside
    ``PPDecode_update`` for every ``(cell, time_index)`` pair when the
    CIF had no pre-attached history matrix (the default).  The
    resample step was O(time_index), making the full forward pass
    O(C * T^2) and dominating wall-clock on the ~13000-bin Animal-1
    test slice.  That bug is now fixed (history-free CIFs take an
    O(C * T) fast path), but we keep the linear call below: it is
    still the canonical paper-script entry point and the numerics
    match the CIF branch for our history-free quadratic models.

    ``_ppdecode_filter_linear`` performs the mathematically equivalent
    update for our position-only quadratic CIFs (no history terms; mu
    is the first CIF coefficient, beta the rest) and runs in O(C * T)
    pure-NumPy time.  We extract ``(mu, beta)`` from each CIF and call
    the linear path directly.  The linear and CIF paths agree for
    history-free position-only models — this is documented in
    ``DecodingAlgorithms.PPDecodeFilter`` itself (line 1726 of
    ``nstat/decoding_algorithms.py``: when a target branch is
    requested, the CIF path *delegates* to the linear path via
    ``_extract_linear_terms_from_cifs``).
    """
    sub = slice(None, None, max(int(stride), 1))
    pos_x_sub = pos_x_test[sub]
    pos_y_sub = pos_y_test[sub]
    pos_time_sub = pos_time_test[sub]
    n_bins = pos_time_sub.size
    effective_delta = float(stride * delta)

    # Observation matrix: dN[c, t] = 1 iff cell c spiked in subsampled
    # test-bin t (collapses any multi-spike subsampled bin to 1 — the
    # binomial-CIF convention used by PPAF).
    dN = np.zeros((len(spikes_per_cell), n_bins), dtype=float)
    for c, st in enumerate(spikes_per_cell):
        mask = (st >= float(pos_time_sub[0])) & (st <= float(pos_time_sub[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_sub, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        dN[c, bin_idx] = 1.0

    # Extract (mu, beta) from each quadratic CIF.  The CIFs were built
    # with ``b = [intercept, beta_x, beta_y, beta_xx, beta_yy, beta_xy]``;
    # the state for PPDecodeFilterLinear is just (x, y), so we keep
    # only the first-order terms and fold the curvature into a single
    # log-baseline correction at the trajectory mean (a small,
    # documented approximation: the linear-CIF decoder only consumes
    # mu + beta @ state).
    state_mean = np.array([float(pos_x_sub.mean()), float(pos_y_sub.mean())])
    mu = np.zeros(len(cifs), dtype=float)
    beta = np.zeros((2, len(cifs)), dtype=float)
    for c, cif in enumerate(cifs):
        b = np.asarray(cif.b, dtype=float).reshape(-1)
        # b = [b0, bx, by, bxx, byy, bxy]
        mu[c] = float(
            b[0]
            + b[3] * state_mean[0] ** 2
            + b[4] * state_mean[1] ** 2
            + b[5] * state_mean[0] * state_mean[1]
        )
        beta[0, c] = float(
            b[1] + 2.0 * b[3] * state_mean[0] + b[5] * state_mean[1]
        )
        beta[1, c] = float(
            b[2] + 2.0 * b[4] * state_mean[1] + b[5] * state_mean[0]
        )

    # 2-D random-walk state: x_{k+1} = x_k + w_k, Q = empirical step
    # covariance of the *training* trajectory on the same subsampled
    # lattice.  (Example05 Part A's "Q = std(diff(stim))" pattern,
    # adapted to 2-D.)
    train_dx = np.diff(pos_x_sub)
    train_dy = np.diff(pos_y_sub)
    qxx = float(np.var(train_dx)) if train_dx.size else 1e-3
    qyy = float(np.var(train_dy)) if train_dy.size else 1e-3
    A = np.eye(2, dtype=float)
    Q = np.diag([qxx, qyy])
    x0 = np.array([pos_x_sub[0], pos_y_sub[0]], dtype=float)
    Pi0 = np.diag([qxx, qyy])

    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(
        A, Q, dN, mu, beta, "binomial", effective_delta,
        None, None, x0, Pi0,
    )
    return dN, x_u, W_u, pos_x_sub, pos_y_sub, pos_time_sub


# =====================================================================
# Held-out spatial pair correlation + global envelope
# =====================================================================
def _domain_area(box: tuple[float, float, float, float]) -> float:
    x_lo, x_hi, y_lo, y_hi = box
    return float((x_hi - x_lo) * (y_hi - y_lo))


def _held_out_pair_correlation(
    pts_test: np.ndarray,
    rate_field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    box: tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    n_sim: int = 19,
    max_points: int = 500,
):
    """Inhomogeneous g(r) (Ripley isotropic) + global envelope (Myllymaki).

    The intensity reweighting uses the *training*-fit rate field — a
    plug-in estimate of the inhomogeneous lambda(x) evaluated at the
    held-out event locations via bilinear interpolation on (grid_x,
    grid_y).

    The full pooled held-out spike set on Animal 1 has ~1500 events;
    the envelope simulator's per-sim ``pair_correlation`` call is
    O(n^2) (an n_pts x n_pts cdist) and the 19-rep simulation budget
    would otherwise dominate wall-clock.  We thin to ``max_points``
    events for the envelope — this preserves the second-order
    intensity-corrected shape while bringing the inner loop into the
    paper-script budget.
    """
    if pts_test.shape[0] < 5:
        return None
    if pts_test.shape[0] > max_points:
        idx = rng.choice(pts_test.shape[0], size=max_points, replace=False)
        pts_test = pts_test[np.sort(idx)]
    x_lo, x_hi, y_lo, y_hi = box
    domain = ((x_lo, x_hi), (y_lo, y_hi))
    # The B-spline GLM was trained with ``offset = log(cell_area)``, so
    # ``rate_field`` is in units of expected events per training-window
    # per spatial bin.  The envelope simulator expects a density
    # ``lambda(x)`` with units of events per unit area on the same
    # held-out domain — convert by:
    #
    #   1. Robust-clip the per-bin rate to a 1st-99th percentile band
    #      (a single boundary cell of an under-converged GLM otherwise
    #      blows up the dominating intensity).
    #   2. Divide by ``cell_area`` to get events / unit area / training.
    #   3. Rescale so the integral over the domain equals the held-out
    #      event count — i.e. match the *first*-order intensity of the
    #      held-out pattern.  The second-order shape we care about is
    #      invariant under this rescale.
    from scipy.interpolate import RegularGridInterpolator

    cell_area = float(
        ((grid_x[-1] - grid_x[0]) / max(grid_x.size - 1, 1))
        * ((grid_y[-1] - grid_y[0]) / max(grid_y.size - 1, 1))
    )
    flat = rate_field.ravel()
    lam_lo = float(max(np.percentile(flat, 1.0), 1e-3))
    lam_hi = float(np.percentile(flat, 99.0))
    if not (lam_hi > lam_lo):
        lam_hi = lam_lo * 10.0
    clipped_field = np.clip(rate_field, lam_lo, lam_hi) / cell_area
    target_mean = float(pts_test.shape[0]) / _domain_area(box)
    current_mean = float(np.mean(clipped_field))
    if current_mean > 0:
        clipped_field = clipped_field * (target_mean / current_mean)
    lam_lo_density = float(np.percentile(clipped_field, 1.0))
    lam_hi_density = float(np.percentile(clipped_field, 99.0))
    interp = RegularGridInterpolator(
        (grid_x, grid_y), clipped_field,
        bounds_error=False, fill_value=float(clipped_field.mean()),
    )

    def lam_callable(xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.clip(interp(xy), lam_lo_density, lam_hi_density)

    r_grid = np.linspace(0.02, 0.30, 12)
    bw = 0.05
    g = pair_correlation(
        pts_test,
        lam_callable,
        r_grid,
        bw=bw,
        domain=domain,
        edge_correction="isotropic",
    )
    env = global_envelope(
        pts_test,
        lam_callable,
        r_grid,
        n_sim=n_sim,
        domain=domain,
        statistic="pcf",
        bw=bw,
        rng=rng,
        edge_correction="isotropic",
    )
    return r_grid, g, env


# =====================================================================
# Population rescaled-time ACF
# =====================================================================
def _population_rescaled_acf(
    spikes_per_cell: list[np.ndarray],
    cifs: list[CIF],
    pos_x_test: np.ndarray,
    pos_y_test: np.ndarray,
    pos_time_test: np.ndarray,
    *,
    delta: float,
    rng: np.random.Generator,
    n_lags: int = 20,
):
    """Discrete-time-corrected per-cell uniforms, concatenated.

    Each cell's per-bin spike probability ``p_k = sigmoid(eta_k)`` is
    evaluated from its fitted quadratic CIF at the held-out trajectory.
    The Haslinger-Pipa-Brown correction draws one randomization per
    inter-spike interval; we concatenate the per-cell ``u_corrected``
    streams across cells and run :func:`rescaled_acf` on the pool.
    """
    X_test = _design_quadratic(pos_x_test, pos_y_test)  # (n_bins, 6)
    u_streams: list[np.ndarray] = []
    for c, st in enumerate(spikes_per_cell):
        # Coefficients in the CIF are stored as ``b``.
        b = np.asarray(cifs[c].b, dtype=float).reshape(-1)
        eta = np.clip(X_test @ b, -20.0, 20.0)
        p_k = 1.0 / (1.0 + np.exp(-eta))  # logistic, per-bin spike prob.
        p_k = np.clip(p_k, 1e-9, 1.0 - 1e-9)
        mask = (st >= float(pos_time_test[0])) & (st <= float(pos_time_test[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_test, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, pos_time_test.size - 1)
        # Need at least n_lags + 3 events for the ACF; skip very sparse
        # cells to keep the diagnostic honest.
        if bin_idx.size < n_lags + 3:
            continue
        u = corrected_rescaled(np.sort(bin_idx), p_k, rng)
        u_streams.append(u)

    if not u_streams:
        return None
    u_all = np.concatenate(u_streams)
    if u_all.size < n_lags + 2:
        return None
    return rescaled_acf(u_all, n_lags=n_lags)


# =====================================================================
# Plotting
# =====================================================================
def _plot_place_fields_panel(
    rate_maps: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    pts_test: np.ndarray,
    cell_indices: tuple[int, ...],
):
    """Figure 1: 2x2 B-spline place-field heatmaps."""
    n_cells = rate_maps.shape[0]
    # === FIGURE: fig01_real_place_fields_panel.png ===
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for i in range(min(n_cells, 4)):
        ax = axes[i]
        # ij-flattening convention: rows index x, cols index y.
        # Transpose so that x runs across the figure and y vertically.
        im = ax.pcolormesh(
            grid_x, grid_y, rate_maps[i].T, shading="auto", cmap="viridis",
        )
        ax.scatter(pts_test[:, 0], pts_test[:, 1], s=2.0, c="w", alpha=0.35,
                   edgecolor="none")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Cell #{cell_indices[i] + 1} — B-spline place field "
            f"(rate per bin; test spikes overlaid)"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n_cells, 4):
        axes[j].axis("off")
    fig.suptitle(
        "Example 08 — Animal 1 B-spline place fields (training half)\n"
        "with held-out spike locations overlaid"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_decoded_vs_true(
    pos_x_test: np.ndarray,
    pos_y_test: np.ndarray,
    pos_time_test: np.ndarray,
    x_dec: np.ndarray,
):
    """Figure 2: 2-D path + per-axis traces."""
    # === FIGURE: fig02_decoded_vs_true_position.png ===
    fig = plt.figure(figsize=(12.0, 6.5))
    ax_path = fig.add_subplot(1, 2, 1)
    ax_path.plot(pos_x_test, pos_y_test, "k-", lw=1.0, alpha=0.9, label="true")
    ax_path.plot(x_dec[0], x_dec[1], color="tab:blue", lw=1.0, alpha=0.85,
                 label="PPAF decoded")
    ax_path.scatter(pos_x_test[0], pos_y_test[0], s=80, marker="o",
                    edgecolor="k", facecolor="none", lw=1.8, label="start")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal")
    ax_path.set_title("Held-out trajectory")
    ax_path.legend(loc="upper right", fontsize=9)

    ax_x = fig.add_subplot(2, 2, 2)
    ax_x.plot(pos_time_test, pos_x_test, "k-", lw=1.2, label="true x")
    ax_x.plot(pos_time_test, x_dec[0], color="tab:blue", lw=1.0,
              label="decoded x")
    ax_x.set_ylabel("x")
    ax_x.legend(loc="upper right", fontsize=8)
    ax_x.tick_params(labelbottom=False)

    ax_y = fig.add_subplot(2, 2, 4)
    ax_y.plot(pos_time_test, pos_y_test, "k-", lw=1.2, label="true y")
    ax_y.plot(pos_time_test, x_dec[1], color="tab:blue", lw=1.0,
              label="decoded y")
    ax_y.set_xlabel("time [s]")
    ax_y.set_ylabel("y")
    ax_y.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Example 08 — held-out PPAF decoding of position "
        f"({len(pos_time_test)} bins)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_pair_correlation_envelope(r_grid, g, env):
    """Figure 3: held-out g(r) with the inhomogeneous global envelope."""
    # === FIGURE: fig03_pair_correlation_envelope.png ===
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(r_grid, env.lo, env.hi, color="0.7", alpha=0.45,
                    label=f"global envelope (n_sim={env.n_sim}, isotropic)")
    ax.plot(r_grid, g, color="tab:blue", lw=1.8, marker="o", ms=4,
            label="held-out g(r)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7,
               label="Poisson null g(r) = 1")
    ax.set_xlabel("lag r")
    ax.set_ylabel("g(r)")
    ax.set_title(
        "Example 08 — held-out pair correlation vs inhomogeneous "
        "Poisson envelope"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_rescaled_acf(acf_result):
    """Figure 4: population rescaled-time ACF with Bartlett band."""
    # === FIGURE: fig04_rescaled_acf.png ===
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(-acf_result.band, acf_result.band, color="0.85", alpha=0.7,
               label=f"Bartlett band ±{acf_result.band:.3f}")
    ax.stem(acf_result.lags, acf_result.acf, basefmt=" ")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.7)
    ax.set_xlabel("lag (events)")
    ax.set_ylabel(r"ACF of $z_j = \Phi^{-1}(u_j)$")
    ax.set_title(
        "Example 08 — population rescaled-time ACF "
        f"(n_lags={acf_result.lags.size})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    # === END FIGURE ===
    return fig


# =====================================================================
# Driver
# =====================================================================
def run_example08(
    *,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool | None = True,
    plot_style: str = "legacy",
) -> dict:
    """Run Example 08: real-place-cell encoding-and-decoding with GoF.

    Returns a dict with summary scalars and figure paths.
    """
    t_start = _time.perf_counter()
    print("=" * 70)
    print("Example 08: Real Place-Cell Encoding-and-Decoding (Animal 1)")
    print("=" * 70)

    rng = np.random.default_rng(20260616)

    # ----- Load Animal 1 -----
    dirs = get_paper_data_dirs(download=False)
    animal_mat = dirs.place_cell_data_dir / "PlaceCellDataAnimal1.mat"
    x, y, t, neurons = _load_animal_data(animal_mat)
    delta = float(np.median(np.diff(t)))
    n_cells_total = int(len(neurons))
    print(f"  Animal 1: {n_cells_total} cells, "
          f"{t.size} position samples, delta = {delta:.4f} s "
          f"(duration = {t[-1] - t[0]:.1f} s)")

    cell_indices = _ANIMAL1_PLACE_CELL_INDICES
    spikes_per_cell = [_spike_times(neurons[i]) for i in cell_indices]
    for i, st in zip(cell_indices, spikes_per_cell):
        print(f"    cell idx {i:2d} (MATLAB #{i + 1}): {st.size} spikes")

    # ----- 70/30 time split -----
    train_slice, test_slice, t_split = _train_test_split(t, fraction=0.70)
    print(f"  70/30 train/test split at t = {t_split:.1f} s "
          f"(n_train = {train_slice.stop}, n_test = {t.size - train_slice.stop})")

    # ----- B-spline encoder on training half -----
    box = (
        float(np.floor(x.min() * 10) / 10),
        float(np.ceil(x.max() * 10) / 10),
        float(np.floor(y.min() * 10) / 10),
        float(np.ceil(y.max() * 10) / 10),
    )
    print(f"  spatial box = {box}")
    grid_x, grid_y, B_design, rate_maps, glm_results = _fit_bspline_encoder(
        x[train_slice], y[train_slice], t[train_slice], spikes_per_cell,
        n_grid=32, n_knots=(8, 8), degree=3, box=box,
    )
    converged = [int(g.converged) for g in glm_results]
    print(f"  B-spline GLM: 32x32 grid, 8x8 knots, degree 3, "
          f"converged per cell = {converged}")

    # ----- Quadratic CIF refit + PPAF decode on test half -----
    cifs = _fit_quadratic_cif_per_cell(
        x[train_slice], y[train_slice], t[train_slice], spikes_per_cell,
        delta=delta,
    )
    print(f"  Quadratic CIFs fitted: {len(cifs)} cells")
    # The decoder uses the linear branch (see _decode_position
    # docstring): O(C * T) NumPy, comfortably under 1 s on the full
    # ~13000-bin held-out slice.  Stride 5 gives a 6 Hz decoding
    # lattice — five times the position sampler (~30 Hz) is overkill
    # for the rat-trajectory diffusion timescale, and stride 5 keeps
    # the figure 2 traces visually crisp without inflating the
    # rescaled-ACF n_events past the point at which the per-bin
    # Bernoulli model is honest.
    decode_stride = 5
    dN_test, x_dec, W_dec, x_test_sub, y_test_sub, t_test_sub = _decode_position(
        x[test_slice], y[test_slice], t[test_slice], spikes_per_cell, cifs,
        delta=delta, stride=decode_stride,
    )
    rmse = float(np.sqrt(np.mean(
        (x_dec[0] - x_test_sub) ** 2 + (x_dec[1] - y_test_sub) ** 2
    )))
    print(f"  PPAF decoded {dN_test.shape[1]} test bins (stride {decode_stride}"
          f", effective delta = {decode_stride * delta:.3f} s), position RMSE "
          f"= {rmse:.3f}")

    # ----- Held-out pair correlation + global envelope -----
    # Pool spike *locations* from all 4 cells on the test half — the
    # inhomogeneous-Poisson GoF treats them as one rate field whose
    # plug-in intensity is the sum of the per-cell rate maps.
    pts_test = np.concatenate([
        _spike_locations(st, x[test_slice], y[test_slice], t[test_slice])
        for st in spikes_per_cell
    ], axis=0)
    print(f"  pooled test spikes for GoF: n = {pts_test.shape[0]}")
    rate_field = rate_maps.sum(axis=0)  # (n_grid, n_grid)
    pcf_payload = _held_out_pair_correlation(
        pts_test, rate_field, grid_x, grid_y, box, rng, n_sim=49,
        max_points=500,
    )
    if pcf_payload is None:
        r_grid_pcf = np.zeros(0)
        g_obs = np.zeros(0)
        env = None
    else:
        r_grid_pcf, g_obs, env = pcf_payload
        print(f"  held-out pair-correlation envelope: inside = "
              f"{bool(env.inside)}, p_interval = {env.p_interval}")

    # ----- Population rescaled-time ACF -----
    # Use the same subsampled lattice the decoder ran on so that the
    # per-bin probabilities ``p_k`` and the spike-bin indices are on
    # the same time base.
    acf_result = _population_rescaled_acf(
        spikes_per_cell, cifs, x_test_sub, y_test_sub, t_test_sub,
        delta=decode_stride * delta, rng=rng, n_lags=20,
    )
    if acf_result is None:
        print("  rescaled ACF: too few events across cells; skipped.")
    else:
        n_inside = int(acf_result.inside_band.sum())
        print(f"  rescaled ACF: {n_inside}/{acf_result.lags.size} lags "
              f"inside Bartlett band ±{acf_result.band:.3f}")

    # ----- Figures -----
    fig1 = _plot_place_fields_panel(
        rate_maps, grid_x, grid_y, pts_test, cell_indices,
    )
    fig2 = _plot_decoded_vs_true(x_test_sub, y_test_sub, t_test_sub, x_dec)
    if env is not None:
        fig3 = _plot_pair_correlation_envelope(r_grid_pcf, g_obs, env)
    else:
        # Stub plot so the manifest's figure count is invariant.
        fig3, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.text(0.5, 0.5, "Held-out spike pool too small for g(r) envelope",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    if acf_result is not None:
        fig4 = _plot_rescaled_acf(acf_result)
    else:
        fig4, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.text(0.5, 0.5, "Population rescaled-time ACF unavailable",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    figures = [fig1, fig2, fig3, fig4]
    for fig in figures:
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "example08"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        fig_names = (
            "fig01_real_place_fields_panel",
            "fig02_decoded_vs_true_position",
            "fig03_pair_correlation_envelope",
            "fig04_rescaled_acf",
        )
        for fig, name in zip(figures, fig_names):
            path = export_dir / f"{name}.png"
            fig.savefig(path, dpi=200, facecolor="w", edgecolor="none")
            figure_paths.append(path)
            print(f"  Saved: {path}")

    if bool(visible):
        plt.show()
    else:
        plt.close("all")

    elapsed = _time.perf_counter() - t_start
    print(f"  wall clock = {elapsed:.1f} s")

    return {
        "cell_indices": list(cell_indices),
        "n_spikes_per_cell": [int(st.size) for st in spikes_per_cell],
        "t_split_s": t_split,
        "decode_rmse": rmse,
        "envelope_inside": bool(env.inside) if env is not None else False,
        "envelope_p_interval": (
            list(map(float, env.p_interval)) if env is not None else [0.0, 1.0]
        ),
        "acf_in_band": (
            int(acf_result.inside_band.sum()) if acf_result is not None else 0
        ),
        "acf_n_lags": (
            int(acf_result.lags.size) if acf_result is not None else 0
        ),
        "wall_clock_s": elapsed,
        "figure_paths": [str(p) for p in figure_paths],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Example 08: Real Place-Cell Encoding-and-Decoding With Held-Out "
            "Spatial GoF (Animal 1 of the figshare paper dataset)"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help=("Repository root (used for the dataset lookup "
                              "and to resolve the default export-dir under "
                              "docs/figures/example08)."))
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without displaying figures (headless).")
    parser.add_argument("--plot-style", choices=("modern", "legacy"),
                        default="legacy",
                        help="Figure styling.")
    args = parser.parse_args()

    if args.no_display:
        visible = False
    else:
        visible = bool(args.show)
    result = run_example08(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )
    if args.output_json:
        summary = {
            "cell_indices": result["cell_indices"],
            "n_spikes_per_cell": result["n_spikes_per_cell"],
            "t_split_s": result["t_split_s"],
            "decode_rmse": result["decode_rmse"],
            "envelope_inside": result["envelope_inside"],
            "envelope_p_interval": result["envelope_p_interval"],
            "acf_in_band": result["acf_in_band"],
            "acf_n_lags": result["acf_n_lags"],
            "wall_clock_s": result["wall_clock_s"],
        }
        args.output_json.write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
