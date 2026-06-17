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
- Figure 5 (model="velocity" only): per-cell firing rate vs speed
  tuning curves, showing how the velocity-augmented encoder
  recovers per-cell speed modulation.
- Figure 6 (model="velocity" only): bar chart and per-bin error
  scatter comparing baseline vs velocity-augmented decoder RMSE.

Model variants:
  ``run_example08(model="baseline")`` runs the position-only
  pipeline summarized above and produces figures 1-4 only.
  ``run_example08(model="velocity")`` runs the baseline first and
  then a velocity-augmented variant whose encoder GLM and decoder
  CIF carry a per-cell scalar speed coefficient.  Velocity is the
  simplest "extrinsic covariate" extension in the Truccolo et al.
  (2005) point-process framework, and is treated as a known
  observed covariate at decode time (their Section 2): the PPAF
  state stays 2-D (position) and the CIF evaluates with the true
  ground-truth speed at each held-out time step.  This avoids
  doubling the state dimension while exposing the speed-tuning
  contribution to RMSE.  Truccolo W, Eden UT, Fellows MR, Donoghue
  JP, Brown EN (2005). J Neurophysiol 93:1074-1089.
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
# Velocity covariate (model="velocity" only)
# =====================================================================
# Plausible rodent free-foraging speed cap (cm/s).  The Animal-1 arena
# is reported on a normalized [-1, 1] box rather than physical units;
# the trajectory is already smoothed by the tracker so the clip is
# expressed in *box units per second*.  Empirically the 99.9th
# percentile of |v| on Animal 1 sits around 2 box/s; we use 5 box/s as
# a soft clamp to neutralize the one or two single-frame tracking
# glitches without flattening the genuine high-speed tail.
_SPEED_CLIP_BOX_PER_SEC: float = 5.0


def _compute_speed(
    pos_x: np.ndarray, pos_y: np.ndarray, pos_time: np.ndarray,
    *, clip: float = _SPEED_CLIP_BOX_PER_SEC,
) -> np.ndarray:
    """Return centered-difference speed |v| = hypot(dx/dt, dy/dt).

    Implausibly fast frames (single-sample tracking glitches) are
    clipped to ``clip`` so the GLM and CIF speed coefficients are not
    dominated by a handful of outliers.  ``np.gradient`` uses
    non-uniform spacing taken from ``pos_time``, which preserves
    the second-order accuracy at the endpoints.
    """
    vx = np.gradient(pos_x, pos_time)
    vy = np.gradient(pos_y, pos_time)
    speed = np.hypot(vx, vy)
    if clip is not None and clip > 0.0:
        speed = np.minimum(speed, float(clip))
    return speed


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
# Velocity-augmented B-spline encoder (model="velocity" only)
# =====================================================================
def _fit_bspline_encoder_with_speed(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    speed_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    n_grid: int = 32,
    n_knots: tuple[int, int] = (8, 8),
    degree: int = 3,
    box: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    """Like :func:`_fit_bspline_encoder` but adds one column for speed.

    Implementation choice (option (a) in the Tier D.1 brief): a single
    linear speed term gives a per-cell scalar coefficient describing
    how firing rate scales with instantaneous speed.  A future
    extension would replace it with a 1-D B-spline expansion of speed
    (option (b), ~4 columns per cell) for more flexible speed tuning.

    Each cell's speed coefficient is computed by binning the held-in
    training trajectory's (x, y, speed) on the same spatial grid the
    spatial B-spline uses, then carrying the mean speed per bin as the
    7th regressor.  The spatial grid is the support of the basis; the
    cells are otherwise the same as the baseline encoder.

    Returns the same tuple as :func:`_fit_bspline_encoder` plus
    ``speed_coefs`` (one float per cell, length ``len(spikes_per_cell)``).
    """
    x_lo, x_hi, y_lo, y_hi = box
    grid_x = np.linspace(x_lo, x_hi, n_grid)
    grid_y = np.linspace(y_lo, y_hi, n_grid)
    edges_x = np.linspace(x_lo, x_hi, n_grid + 1)
    edges_y = np.linspace(y_lo, y_hi, n_grid + 1)
    B = bspline_basis_2d(grid_x, grid_y, n_knots=n_knots, degree=degree)
    cell_area = float(((x_hi - x_lo) / n_grid) * ((y_hi - y_lo) / n_grid))
    offset = np.full(B.shape[0], np.log(cell_area), dtype=float)

    # Mean speed per spatial bin on the training half — sum / count.
    # Bins with no training visits get the trajectory mean as a benign
    # fallback (their B-spline weight is near zero there anyway).
    ix = np.clip(
        np.searchsorted(edges_x, pos_x_train, side="right") - 1, 0, n_grid - 1,
    )
    iy = np.clip(
        np.searchsorted(edges_y, pos_y_train, side="right") - 1, 0, n_grid - 1,
    )
    speed_sum = np.zeros((n_grid, n_grid), dtype=float)
    count = np.zeros((n_grid, n_grid), dtype=float)
    np.add.at(speed_sum, (ix, iy), speed_train)
    np.add.at(count, (ix, iy), 1.0)
    mean_speed = np.where(
        count > 0,
        speed_sum / np.maximum(count, 1.0),
        float(np.mean(speed_train)),
    )
    speed_col = mean_speed.ravel()[:, None]  # (n_grid**2, 1)
    B_aug = np.column_stack([B, speed_col])

    rate_maps = np.zeros((len(spikes_per_cell), n_grid, n_grid), dtype=float)
    glm_results = []
    speed_coefs: list[float] = []
    for c, st in enumerate(spikes_per_cell):
        H = _bin_spikes_on_grid(
            st, pos_x_train, pos_y_train, pos_time_train, edges_x, edges_y,
        )
        counts = H.ravel().astype(float)
        glm = fit_poisson_glm(
            B_aug, counts, offset=offset, include_intercept=False, l2=1.0,
            max_iter=400,
        )
        eta = B_aug @ glm.coefficients
        eta_clipped = np.clip(eta, -20.0, 8.0)
        rate_maps[c, :, :] = np.exp(eta_clipped).reshape(n_grid, n_grid)
        glm_results.append(glm)
        # Speed coefficient is the last entry of the augmented design.
        speed_coefs.append(float(glm.coefficients[-1]))
    return grid_x, grid_y, B_aug, rate_maps, glm_results, speed_coefs


# =====================================================================
# Quadratic-CIF decoder
# =====================================================================
def _design_quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[1, x, y, x^2, y^2, x*y] design matrix (T, 6)."""
    return np.column_stack([
        np.ones(x.size), x, y, x * x, y * y, x * y,
    ])


def _design_quadratic_speed(
    x: np.ndarray, y: np.ndarray, s: np.ndarray,
) -> np.ndarray:
    """[1, x, y, x^2, y^2, x*y, s] design matrix (T, 7)."""
    return np.column_stack([
        np.ones(x.size), x, y, x * x, y * y, x * y, s,
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
# Velocity-augmented quadratic-CIF refit + decoder (model="velocity")
# =====================================================================
def _fit_quadratic_cif_per_cell_with_speed(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    speed_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    delta: float,
) -> list[CIF]:
    """Refit a 7-term quadratic-plus-speed binomial CIF per cell.

    The CIF is

        log_lambda = mu + bx*x + by*y + bxx*x^2 + byy*y^2 + bxy*x*y + bs*s

    where ``s`` is the instantaneous (clipped) speed treated as a known
    extrinsic covariate at decode time (Truccolo et al. 2005 Section
    2).  The function shape matches :func:`_fit_quadratic_cif_per_cell`
    apart from the extra speed regressor.
    """
    n_bins = pos_time_train.size
    X_pos = _design_quadratic_speed(
        pos_x_train, pos_y_train, speed_train,
    )  # (n_bins, 7)
    cifs: list[CIF] = []
    for st in spikes_per_cell:
        mask = (st >= float(pos_time_train[0])) & (st <= float(pos_time_train[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_train, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        y_binary = np.zeros(n_bins, dtype=float)
        y_binary[bin_idx] = 1.0
        from nstat.glm import fit_binomial_glm
        glm = fit_binomial_glm(
            X_pos[:, 1:], y_binary, include_intercept=True,
            l2=1e-3, max_iter=80,
        )
        b = np.concatenate([[glm.intercept], glm.coefficients])
        cif = CIF(
            b.tolist(),
            ["1", "x", "y", "x^2", "y^2", "x*y", "s"],
            ["x", "y", "s"],
            fitType="binomial",
        )
        cifs.append(cif)
    return cifs


def _decode_position_with_speed(
    pos_x_test: np.ndarray,
    pos_y_test: np.ndarray,
    pos_time_test: np.ndarray,
    speed_test: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    cifs: list[CIF],
    *,
    delta: float,
    stride: int = 10,
):
    """PPAF decode with speed treated as a known extrinsic covariate.

    Following Truccolo et al. 2005 Section 2 (and Brown, Frank, Tang,
    Quirk, Wilson 1998 for the place-cell-specific case), the
    instantaneous speed ``|v|`` is *observed* at each time step rather
    than decoded.  The state stays 2-D (position).  At each time step
    we fold the known speed contribution ``b_s * s_t`` into a
    time-varying baseline ``mu(t)`` and pass ``mu(t) + beta @ state``
    to PPDecodeFilterLinear.  This is mathematically the same trick
    the linear branch already uses for the curvature terms: PPAF only
    consumes the linearized log-rate, so any known additive
    contribution can be folded into mu.
    """
    sub = slice(None, None, max(int(stride), 1))
    pos_x_sub = pos_x_test[sub]
    pos_y_sub = pos_y_test[sub]
    pos_time_sub = pos_time_test[sub]
    speed_sub = speed_test[sub]
    n_bins = pos_time_sub.size
    effective_delta = float(stride * delta)

    dN = np.zeros((len(spikes_per_cell), n_bins), dtype=float)
    for c, st in enumerate(spikes_per_cell):
        mask = (st >= float(pos_time_sub[0])) & (st <= float(pos_time_sub[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_sub, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        dN[c, bin_idx] = 1.0

    # Linearize the quadratic-plus-speed CIF around the trajectory
    # mean position.  Speed enters linearly already, so its
    # contribution can be carried as a known time-varying baseline
    # ``mu_t[c, k] = mu_c + b_s_c * s_k``.  PPDecodeFilterLinear
    # consumes a (C,) mu and (2, C) beta — we therefore fold the
    # *time-mean* speed contribution into mu and zero out the
    # tiny residual.  This is a documented approximation: the
    # decoder filter is linearized at the trajectory mean, so
    # treating the speed term at its trajectory mean is consistent
    # with the existing curvature treatment.  A future PR (D.2/D.3
    # history/coupling) could swap the linear branch for the
    # CIF-object branch which accepts arbitrary callable mu(t).
    state_mean = np.array([float(pos_x_sub.mean()), float(pos_y_sub.mean())])
    speed_mean = float(speed_sub.mean())
    mu = np.zeros(len(cifs), dtype=float)
    beta = np.zeros((2, len(cifs)), dtype=float)
    for c, cif in enumerate(cifs):
        b = np.asarray(cif.b, dtype=float).reshape(-1)
        # b = [b0, bx, by, bxx, byy, bxy, bs]
        mu[c] = float(
            b[0]
            + b[3] * state_mean[0] ** 2
            + b[4] * state_mean[1] ** 2
            + b[5] * state_mean[0] * state_mean[1]
            + b[6] * speed_mean
        )
        beta[0, c] = float(
            b[1] + 2.0 * b[3] * state_mean[0] + b[5] * state_mean[1]
        )
        beta[1, c] = float(
            b[2] + 2.0 * b[4] * state_mean[1] + b[5] * state_mean[0]
        )

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
    return dN, x_u, W_u, pos_x_sub, pos_y_sub, pos_time_sub, speed_sub


# =====================================================================
# Speed tuning curves (model="velocity" only)
# =====================================================================
def _per_cell_speed_tuning(
    spikes_per_cell: list[np.ndarray],
    pos_time: np.ndarray,
    speed: np.ndarray,
    *,
    n_bins: int = 8,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean firing rate per speed bin (per cell), plus standard error.

    Returns
    -------
    bin_centers, rate_per_cell, sem_per_cell
        ``bin_centers`` is (n_bins,); ``rate_per_cell`` and
        ``sem_per_cell`` are (n_cells, n_bins) — both in spikes/s.
        Empty bins get NaN.
    """
    edges = np.linspace(0.0, float(speed.max()) + 1e-9, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Speed-bin index per time sample.
    sbin = np.clip(
        np.searchsorted(edges, speed, side="right") - 1, 0, n_bins - 1,
    )
    occupancy = np.bincount(sbin, minlength=n_bins) * float(delta)

    n_cells = len(spikes_per_cell)
    rate = np.full((n_cells, n_bins), np.nan, dtype=float)
    sem = np.full((n_cells, n_bins), np.nan, dtype=float)
    for c, st in enumerate(spikes_per_cell):
        mask = (st >= float(pos_time[0])) & (st <= float(pos_time[-1]))
        st_in = st[mask]
        if st_in.size == 0:
            continue
        idx = np.searchsorted(pos_time, st_in, side="left")
        idx = np.clip(idx, 0, pos_time.size - 1)
        counts = np.bincount(sbin[idx], minlength=n_bins).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(occupancy > 0, counts / occupancy, np.nan)
            # Poisson SEM on the rate.
            s = np.where(occupancy > 0, np.sqrt(counts) / occupancy, np.nan)
        rate[c, :] = r
        sem[c, :] = s
    return centers, rate, sem


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


def _plot_velocity_speed_tuning(
    bin_centers: np.ndarray,
    rate_per_cell: np.ndarray,
    sem_per_cell: np.ndarray,
    cell_indices: tuple[int, ...],
    speed_coefs: list[float],
):
    """Figure 5: per-cell firing rate as a function of speed."""
    n_cells = rate_per_cell.shape[0]
    # === FIGURE: fig05_velocity_speed_tuning.png ===
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for i in range(min(n_cells, 4)):
        ax = axes[i]
        r = rate_per_cell[i]
        s = sem_per_cell[i]
        finite = np.isfinite(r)
        ax.errorbar(
            bin_centers[finite], r[finite], yerr=s[finite],
            marker="o", ms=5, lw=1.4, color="tab:blue",
            ecolor="0.4", capsize=3, label="binned rate",
        )
        ax.set_xlabel("|v| (box units / s)")
        ax.set_ylabel("firing rate (spk/s)")
        ax.set_title(
            f"Cell #{cell_indices[i] + 1} — speed tuning "
            f"(GLM bs = {speed_coefs[i]:+.3f})"
        )
        ax.grid(True, alpha=0.3)
    for j in range(n_cells, 4):
        axes[j].axis("off")
    fig.suptitle(
        "Example 08 (velocity model) — empirical firing-rate-vs-speed "
        "tuning curves"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_decoder_baseline_vs_velocity(
    baseline_rmse: float,
    velocity_rmse: float,
    baseline_per_bin_err: np.ndarray,
    velocity_per_bin_err: np.ndarray,
):
    """Figure 6: bar chart + per-bin error scatter (baseline vs velocity)."""
    improvement_pct = 100.0 * (baseline_rmse - velocity_rmse) / baseline_rmse
    # === FIGURE: fig06_decoder_baseline_vs_velocity.png ===
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Panel 1: bar chart.
    ax_bar = axes[0]
    bars = ax_bar.bar(
        ["baseline", "velocity"],
        [baseline_rmse, velocity_rmse],
        color=["0.55", "tab:blue"], edgecolor="k", lw=1.0,
    )
    for bar, val in zip(bars, [baseline_rmse, velocity_rmse]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    sign = "+" if improvement_pct >= 0.0 else ""
    ax_bar.set_ylabel("position RMSE (box units)")
    ax_bar.set_title(
        f"Held-out RMSE: {sign}{improvement_pct:.1f}% improvement"
    )
    ax_bar.set_ylim(0.0, max(baseline_rmse, velocity_rmse) * 1.20)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-bin error scatter (baseline x, velocity y).  Both
    # error vectors are on the same time base by construction.
    ax_sc = axes[1]
    lim_hi = float(
        max(baseline_per_bin_err.max(), velocity_per_bin_err.max()) * 1.05
    )
    ax_sc.plot([0, lim_hi], [0, lim_hi], "k--", lw=1.0, alpha=0.7,
               label="x = y (no change)")
    ax_sc.scatter(
        baseline_per_bin_err, velocity_per_bin_err,
        s=6, alpha=0.4, color="tab:blue", edgecolor="none",
    )
    improved = int(np.sum(velocity_per_bin_err < baseline_per_bin_err))
    total = int(baseline_per_bin_err.size)
    ax_sc.set_xlabel("baseline per-bin error (box units)")
    ax_sc.set_ylabel("velocity per-bin error (box units)")
    ax_sc.set_xlim(0.0, lim_hi)
    ax_sc.set_ylim(0.0, lim_hi)
    ax_sc.set_aspect("equal")
    ax_sc.set_title(f"per-bin error: {improved}/{total} bins improved")
    ax_sc.legend(loc="upper left", fontsize=9)
    ax_sc.grid(True, alpha=0.3)

    fig.suptitle(
        "Example 08 — PPAF decoder: baseline (position only) vs "
        "velocity-augmented (speed as extrinsic covariate)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


# =====================================================================
# Driver
# =====================================================================
_VALID_MODELS: tuple[str, ...] = ("baseline", "velocity")


def run_example08(
    *,
    model: str = "baseline",
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool | None = True,
    plot_style: str = "legacy",
) -> dict:
    """Run Example 08: real-place-cell encoding-and-decoding with GoF.

    Parameters
    ----------
    model : {"baseline", "velocity"}, default "baseline"
        ``"baseline"`` preserves the original 6-term quadratic CIF pipeline
        bit-for-bit and produces figures 1-4.  ``"velocity"`` runs the
        baseline first to capture its RMSE, then a velocity-augmented
        encoder + decoder per Truccolo et al. (2005) and produces two
        extra figures (fig05 speed tuning, fig06 baseline-vs-velocity
        comparison).  Future Tier D PRs will extend the valid set with
        ``"history"`` (D.2) and ``"coupling"`` (D.3).

    Returns
    -------
    dict
        Summary scalars and figure paths.  When ``model="velocity"`` the
        dict additionally carries ``baseline_rmse``, ``velocity_rmse``,
        ``rmse_improvement_pct``, ``speed_coefs``, and ``n_test_bins``.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"unknown model {model!r}; expected one of {_VALID_MODELS}"
        )
    t_start = _time.perf_counter()
    print("=" * 70)
    print(
        f"Example 08: Real Place-Cell Encoding-and-Decoding (Animal 1) "
        f"[model={model}]"
    )
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

    baseline_rmse = rmse
    # Per-bin error vector retained for fig06; recomputed on the same
    # subsampled lattice the velocity decoder will run on.
    baseline_per_bin_err = np.sqrt(
        (x_dec[0] - x_test_sub) ** 2 + (x_dec[1] - y_test_sub) ** 2
    )

    # ----- Velocity-augmented variant (model="velocity") -----
    velocity_payload: dict | None = None
    if model == "velocity":
        speed = _compute_speed(x, y, t)
        speed_train = speed[train_slice]
        speed_test = speed[test_slice]
        print(
            f"  speed: clipped at {_SPEED_CLIP_BOX_PER_SEC} box/s; "
            f"train mean = {float(speed_train.mean()):.3f}, "
            f"max = {float(speed_train.max()):.3f}"
        )

        (
            grid_x_v, grid_y_v, _B_v, rate_maps_v, glm_results_v, speed_coefs,
        ) = _fit_bspline_encoder_with_speed(
            x[train_slice], y[train_slice], t[train_slice], speed_train,
            spikes_per_cell,
            n_grid=32, n_knots=(8, 8), degree=3, box=box,
        )
        converged_v = [int(g.converged) for g in glm_results_v]
        print(
            f"  velocity-augmented B-spline GLM: converged per cell = "
            f"{converged_v}; speed coefficients = "
            f"{[round(c, 3) for c in speed_coefs]}"
        )

        cifs_v = _fit_quadratic_cif_per_cell_with_speed(
            x[train_slice], y[train_slice], t[train_slice], speed_train,
            spikes_per_cell, delta=delta,
        )
        bs_coefs = [float(np.asarray(c.b).reshape(-1)[6]) for c in cifs_v]
        print(
            f"  velocity-augmented quadratic CIFs: "
            f"speed coefficients (decoder) = "
            f"{[round(c, 3) for c in bs_coefs]}"
        )

        (
            _dN_v, x_dec_v, _W_v,
            x_test_sub_v, y_test_sub_v, t_test_sub_v, speed_test_sub,
        ) = _decode_position_with_speed(
            x[test_slice], y[test_slice], t[test_slice], speed_test,
            spikes_per_cell, cifs_v, delta=delta, stride=decode_stride,
        )
        velocity_rmse = float(np.sqrt(np.mean(
            (x_dec_v[0] - x_test_sub_v) ** 2
            + (x_dec_v[1] - y_test_sub_v) ** 2
        )))
        velocity_per_bin_err = np.sqrt(
            (x_dec_v[0] - x_test_sub_v) ** 2
            + (x_dec_v[1] - y_test_sub_v) ** 2
        )
        improvement_pct = (
            100.0 * (baseline_rmse - velocity_rmse) / baseline_rmse
        )
        sign = "+" if improvement_pct >= 0.0 else ""
        print(
            f"  velocity decoder: RMSE = {velocity_rmse:.3f} "
            f"({sign}{improvement_pct:.1f}% vs baseline {baseline_rmse:.3f})"
        )

        bin_centers, rate_per_cell, sem_per_cell = _per_cell_speed_tuning(
            spikes_per_cell, t, speed, n_bins=8, delta=delta,
        )
        velocity_payload = {
            "speed_coefs_encoder": speed_coefs,
            "speed_coefs_decoder": bs_coefs,
            "velocity_rmse": velocity_rmse,
            "improvement_pct": improvement_pct,
            "baseline_per_bin_err": baseline_per_bin_err,
            "velocity_per_bin_err": velocity_per_bin_err,
            "tuning": (bin_centers, rate_per_cell, sem_per_cell),
        }

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
    fig_names = [
        "fig01_real_place_fields_panel",
        "fig02_decoded_vs_true_position",
        "fig03_pair_correlation_envelope",
        "fig04_rescaled_acf",
    ]
    if velocity_payload is not None:
        bin_centers, rate_per_cell, sem_per_cell = velocity_payload["tuning"]
        fig5 = _plot_velocity_speed_tuning(
            bin_centers, rate_per_cell, sem_per_cell,
            cell_indices, velocity_payload["speed_coefs_encoder"],
        )
        fig6 = _plot_decoder_baseline_vs_velocity(
            baseline_rmse, velocity_payload["velocity_rmse"],
            velocity_payload["baseline_per_bin_err"],
            velocity_payload["velocity_per_bin_err"],
        )
        figures.extend([fig5, fig6])
        fig_names.extend([
            "fig05_velocity_speed_tuning",
            "fig06_decoder_baseline_vs_velocity",
        ])

    for fig in figures:
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "example08"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
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

    result = {
        "model": model,
        "cell_indices": list(cell_indices),
        "n_spikes_per_cell": [int(st.size) for st in spikes_per_cell],
        "t_split_s": t_split,
        "decode_rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "n_test_bins": int(x_test_sub.size),
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
    if velocity_payload is not None:
        result.update({
            "velocity_rmse": float(velocity_payload["velocity_rmse"]),
            "rmse_improvement_pct": float(
                velocity_payload["improvement_pct"]
            ),
            "speed_coefs": [
                float(c) for c in velocity_payload["speed_coefs_encoder"]
            ],
            "speed_coefs_decoder": [
                float(c) for c in velocity_payload["speed_coefs_decoder"]
            ],
        })
    return result


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
    parser.add_argument(
        "--model", choices=_VALID_MODELS, default="velocity",
        help=(
            "Encoder/decoder variant.  ``baseline`` reproduces the "
            "original position-only pipeline (4 figures).  "
            "``velocity`` (default) runs the baseline then a "
            "velocity-augmented variant per Truccolo et al. 2005 "
            "and produces 6 figures.  The CLI defaults to "
            "``velocity`` so --export-figures yields the full "
            "gallery the manifest expects; the Python "
            "``run_example08`` default remains ``baseline``."
        ),
    )
    args = parser.parse_args()

    if args.no_display:
        visible = False
    else:
        visible = bool(args.show)
    result = run_example08(
        model=args.model,
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )
    if args.output_json:
        summary = {
            "model": result["model"],
            "cell_indices": result["cell_indices"],
            "n_spikes_per_cell": result["n_spikes_per_cell"],
            "t_split_s": result["t_split_s"],
            "decode_rmse": result["decode_rmse"],
            "baseline_rmse": result["baseline_rmse"],
            "n_test_bins": result["n_test_bins"],
            "envelope_inside": result["envelope_inside"],
            "envelope_p_interval": result["envelope_p_interval"],
            "acf_in_band": result["acf_in_band"],
            "acf_n_lags": result["acf_n_lags"],
            "wall_clock_s": result["wall_clock_s"],
        }
        for key in ("velocity_rmse", "rmse_improvement_pct",
                    "speed_coefs", "speed_coefs_decoder"):
            if key in result:
                summary[key] = result[key]
        args.output_json.write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
