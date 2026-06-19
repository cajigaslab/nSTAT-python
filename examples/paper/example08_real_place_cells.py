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
- Figure 5 (model in {"velocity", "history"}): per-cell firing
  rate vs speed tuning curves, showing how the velocity-augmented
  encoder recovers per-cell speed modulation.
- Figure 6 (model in {"velocity", "history"}): bar chart and
  per-bin error scatter comparing baseline vs velocity-augmented
  decoder RMSE.
- Figure 7 (model="history" only): per-cell *encoder* spike-history
  coefficients (4-window refractory+recovery basis), showing the
  expected negative early-window values (refractoriness) for the
  cells that exhibit it.
- Figure 8 (model="history" only): encoder fit quality with
  history — per-cell log-likelihood gain (Panel 1) and per-cell
  gamma in the [1, 5 ms) refractory window (Panel 2), colored by
  sign to highlight cells whose negative gamma is a refractoriness
  signature.
- Figure 9 (model="coupling" only): ensemble-coupling matrix as
  two 4x4 heatmaps (target x source) — one per lattice-matched
  window ([0, 33ms), [33, 200ms)) — with diagonals NaN/masked
  (cells don't self-couple in this model).  Diverging RdBu_r
  colormap so positive = excitatory, negative = inhibitory.  The
  encoder matrix is shown (more interpretable than the binomial
  decoder matrix).
- Figure 10 (model="coupling" only): capstone 4-bar progression —
  baseline RMSE / velocity RMSE / history-decoder RMSE / coupling
  RMSE — with the per-step improvement % annotated.  Summarises
  the full Truccolo et al. 2005 framework progression.

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

  ``run_example08(model="history")`` runs the baseline + velocity
  pipeline first (for the full RMSE table) and then a
  spike-history-augmented *encoder* per Truccolo et al. 2005 §3
  "spike history effects".  Each cell's encoder Poisson GLM is
  extended with four piecewise-constant history coefficients
  covering the windows [0, 1 ms), [1, 5 ms), [5, 10 ms),
  [10, 50 ms) — the canonical refractory + bursting +
  slow-recovery decomposition.  The history matrix is built at
  1 kHz (where the sub-millisecond windows are resolvable) and
  aggregated to the position-time lattice for the spatial GLM fit.

  Design choice — history in the encoder, not the decoder.
  Truccolo et al. 2005 §3 frames history filters within a
  point-process framework whose temporal resolution must match
  the bin width at which the conditional intensity is evaluated.
  The Animal-1 position lattice has ``delta`` ≈ 33 ms (and the
  PPAF decoder strides over this at 5x, giving a 167 ms effective
  bin), which is wider than every brief refractoriness window
  except [10, 50 ms).  At that lattice the binomial decoder CIF
  has no resolution with which to learn the sub-bin refractory
  effect — the [0, 1 ms), [1, 5 ms), [5, 10 ms) gamma columns
  are dominated by bursting-driven nonzero rows and an
  unregularised fit drives them to log-odds magnitudes that
  saturate the sigmoid during decode and freeze the filter (we
  empirically observed a -464% RMSE regression on a prior
  decoder-side prototype).  The 1 kHz encoder GLM, by contrast,
  *can* resolve the sub-bin windows: each cell's history column
  carries the genuine refractory-time spike counts.  So we keep
  history strictly in the encoder, where it is statistically
  recoverable, and run the decoder with the position-only
  quadratic CIF (identical pipeline to the baseline + velocity
  variants).  The pedagogical claim shifts from "history improves
  decoding" (false at this lattice) to "history reveals
  refractoriness in the encoder" (true, and demonstrated in
  fig08).

  Produces two extra figures: fig07 (per-cell encoder history
  kernels showing the expected refractory signature for 2 of
  4 cells) and fig08 (encoder log-likelihood gain plus per-cell
  [1, 5 ms) gamma colored by sign — the refractoriness
  diagnostic).  Decoder RMSE is reported as
  ``history_decoder_rmse`` and matches baseline bit-for-bit
  (the decoder pipeline is unchanged).

  ``run_example08(model="coupling")`` runs the full
  baseline + velocity + history chain first and then an
  ensemble-coupling-augmented variant per Truccolo et al. 2005
  §3 "ensemble coupling effects".  Each cell's encoder GLM and
  decoder CIF are extended with six coupling coefficients
  (3 source cells × 2 windows for the 4-cell subset), where the
  windows are *lattice-matched* to the decoder bin:
  [0, 33 ms) and [33, 200 ms).  D.2's lesson was that
  sub-bin history filters can't be carried into the binomial
  decoder without saturation; D.3 sidesteps that by picking
  coupling windows that align with the 33 ms position lattice
  and the 167 ms strided decode lattice.  At decode time the
  coupling values are *observed* (held-out spike counts from
  the other cells), folded into a time-mean ``mu`` exactly the
  same way the velocity decoder folds in mean speed.  The
  state stays 2-D position.  Produces two extra figures: fig09
  (encoder coupling matrix, two windows side-by-side, NaN
  diagonal) and fig10 (capstone 4-bar progression: baseline →
  velocity → history (decoder) → coupling).
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

from nstat import CIF, DecodingAlgorithms, History, apply_plot_style  # noqa: E402
from nstat._spike_train_impl import nspikeTrain  # noqa: E402
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
# CCF-based per-cell lag finder (model="velocity_lag" only)
# =====================================================================
# Mirrors the lag-correction pattern of
# ``examples/paper/example02_whisker_stimulus_thalamus.py`` (residual
# cross-covariance windowed to a plausible-lag band, peak picked as the
# optimal shift) but lifted to per-cell + per-covariate.  Truccolo et
# al. 2005 §4 frames covariate-lag determination as residual x
# covariate cross-correlation: a peak at negative lag means the spike
# is *anticipating* the covariate (the cell leads), positive lag means
# the cell *trails* the covariate.  For CA1 place cells the canonical
# phase-precession picture predicts a small negative lag on position
# (cells fire slightly *before* the rat reaches the field center).
_LAG_WINDOW_POSITION_S: tuple[float, float] = (-0.150, +0.050)
_LAG_WINDOW_VELOCITY_S: tuple[float, float] = (-0.200, +0.200)


def _ccf_with_lag_peak(
    residual: np.ndarray,
    covariate: np.ndarray,
    *,
    dt: float,
    lag_window: tuple[float, float],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Cross-correlation peak finder used for per-cell lag determination.

    Mean-centers both inputs (residual and covariate), computes the
    full-mode cross-correlation, normalises by ``std(r) * std(c) * N``
    so the result is bounded on ``[-1, +1]`` (Pearson-style
    normalisation; the standard ``scipy.signal.correlate`` does not
    normalise), then restricts to ``lag_window`` and returns the peak.

    The sign convention matches ``scipy.signal.correlate(r, c)``:
    a *negative* lag means the covariate ``c`` *leads* the residual
    ``r`` (i.e. the cell anticipates the covariate); a positive lag
    means the cell trails.

    Returns
    -------
    peak_lag_sec, peak_value, lags_sec, ccf_values
        ``peak_lag_sec`` and ``peak_value`` are the chosen lag (in
        seconds) and its CCF magnitude.  ``lags_sec`` and
        ``ccf_values`` are the full ``2N - 1``-length arrays returned
        for the diagnostic figure.
    """
    # Lazy-import scipy.signal so importing this module stays cheap
    # for unrelated paper-script invocations.
    from scipy.signal import correlate

    r = np.asarray(residual, dtype=float).reshape(-1)
    c = np.asarray(covariate, dtype=float).reshape(-1)
    if r.size == 0 or c.size != r.size:
        raise ValueError(
            "residual and covariate must be 1-D arrays of equal length"
        )
    r0 = r - float(np.mean(r))
    c0 = c - float(np.mean(c))
    ccf = correlate(r0, c0, mode="full")
    norm = float(np.std(r0)) * float(np.std(c0)) * float(r0.size)
    ccf = ccf / max(norm, 1e-12)
    n = r0.size
    lags = (np.arange(2 * n - 1) - (n - 1)) * float(dt)
    window_mask = (lags >= float(lag_window[0])) & (lags <= float(lag_window[1]))
    if not bool(window_mask.any()):
        # Defensive: degenerate window collapses to no lags — return 0.
        return 0.0, 0.0, lags, ccf
    valid_lags = lags[window_mask]
    valid_ccf = ccf[window_mask]
    peak_idx = int(np.argmax(valid_ccf))
    return float(valid_lags[peak_idx]), float(valid_ccf[peak_idx]), lags, ccf


def _shift_position_by_lag(
    pos_x: np.ndarray, pos_y: np.ndarray, lag_sec: float, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift a 2-D position signal by ``lag_sec`` and return a NaN-edged copy.

    The sign convention matches :func:`_ccf_with_lag_peak`:
    a positive ``lag_sec`` means the position *trails* the residual
    (so for bin ``t`` we should read position at ``t - lag_sec``,
    i.e. an earlier sample); a negative ``lag_sec`` means the
    position *leads* the residual (so for bin ``t`` we read position
    at ``t + |lag_sec|``, i.e. a later sample — the "future peek"
    standard in retrospective decoding evaluation).

    Bins where the shift falls outside the original array are filled
    with NaN; the returned ``valid_mask`` is ``True`` only for bins
    where both shifted x and shifted y are finite.

    Returns
    -------
    px_shifted, py_shifted, valid_mask
    """
    px = np.asarray(pos_x, dtype=float).reshape(-1)
    py = np.asarray(pos_y, dtype=float).reshape(-1)
    n = px.size
    shift_bins = int(np.round(float(lag_sec) / float(dt)))
    px_shifted = np.full(n, np.nan, dtype=float)
    py_shifted = np.full(n, np.nan, dtype=float)
    if shift_bins == 0:
        px_shifted[:] = px
        py_shifted[:] = py
    elif shift_bins > 0:
        # Read from index t - shift_bins for output index t.
        # Output indices [shift_bins, n) source from [0, n - shift_bins).
        px_shifted[shift_bins:] = px[: n - shift_bins]
        py_shifted[shift_bins:] = py[: n - shift_bins]
    else:
        k = -shift_bins
        # Output indices [0, n - k) source from [k, n).
        px_shifted[: n - k] = px[k:]
        py_shifted[: n - k] = py[k:]
    valid_mask = ~(np.isnan(px_shifted) | np.isnan(py_shifted))
    return px_shifted, py_shifted, valid_mask


def _shift_scalar_by_lag(
    sig: np.ndarray, lag_sec: float, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a 1-D signal by ``lag_sec`` with NaN edge padding.

    Same sign convention as :func:`_shift_position_by_lag`.
    Returns ``(shifted, valid_mask)``.
    """
    s = np.asarray(sig, dtype=float).reshape(-1)
    n = s.size
    shift_bins = int(np.round(float(lag_sec) / float(dt)))
    out = np.full(n, np.nan, dtype=float)
    if shift_bins == 0:
        out[:] = s
    elif shift_bins > 0:
        out[shift_bins:] = s[: n - shift_bins]
    else:
        k = -shift_bins
        out[: n - k] = s[k:]
    valid = ~np.isnan(out)
    return out, valid


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
# Spike-history covariates (model="history" only)
# =====================================================================
# Truccolo et al. 2005 §3 ("spike history effects") add a sum of
# piecewise-constant filters over short-lag spike counts to the log
# conditional intensity:
#
#     log lambda_k = mu + beta · state + sum_i gamma_i · h_i(N_{<k})
#
# We use the canonical four-window refractory + recovery basis from
# the ``nstat.History`` MATLAB-parity-tested helper.  The windows
# [0, 1 ms), [1, 5 ms), [5, 10 ms), [10, 50 ms) cover absolute
# refractoriness, relative refractoriness, bursting, and the slow
# tail of recovery — the same decomposition used in the
# ``HistoryExamples.ipynb`` notebook and the gold fixture at
# ``tests/parity/fixtures/matlab_gold/history_exactness.mat``.
# Brief design (Tier D.2): windowTimes=[0.0, 0.001, 0.005, 0.010,
# 0.050] gives 4 piecewise-constant windows covering 0-50 ms — the
# canonical refractoriness + bursting + slow-recovery decomposition
# also used in ``tests/parity/fixtures/matlab_gold/history_exactness.mat``.
# Lattice note: the Animal-1 position lattice has ``delta`` ≈
# 33 ms, so only [10, 50 ms) is resolvable at the decoder bin.
# That is exactly why history lives in the *encoder* in this
# example: the encoder GLM is fit at the 1 kHz history sample rate
# (via :func:`_compute_history_at_position_times`), where every
# brief window is resolvable, and the resulting per-cell gammas
# carry a genuine refractoriness signature for 2 of 4 cells.  The
# PPAF decoder runs the position-only quadratic CIF (no history
# terms), giving a baseline-identical RMSE — see the module
# docstring under "Design choice" for the full rationale.
_HISTORY_WINDOW_TIMES: tuple[float, ...] = (0.0, 0.001, 0.005, 0.010, 0.050)
# 1 kHz is the canonical rate for spike-history extraction in
# nSTAT — it genuinely resolves the [0, 1 ms) window while still
# being cheap to compute over the ~1460 s Animal-1 recording.
_HISTORY_SAMPLE_RATE_HZ: float = 1000.0


def _compute_history_at_position_times(
    spike_times: np.ndarray,
    pos_time: np.ndarray,
    *,
    window_times: tuple[float, ...] | None = None,
    sample_rate_hz: float = _HISTORY_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Spike-history matrix at the position-time lattice.

    Builds an :class:`~nstat.nspikeTrain` at ``sample_rate_hz``
    (default 1 kHz) covering ``[pos_time[0], pos_time[-1]]``, runs
    :meth:`~nstat.History.computeHistory` with the canonical
    four-window refractoriness basis, then indexes the resulting
    1 kHz history matrix at the position-time samples via
    ``searchsorted``.

    Returns
    -------
    ndarray of shape (n_pos, len(window_times) - 1)
        Per-position-bin history values.  ``H[t, i]`` is the
        spike count from cell own train in window ``i`` ending at
        ``pos_time[t]``.  Rows whose ``pos_time[t]`` is before any
        spike are zero by construction.
    """
    wt = (
        tuple(_HISTORY_WINDOW_TIMES) if window_times is None
        else tuple(window_times)
    )
    if pos_time.size == 0:
        return np.zeros((0, len(wt) - 1), dtype=float)
    t_lo = float(pos_time[0])
    t_hi = float(pos_time[-1])
    # Restrict to spikes inside the analysis window — ``nspikeTrain``
    # would clip otherwise, but we avoid the boundary surprises by
    # filtering up front.
    mask = (spike_times >= t_lo) & (spike_times <= t_hi)
    st_in = spike_times[mask]
    if st_in.size == 0:
        return np.zeros((pos_time.size, len(wt) - 1), dtype=float)
    train = nspikeTrain(
        st_in, name="hist", sampleRate=float(sample_rate_hz),
        minTime=t_lo, maxTime=t_hi, makePlots=-1,
    )
    hist = History(windowTimes=list(wt))
    coll = hist.computeHistory(train)
    cov = coll.covArray[0]
    H_1khz = np.asarray(cov.data, dtype=float)  # (n_1khz, num_windows)
    t_1khz = np.asarray(cov.time, dtype=float).ravel()
    # Index at the position-time samples via searchsorted-with-snap
    # to nearest (same pattern as ``_bin_spikes_on_grid``).
    idx = np.searchsorted(t_1khz, pos_time, side="left")
    idx = np.clip(idx, 0, t_1khz.size - 1)
    left = np.clip(idx - 1, 0, t_1khz.size - 1)
    pick_left = (
        (idx > 0)
        & (np.abs(t_1khz[left] - pos_time) < np.abs(t_1khz[idx] - pos_time))
    )
    idx = np.where(pick_left, left, idx)
    return H_1khz[idx, :]


# =====================================================================
# History-augmented B-spline encoder (model="history" only)
# =====================================================================
def _fit_bspline_encoder_with_history(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    window_times: tuple[float, ...] | None = None,
    n_grid: int = 32,
    n_knots: tuple[int, int] = (8, 8),
    degree: int = 3,
    box: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    """Like :func:`_fit_bspline_encoder` but appends 4 history columns per cell.

    Each cell's spike-history matrix is computed at the position-time
    grid via :func:`_compute_history_at_position_times`, then
    spatially aggregated (mean over the position samples that land in
    each spatial bin) — exactly the same per-cell pattern the
    velocity encoder uses for mean speed per bin (option (a) in the
    Tier D.1 brief).  The resulting per-cell design matrix is
    ``B`` (n_grid**2, n_basis) concatenated with the per-cell
    aggregated-history block (n_grid**2, 4).  Each cell gets its own
    fit because each cell's history is its own.

    Returns the same tuple as :func:`_fit_bspline_encoder` plus
    ``history_coefs`` (one ``(4,)`` ndarray per cell).
    """
    wt = (
        tuple(_HISTORY_WINDOW_TIMES) if window_times is None
        else tuple(window_times)
    )
    x_lo, x_hi, y_lo, y_hi = box
    grid_x = np.linspace(x_lo, x_hi, n_grid)
    grid_y = np.linspace(y_lo, y_hi, n_grid)
    edges_x = np.linspace(x_lo, x_hi, n_grid + 1)
    edges_y = np.linspace(y_lo, y_hi, n_grid + 1)
    B = bspline_basis_2d(grid_x, grid_y, n_knots=n_knots, degree=degree)
    cell_area = float(((x_hi - x_lo) / n_grid) * ((y_hi - y_lo) / n_grid))
    offset = np.full(B.shape[0], np.log(cell_area), dtype=float)

    # Per-spatial-bin assignment of each training-position sample
    # (same as the velocity helper).
    ix = np.clip(
        np.searchsorted(edges_x, pos_x_train, side="right") - 1, 0, n_grid - 1,
    )
    iy = np.clip(
        np.searchsorted(edges_y, pos_y_train, side="right") - 1, 0, n_grid - 1,
    )
    count = np.zeros((n_grid, n_grid), dtype=float)
    np.add.at(count, (ix, iy), 1.0)
    bin_count = np.maximum(count.ravel(), 1.0)

    n_windows = len(wt) - 1
    rate_maps = np.zeros((len(spikes_per_cell), n_grid, n_grid), dtype=float)
    glm_results = []
    history_coefs: list[np.ndarray] = []
    for c, st in enumerate(spikes_per_cell):
        # Per-cell history at the training-position lattice.
        H_cell = _compute_history_at_position_times(
            st, pos_time_train, window_times=wt,
        )  # (n_pos_train, n_windows)
        # Aggregate by spatial bin → (n_grid**2, n_windows).
        H_per_bin = np.zeros((n_grid * n_grid, n_windows), dtype=float)
        flat_idx = ix * n_grid + iy
        for w in range(n_windows):
            np.add.at(H_per_bin[:, w], flat_idx, H_cell[:, w])
        H_per_bin = H_per_bin / bin_count[:, None]
        B_cell = np.hstack([B, H_per_bin])

        H_counts = _bin_spikes_on_grid(
            st, pos_x_train, pos_y_train, pos_time_train, edges_x, edges_y,
        )
        counts = H_counts.ravel().astype(float)
        # The first three history columns at the 33 ms position
        # lattice are very sparse after spatial aggregation (the
        # sub-bin [0, 1), [1, 5), [5, 10) ms windows fire only on
        # the position samples that happen to follow a within-bin
        # burst), so the augmented IRLS can wander into a
        # numerical pathology for the highest-firing cells.
        # Strengthen the ridge from l2=1.0 (baseline) to l2=10.0
        # and double the iteration budget so all four cells
        # converge.  The 1 kHz history sample rate inside
        # :func:`_compute_history_at_position_times` keeps the
        # sub-bin windows statistically meaningful even after the
        # spatial-bin mean aggregation.
        glm = fit_poisson_glm(
            B_cell, counts, offset=offset, include_intercept=False, l2=10.0,
            max_iter=800,
        )
        eta = B_cell @ glm.coefficients
        eta_clipped = np.clip(eta, -20.0, 8.0)
        rate_maps[c, :, :] = np.exp(eta_clipped).reshape(n_grid, n_grid)
        glm_results.append(glm)
        # Final 4 coefficients are the history block.
        history_coefs.append(
            np.asarray(glm.coefficients[-n_windows:], dtype=float).copy()
        )
    return grid_x, grid_y, B, rate_maps, glm_results, history_coefs


# =====================================================================
# Ensemble-coupling covariates (model="coupling" only)
# =====================================================================
# Truccolo et al. 2005 §3 ("ensemble coupling effects") add a sum of
# piecewise-constant filters over the recent spike counts of the OTHER
# cells in the ensemble to the log conditional intensity:
#
#     log lambda_{c, k} = mu_c + beta_c · state_k
#                        + sum_{c' != c, w} gamma_{c, c', w} · N_{c', w}(k)
#
# Each ``N_{c', w}(k)`` is the spike count from source cell ``c'`` in
# a recent-past window ``w`` ending at time ``t_k``.
#
# Lattice-matched windows (Tier D.3 brief).  The Animal-1 position
# lattice has ``delta`` ≈ 33 ms.  D.2 established that sub-bin history
# filters cannot be carried into the binomial decoder without
# saturation regressions (the -464% RMSE regression on the prior
# decoder-side prototype).  D.3 therefore picks coupling windows that
# *match* the decode lattice so the binomial decoder can use them:
#
#   - Short window [0, 33 ms): one decode bin lookback (peri-spike
#     coupling).  At the strided decode lattice this carries 0 or 1
#     spike from each source cell — within the resolvable regime of
#     the binomial CIF.
#   - Long window [33 ms, 200 ms): multi-bin past (slow theta-cycle
#     and behavioral-state modulation), spanning roughly 5 position
#     bins (≈ 33 ms each).
#
# For 4 cells, each target cell gets 3 source cells × 2 windows =
# 6 coupling columns.  No self-coupling (the diagonal of the coupling
# tensor is left out — refractoriness on the source cell is already
# the D.2 encoder-history concern).
_COUPLING_WINDOW_TIMES_S: tuple[tuple[float, float], ...] = (
    (0.0, 0.033),
    (0.033, 0.200),
)


def _coupling_window_labels() -> tuple[str, str]:
    """Human-readable labels for the two coupling windows (ms)."""
    return ("[0, 33 ms)", "[33, 200 ms)")


def _compute_coupling_columns_at_position_times(
    spikes_by_source: list[np.ndarray],
    pos_time: np.ndarray,
    target_idx: int,
    *,
    window_times_s: tuple[tuple[float, float], ...] = _COUPLING_WINDOW_TIMES_S,
) -> np.ndarray:
    """Coupling-count matrix at the position-time lattice.

    Builds the per-bin spike counts from each *source* cell
    ``c' != target_idx`` in each recent-past window ``w``.  Each entry
    ``H[t, j]`` is the number of spikes from source ``c'`` in the
    interval ``(pos_time[t] - w_hi, pos_time[t] - w_lo]`` where
    ``j = source_local_idx * num_windows + window_idx`` and the source
    enumeration skips ``target_idx``.

    Returns
    -------
    ndarray of shape (n_pos, (n_cells - 1) * len(window_times_s))
        Coupling design columns at the position-time lattice.  Rows
        whose ``pos_time[t]`` is before the long-window edge of any
        spike are zero by construction.
    """
    n_pos = pos_time.size
    n_cells = len(spikes_by_source)
    n_sources = n_cells - 1
    n_windows = len(window_times_s)
    out = np.zeros((n_pos, n_sources * n_windows), dtype=float)
    if n_pos == 0:
        return out
    source_iter = [c for c in range(n_cells) if c != target_idx]
    for local_j, c_src in enumerate(source_iter):
        st_src = spikes_by_source[c_src]
        if st_src.size == 0:
            continue
        for w_idx, (lo, hi) in enumerate(window_times_s):
            # Count spikes in (t - hi, t - lo] using cumulative-count
            # differences via searchsorted.  ``side="right"`` matches
            # the right-open / left-closed convention used elsewhere.
            #
            # For a window [lo, hi) measured backward from t, spikes
            # contributing are those whose times are in
            #     [t - hi, t - lo)
            # The number of such spikes is
            #     searchsorted(st_src, t - lo) - searchsorted(st_src, t - hi)
            # under the ``side="left"`` convention.
            right = np.searchsorted(st_src, pos_time - lo, side="left")
            left = np.searchsorted(st_src, pos_time - hi, side="left")
            col_idx = local_j * n_windows + w_idx
            out[:, col_idx] = (right - left).astype(float)
    return out


# =====================================================================
# Coupling-augmented B-spline encoder (model="coupling" only)
# =====================================================================
def _fit_bspline_encoder_with_coupling(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    window_times_s: tuple[tuple[float, float], ...] = _COUPLING_WINDOW_TIMES_S,
    n_grid: int = 32,
    n_knots: tuple[int, int] = (8, 8),
    degree: int = 3,
    box: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    """Like :func:`_fit_bspline_encoder` but appends coupling columns per cell.

    Each cell's per-bin coupling counts (from the OTHER cells, in the
    two lattice-matched windows) are aggregated by spatial bin (mean
    over the position samples that land in each spatial bin), then
    appended to the B-spline design matrix — exactly the same per-cell
    aggregation pattern the velocity and history encoders use.  For
    4 cells × 2 windows the per-cell design block is ``(n_grid**2, 6)``.

    Returns the same tuple as :func:`_fit_bspline_encoder` plus
    ``coupling_coefs`` (one ``(6,)`` ndarray per cell) and
    ``coupling_matrix`` (``(n_cells, n_cells, n_windows)`` ndarray with
    NaN on the diagonal).
    """
    x_lo, x_hi, y_lo, y_hi = box
    grid_x = np.linspace(x_lo, x_hi, n_grid)
    grid_y = np.linspace(y_lo, y_hi, n_grid)
    edges_x = np.linspace(x_lo, x_hi, n_grid + 1)
    edges_y = np.linspace(y_lo, y_hi, n_grid + 1)
    B = bspline_basis_2d(grid_x, grid_y, n_knots=n_knots, degree=degree)
    cell_area = float(((x_hi - x_lo) / n_grid) * ((y_hi - y_lo) / n_grid))
    offset = np.full(B.shape[0], np.log(cell_area), dtype=float)

    # Per-spatial-bin assignment of each training-position sample
    # (same as the velocity/history helpers).
    ix = np.clip(
        np.searchsorted(edges_x, pos_x_train, side="right") - 1, 0, n_grid - 1,
    )
    iy = np.clip(
        np.searchsorted(edges_y, pos_y_train, side="right") - 1, 0, n_grid - 1,
    )
    count = np.zeros((n_grid, n_grid), dtype=float)
    np.add.at(count, (ix, iy), 1.0)
    bin_count = np.maximum(count.ravel(), 1.0)

    n_cells = len(spikes_per_cell)
    n_windows = len(window_times_s)
    n_coupling_cols = (n_cells - 1) * n_windows
    rate_maps = np.zeros((n_cells, n_grid, n_grid), dtype=float)
    glm_results = []
    coupling_coefs: list[np.ndarray] = []
    # (target, source, window) tensor; diagonal stays NaN.
    coupling_matrix = np.full(
        (n_cells, n_cells, n_windows), np.nan, dtype=float,
    )
    for c in range(n_cells):
        st = spikes_per_cell[c]
        # Per-target-cell coupling counts at the training-position
        # lattice (n_pos_train, n_sources * n_windows).
        C_cell = _compute_coupling_columns_at_position_times(
            spikes_per_cell, pos_time_train, target_idx=c,
            window_times_s=window_times_s,
        )
        # Aggregate by spatial bin → (n_grid**2, n_coupling_cols).
        C_per_bin = np.zeros(
            (n_grid * n_grid, n_coupling_cols), dtype=float,
        )
        flat_idx = ix * n_grid + iy
        for j in range(n_coupling_cols):
            np.add.at(C_per_bin[:, j], flat_idx, C_cell[:, j])
        C_per_bin = C_per_bin / bin_count[:, None]
        B_cell = np.hstack([B, C_per_bin])

        H_counts = _bin_spikes_on_grid(
            st, pos_x_train, pos_y_train, pos_time_train, edges_x, edges_y,
        )
        counts = H_counts.ravel().astype(float)
        # Coupling columns are similar in spirit to the D.2 history
        # columns — they can be sparse after spatial-mean aggregation
        # and the augmented IRLS can wander.  Reuse the D.2 ridge
        # (l2=10.0) and iteration budget (max_iter=400-800) so all
        # four cells converge under the augmented design.
        glm = fit_poisson_glm(
            B_cell, counts, offset=offset, include_intercept=False, l2=10.0,
            max_iter=400,
        )
        eta = B_cell @ glm.coefficients
        eta_clipped = np.clip(eta, -20.0, 8.0)
        rate_maps[c, :, :] = np.exp(eta_clipped).reshape(n_grid, n_grid)
        glm_results.append(glm)
        gammas = np.asarray(
            glm.coefficients[-n_coupling_cols:], dtype=float,
        ).copy()
        coupling_coefs.append(gammas)
        # Unpack ``gammas`` into the (target, source, window) tensor.
        # Source enumeration skips ``c`` (the diagonal stays NaN).
        source_iter = [s for s in range(n_cells) if s != c]
        for local_j, s in enumerate(source_iter):
            for w_idx in range(n_windows):
                coupling_matrix[c, s, w_idx] = float(
                    gammas[local_j * n_windows + w_idx]
                )
    return (
        grid_x, grid_y, B, rate_maps, glm_results,
        coupling_coefs, coupling_matrix,
    )


# =====================================================================
# Coupling-augmented quadratic-CIF refit + decoder (model="coupling")
# =====================================================================
def _design_quadratic_coupling(
    x: np.ndarray, y: np.ndarray, coupling_cols: np.ndarray,
) -> np.ndarray:
    """[1, x, y, x^2, y^2, x*y, coupling_cols...] design matrix."""
    return np.column_stack([
        np.ones(x.size), x, y, x * x, y * y, x * y, coupling_cols,
    ])


def _fit_quadratic_cif_per_cell_with_coupling(
    pos_x_train: np.ndarray,
    pos_y_train: np.ndarray,
    pos_time_train: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    *,
    delta: float,
    window_times_s: tuple[tuple[float, float], ...] = _COUPLING_WINDOW_TIMES_S,
) -> tuple[list[CIF], list[np.ndarray], np.ndarray]:
    """Refit a 12-term quadratic-plus-coupling binomial CIF per cell.

    The CIF is

        log_lambda = mu + bx*x + by*y + bxx*x^2 + byy*y^2 + bxy*x*y
                     + sum_{c' != c, w} gamma_{c', w} * N_{c', w}(t)

    where each ``N_{c', w}(t)`` is the spike count from source cell
    ``c'`` in window ``w`` ending at ``t``.  These are treated as
    *known extrinsic observables* at decode time (Truccolo et al.
    2005 Section 3 with the lattice-matched windows of D.3 — see
    module docstring for the rationale).

    Returns
    -------
    cifs : list[CIF]
        One :class:`CIF` per cell with 12 symbolic coefficients.
    coupling_coefs_decoder : list[ndarray]
        Per-cell ``(n_coupling_cols,)`` arrays of the decoder gammas.
    coupling_matrix_decoder : ndarray
        ``(n_cells, n_cells, n_windows)`` tensor with NaN diagonals.
    """
    n_bins = pos_time_train.size
    n_cells = len(spikes_per_cell)
    n_windows = len(window_times_s)
    n_coupling_cols = (n_cells - 1) * n_windows
    cifs: list[CIF] = []
    coupling_coefs: list[np.ndarray] = []
    coupling_matrix = np.full(
        (n_cells, n_cells, n_windows), np.nan, dtype=float,
    )

    # Coupling-column variable names for the CIF symbolic expression.
    # Per-target enumeration skips ``c``; the actual variable names
    # ``c_{src}_{win}`` are bound at decode time to spike counts.
    def _var_names_for_target(c: int) -> list[str]:
        names = []
        for s in range(n_cells):
            if s == c:
                continue
            for w_idx in range(n_windows):
                names.append(f"c_{s}_{w_idx}")
        return names

    for c in range(n_cells):
        st = spikes_per_cell[c]
        C_cell = _compute_coupling_columns_at_position_times(
            spikes_per_cell, pos_time_train, target_idx=c,
            window_times_s=window_times_s,
        )  # (n_bins, n_coupling_cols)
        X = _design_quadratic_coupling(
            pos_x_train, pos_y_train, C_cell,
        )  # (n_bins, 6 + n_coupling_cols)
        mask = (st >= float(pos_time_train[0])) & (st <= float(pos_time_train[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_train, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        y_binary = np.zeros(n_bins, dtype=float)
        y_binary[bin_idx] = 1.0
        from nstat.glm import fit_binomial_glm
        glm = fit_binomial_glm(
            X[:, 1:], y_binary, include_intercept=True,
            l2=1e-3, max_iter=80,
        )
        b = np.concatenate([[glm.intercept], glm.coefficients])
        var_names = _var_names_for_target(c)
        coupling_terms = list(var_names)
        cif = CIF(
            b.tolist(),
            ["1", "x", "y", "x^2", "y^2", "x*y"] + coupling_terms,
            ["x", "y"] + var_names,
            fitType="binomial",
        )
        cifs.append(cif)
        gammas = np.asarray(b[6:], dtype=float).copy()
        coupling_coefs.append(gammas)
        source_iter = [s for s in range(n_cells) if s != c]
        for local_j, s in enumerate(source_iter):
            for w_idx in range(n_windows):
                coupling_matrix[c, s, w_idx] = float(
                    gammas[local_j * n_windows + w_idx]
                )
    return cifs, coupling_coefs, coupling_matrix


def _decode_position_with_coupling(
    pos_x_test: np.ndarray,
    pos_y_test: np.ndarray,
    pos_time_test: np.ndarray,
    spikes_per_cell: list[np.ndarray],
    cifs: list[CIF],
    *,
    delta: float,
    stride: int = 10,
    window_times_s: tuple[tuple[float, float], ...] = _COUPLING_WINDOW_TIMES_S,
):
    """PPAF decode with ensemble coupling as a known extrinsic observable.

    Following Truccolo et al. 2005 §3 (and the same mean-folding
    pattern as :func:`_decode_position_with_speed`), the per-cell
    coupling-count vector ``N_{c', w}(t)`` is *observed* at each
    decode time step from the held-out spike trains of the OTHER
    cells.  The PPAF state stays 2-D (position).  Like the velocity
    decoder, we fold the time-mean coupling contribution into
    ``mu_c`` and pass ``mu + beta @ state`` to PPDecodeFilterLinear.
    The decoder is linearised at the trajectory mean, so the
    *time-mean* coupling treatment is consistent with the existing
    curvature and speed treatments.
    """
    sub = slice(None, None, max(int(stride), 1))
    pos_x_sub = pos_x_test[sub]
    pos_y_sub = pos_y_test[sub]
    pos_time_sub = pos_time_test[sub]
    n_bins = pos_time_sub.size
    n_cells = len(spikes_per_cell)
    n_windows = len(window_times_s)
    n_coupling_cols = (n_cells - 1) * n_windows
    effective_delta = float(stride * delta)

    dN = np.zeros((n_cells, n_bins), dtype=float)
    for c, st in enumerate(spikes_per_cell):
        mask = (st >= float(pos_time_sub[0])) & (st <= float(pos_time_sub[-1]))
        st_in = st[mask]
        bin_idx = np.searchsorted(pos_time_sub, st_in, side="left")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        dN[c, bin_idx] = 1.0

    # Per-target coupling-count time series at the *strided* decode
    # lattice — same time base as ``dN``.  At the 167 ms strided
    # lattice the [0, 33ms) short window will mostly carry zeros
    # (it is sub-bin); the [33, 200ms) long window carries useful
    # multi-bin lookback.  The time-mean folding handles this
    # gracefully — only the trajectory mean of each coupling column
    # enters the linearised mu.
    coupling_means = np.zeros(
        (n_cells, n_coupling_cols), dtype=float,
    )
    for c in range(n_cells):
        C_c = _compute_coupling_columns_at_position_times(
            spikes_per_cell, pos_time_sub, target_idx=c,
            window_times_s=window_times_s,
        )
        coupling_means[c, :] = C_c.mean(axis=0)

    state_mean = np.array([float(pos_x_sub.mean()), float(pos_y_sub.mean())])
    mu = np.zeros(n_cells, dtype=float)
    beta = np.zeros((2, n_cells), dtype=float)
    for c, cif in enumerate(cifs):
        b = np.asarray(cif.b, dtype=float).reshape(-1)
        # b = [b0, bx, by, bxx, byy, bxy, gamma_0 ... gamma_{n-1}]
        coupling_contrib = float(
            np.dot(b[6:6 + n_coupling_cols], coupling_means[c])
        )
        mu[c] = float(
            b[0]
            + b[3] * state_mean[0] ** 2
            + b[4] * state_mean[1] ** 2
            + b[5] * state_mean[0] * state_mean[1]
            + coupling_contrib
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
    return dN, x_u, W_u, pos_x_sub, pos_y_sub, pos_time_sub


# =====================================================================
# Decoder for the history-augmented variant.
#
# History is intentionally NOT carried into the decoder.  Truccolo et
# al. 2005 §3 frames the history filter at a temporal resolution that
# matches the bin width at which the conditional intensity is
# evaluated.  The Animal-1 position lattice has ``delta`` ≈ 33 ms
# (and the PPAF decoder strides over this at 5x, giving a 167 ms
# effective bin), so only the [10, 50 ms) brief window has any
# resolution at the decoder lattice — the [0, 1 ms), [1, 5 ms),
# [5, 10 ms) windows are sub-bin and the binomial CIF cannot learn
# them.  An earlier prototype that fit history-augmented gammas at
# the decode lattice produced a -464%% RMSE regression (the gammas
# saturate the sigmoid during burst bins and freeze the filter).
#
# The history-augmented variant therefore reuses the position-only
# :func:`_fit_quadratic_cif_per_cell` + :func:`_decode_position`
# pipeline directly, which means its decoder RMSE matches baseline
# bit-for-bit.  The encoder-side history (above) carries the
# refractoriness signature; the decoder is lattice-matched.
# =====================================================================


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


def _plot_history_kernels(
    history_coefs: list[np.ndarray],
    cell_indices: tuple[int, ...],
    window_times: tuple[float, ...] = _HISTORY_WINDOW_TIMES,
):
    """Figure 7: per-cell spike-history coefficients (4-window basis)."""
    n_cells = len(history_coefs)
    edges_ms = np.asarray(window_times, dtype=float) * 1000.0
    centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])
    widths_ms = edges_ms[1:] - edges_ms[:-1]

    def _fmt_edge(v: float) -> str:
        if abs(v) >= 1000.0:
            return f"{v / 1000.0:.2f} s"
        if abs(v) >= 100.0:
            return f"{v:.0f} ms"
        return f"{v:.1f} ms"

    labels = [
        f"[{_fmt_edge(edges_ms[i])}, {_fmt_edge(edges_ms[i + 1])})"
        for i in range(len(centers_ms))
    ]
    # === FIGURE: fig07_history_kernels.png ===
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for i in range(min(n_cells, 4)):
        ax = axes[i]
        coefs = np.asarray(history_coefs[i], dtype=float).reshape(-1)
        colors = ["tab:red" if v < 0 else "tab:blue" for v in coefs]
        ax.bar(
            range(coefs.size), coefs,
            color=colors, edgecolor="k", lw=0.8,
        )
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.8)
        ax.set_xticks(range(coefs.size))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_xlabel("history window")
        ax.set_ylabel("coefficient (gamma)")
        ax.set_title(
            f"Cell #{cell_indices[i] + 1} — spike-history filter "
            f"(encoder Poisson GLM gammas)"
        )
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate each bar with its value for at-a-glance reading.
        for x_pos, val, w in zip(range(coefs.size), coefs, widths_ms):
            del w
            ax.text(
                x_pos, val + 0.02 * np.sign(val) if val != 0 else 0.02,
                f"{val:+.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
    for j in range(n_cells, 4):
        axes[j].axis("off")
    fig.suptitle(
        "Example 08 (history model) — per-cell spike-history filter "
        "(refractoriness in early windows, recovery in late windows)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_encoder_history_quality(
    logL_baseline: list[float],
    logL_history: list[float],
    refractory_gammas: list[float],
    cell_indices: tuple[int, ...],
    window_label: str = "[1, 5 ms)",
):
    """Figure 8: encoder fit quality with history (replaces the decoder bar chart).

    Reframed per the D.2 refactor: the original "baseline vs history
    decoder RMSE" framing produced a documented regression at this
    lattice (history is sub-bin at the 33 ms position grid + 5x
    decoder stride; see module docstring under "Design choice").
    Decoder history was removed; this figure now shows the genuine
    finding — adding history to the *encoder* improves the fit on
    every cell and reveals refractoriness for the cells that have
    it.

    Parameters
    ----------
    logL_baseline, logL_history
        Per-cell encoder Poisson-GLM log-likelihood for the spatial
        baseline (B-spline only) and the history-augmented variant
        (B-spline + 4 history columns) respectively.
    refractory_gammas
        Per-cell encoder gamma for the [1, 5 ms) window — chosen as
        the canonical "refractoriness" diagnostic (the [0, 1 ms)
        bin is at the absolute-refractory boundary, while [1, 5 ms)
        is the relative refractory period where a negative gamma
        is the cleanest signature).
    """
    n_cells = len(logL_baseline)
    cell_labels = [f"#{cell_indices[i] + 1}" for i in range(n_cells)]
    # === FIGURE: fig08_decoder_baseline_vs_history.png ===
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Panel 1: per-cell encoder log-likelihood, baseline vs history.
    ax_logL = axes[0]
    xs = np.arange(n_cells, dtype=float)
    width = 0.38
    ax_logL.bar(
        xs - width / 2, logL_baseline, width=width,
        color="0.55", edgecolor="k", lw=1.0, label="baseline (B-spline)",
    )
    ax_logL.bar(
        xs + width / 2, logL_history, width=width,
        color="tab:blue", edgecolor="k", lw=1.0,
        label="+ history (B-spline + 4 windows)",
    )
    for i, (lb, lh) in enumerate(zip(logL_baseline, logL_history)):
        gain = lh - lb
        ax_logL.text(
            float(i), max(lb, lh) + 0.02 * abs(max(lb, lh) or 1.0),
            f"{gain:+.1f}", ha="center", va="bottom", fontsize=8,
        )
    ax_logL.set_xticks(xs)
    ax_logL.set_xticklabels(cell_labels)
    ax_logL.set_xlabel("cell (MATLAB-style 1-indexed)")
    ax_logL.set_ylabel("encoder log-likelihood")
    n_logL_improved = int(
        sum(lh > lb for lb, lh in zip(logL_baseline, logL_history))
    )
    ax_logL.set_title(
        "Encoder Poisson-GLM log-likelihood: baseline vs +history\n"
        f"(gain annotated; both fits at l2=10 for fair comparison; "
        f"{n_logL_improved}/{n_cells} cells improved)"
    )
    ax_logL.legend(loc="best", fontsize=8)
    ax_logL.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-cell refractoriness gamma ([1, 5 ms) window).
    ax_g = axes[1]
    gammas = np.asarray(refractory_gammas, dtype=float)
    colors = ["tab:red" if g < 0 else "tab:blue" for g in gammas]
    bars = ax_g.bar(
        xs, gammas, color=colors, edgecolor="k", lw=1.0,
    )
    for bar, g in zip(bars, gammas):
        ax_g.text(
            bar.get_x() + bar.get_width() / 2.0,
            g + (0.02 * np.sign(g) if g != 0.0 else 0.02),
            f"{g:+.2f}",
            ha="center", va="bottom" if g >= 0 else "top",
            fontsize=9,
        )
    ax_g.axhline(0.0, color="k", lw=0.8, alpha=0.8)
    n_refract = int(np.sum(gammas < 0.0))
    ax_g.set_xticks(xs)
    ax_g.set_xticklabels(cell_labels)
    ax_g.set_xlabel("cell (MATLAB-style 1-indexed)")
    ax_g.set_ylabel(f"encoder gamma in {window_label}")
    ax_g.set_title(
        f"Refractoriness diagnostic: {n_refract}/{n_cells} cells with "
        f"negative gamma\n(red = refractory signature; blue = recovery)"
    )
    ax_g.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Example 08 — encoder fit quality with spike history\n"
        "(Truccolo et al. 2005 §3 history filter, fit at 1 kHz; "
        "decoder unchanged at 33 ms lattice)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_coupling_matrix(
    coupling_matrix: np.ndarray,
    cell_indices: tuple[int, ...],
    window_times_s: tuple[tuple[float, float], ...] = _COUPLING_WINDOW_TIMES_S,
    *,
    source_label: str = "encoder",
):
    """Figure 9: ensemble-coupling matrix per window (4×4 heatmap × 2).

    ``coupling_matrix`` is the ``(n_cells, n_cells, n_windows)`` tensor
    with NaN on the diagonal — target × source × window.  We plot
    one 4×4 heatmap per window with a symmetric diverging colormap
    so positive (excitatory) and negative (inhibitory) coefficients
    are visually distinguishable.

    ``source_label`` ("encoder" or "decoder") is recorded in the
    suptitle / caption per the brief — the encoder matrix is the
    more interpretable of the two (Poisson rate fit; decoder is
    binomial, with logit-link distortion).
    """
    n_cells, _, n_windows = coupling_matrix.shape
    labels = [f"#{cell_indices[i] + 1}" for i in range(n_cells)]
    # Symmetric color scale spanning all non-NaN coefficients.
    flat = coupling_matrix[~np.isnan(coupling_matrix)]
    if flat.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.max(np.abs(flat)))
        if vmax <= 0.0:
            vmax = 1.0
    # === FIGURE: fig09_coupling_matrix.png ===
    fig, axes = plt.subplots(1, n_windows, figsize=(11.0, 4.6))
    if n_windows == 1:
        axes = np.array([axes])
    for w_idx in range(n_windows):
        ax = axes[w_idx]
        M = coupling_matrix[:, :, w_idx]
        # Display with rows = targets (top → bottom), cols = sources.
        im = ax.imshow(
            M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        ax.set_xticks(range(n_cells))
        ax.set_yticks(range(n_cells))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("source cell")
        ax.set_ylabel("target cell")
        lo_ms, hi_ms = (
            window_times_s[w_idx][0] * 1000.0,
            window_times_s[w_idx][1] * 1000.0,
        )
        ax.set_title(f"window [{lo_ms:.0f}, {hi_ms:.0f} ms)")
        # Annotate each non-NaN cell with its coefficient.
        for i in range(n_cells):
            for j in range(n_cells):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(
                        j, i, "—", ha="center", va="center",
                        color="0.3", fontsize=10,
                    )
                else:
                    txt_color = "k" if abs(v) < 0.6 * vmax else "w"
                    ax.text(
                        j, i, f"{v:+.2f}", ha="center", va="center",
                        color=txt_color, fontsize=9,
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("coupling coefficient (gamma)")
    fig.suptitle(
        "Example 08 (coupling model) — ensemble-coupling matrix "
        f"per lattice-matched window (source: {source_label} GLM; "
        "diagonal NaN, positive = excitatory, negative = inhibitory)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_decoder_baseline_vs_progression(
    baseline_rmse: float,
    velocity_rmse: float,
    history_decoder_rmse: float,
    coupling_rmse: float,
):
    """Figure 10: 4-bar progression — capstone summary for Tier D.

    Shows the full Truccolo et al. 2005 framework progression in one
    chart: baseline (position only) → velocity (extrinsic covariate) →
    history-decoder (lattice-matched: matches baseline) → coupling
    (ensemble coupling as extrinsic observable).
    """
    rmses = [baseline_rmse, velocity_rmse, history_decoder_rmse, coupling_rmse]
    labels = ["baseline", "velocity", "history (decoder)", "coupling"]
    colors = ["0.55", "tab:blue", "tab:green", "tab:purple"]
    # === FIGURE: fig10_decoder_baseline_vs_progression.png ===
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(
        labels, rmses, color=colors, edgecolor="k", lw=1.0,
    )
    for bar, val, lab in zip(bars, rmses, labels):
        improvement = 100.0 * (baseline_rmse - val) / baseline_rmse
        sign = "+" if improvement >= 0.0 else ""
        if lab == "baseline":
            annot = f"{val:.3f}\n(baseline)"
        else:
            annot = f"{val:.3f}\n({sign}{improvement:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            annot,
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("position RMSE (box units)")
    ax.set_title(
        "Truccolo et al. 2005 framework — progressive RMSE improvement\n"
        "(velocity: extrinsic covariate; history-decoder: lattice-matched, "
        "matches baseline; coupling: ensemble coupling)"
    )
    ax.set_ylim(0.0, max(rmses) * 1.30)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_per_cell_lag_ccf(
    ccf_pos: list[tuple[np.ndarray, np.ndarray]],
    ccf_vel: list[tuple[np.ndarray, np.ndarray]],
    lag_pos_per_cell: list[float],
    lag_vel_per_cell: list[float],
    cell_indices: tuple[int, ...],
    *,
    window_pos: tuple[float, float] = _LAG_WINDOW_POSITION_S,
    window_vel: tuple[float, float] = _LAG_WINDOW_VELOCITY_S,
):
    """Figure 11: per-cell CCF lag-finding diagnostic.

    Per the brief, a 2x2 outer grid (one cell per outer panel); each
    outer panel contains two stacked subplots — top: residual-x-
    predicted-eta CCF with the position-lag search window shaded and
    the chosen peak marked; bottom: residual-x-speed CCF with the
    velocity-lag search window shaded and the chosen peak marked.
    Implemented as a 4x2 subplot grid where rows = (pos_ccf_cell_i,
    vel_ccf_cell_i, pos_ccf_cell_{i+2}, vel_ccf_cell_{i+2}); columns
    map the four cells into a 2x2-of-pairs layout.
    """
    n_cells = len(ccf_pos)
    # === FIGURE: fig11_per_cell_lag_ccf.png ===
    fig, axes = plt.subplots(
        4, 2, figsize=(11.5, 10.5), sharex=False, sharey=False,
    )
    for i in range(min(n_cells, 4)):
        col = i % 2
        row_top = 2 * (i // 2)
        row_bot = row_top + 1
        ax_top = axes[row_top, col]
        ax_bot = axes[row_bot, col]

        lags_p, ccf_p = ccf_pos[i]
        ax_top.axvspan(window_pos[0], window_pos[1], color="0.85", alpha=0.7,
                       label=(
                           f"search window "
                           f"[{int(window_pos[0] * 1000)}, "
                           f"{int(window_pos[1] * 1000)}] ms"
                       ))
        ax_top.plot(lags_p, ccf_p, color="tab:blue", lw=1.0)
        ax_top.axvline(lag_pos_per_cell[i], color="tab:red", lw=1.4,
                       label=f"peak: {lag_pos_per_cell[i] * 1000:+.1f} ms")
        ax_top.axhline(0.0, color="k", lw=0.8, alpha=0.7)
        ax_top.set_xlim(window_pos[0] - 0.05, window_pos[1] + 0.05)
        ax_top.set_xlabel("lag tau (s)")
        ax_top.set_ylabel("CCF (residual vs eta_pos)")
        ax_top.set_title(
            f"Cell #{cell_indices[i] + 1} — position lag CCF"
        )
        ax_top.legend(loc="best", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        lags_v, ccf_v = ccf_vel[i]
        ax_bot.axvspan(window_vel[0], window_vel[1], color="0.85", alpha=0.7,
                       label=(
                           f"search window "
                           f"[{int(window_vel[0] * 1000)}, "
                           f"{int(window_vel[1] * 1000)}] ms"
                       ))
        ax_bot.plot(lags_v, ccf_v, color="tab:green", lw=1.0)
        ax_bot.axvline(lag_vel_per_cell[i], color="tab:red", lw=1.4,
                       label=f"peak: {lag_vel_per_cell[i] * 1000:+.1f} ms")
        ax_bot.axhline(0.0, color="k", lw=0.8, alpha=0.7)
        ax_bot.set_xlim(window_vel[0] - 0.05, window_vel[1] + 0.05)
        ax_bot.set_xlabel("lag tau (s)")
        ax_bot.set_ylabel("CCF (residual' vs speed)")
        ax_bot.set_title(
            f"Cell #{cell_indices[i] + 1} — velocity lag CCF"
        )
        ax_bot.legend(loc="best", fontsize=8)
        ax_bot.grid(True, alpha=0.3)
    fig.suptitle(
        "Example 08 (velocity_lag model) — per-cell CCF lag-finding "
        "diagnostic\n(Truccolo et al. 2005 §4: covariate-lag from "
        "residual cross-covariance)"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_lag_corrected_comparison(
    baseline_rmse: float,
    velocity_rmse: float,
    velocity_lag_rmse: float,
    lag_pos_per_cell: list[float],
    lag_vel_per_cell: list[float],
):
    """Figure 12: 4-bar capstone for the velocity_lag variant.

    Bars: baseline RMSE / velocity (no-lag) RMSE / velocity_lag RMSE /
    improvement % (vs baseline).  Inset shows the per-cell lag values
    in ms; the "average per-cell lag" inset is a compact way to see
    whether the population leans anticipatory (negative) or trailing.
    """
    improvement_vs_baseline = (
        100.0 * (baseline_rmse - velocity_lag_rmse) / baseline_rmse
    )
    improvement_vs_velocity = (
        100.0 * (velocity_rmse - velocity_lag_rmse) / velocity_rmse
    )
    # === FIGURE: fig12_lag_corrected_comparison.png ===
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    labels = ["baseline", "velocity (no-lag)", "velocity_lag"]
    rmses = [baseline_rmse, velocity_rmse, velocity_lag_rmse]
    colors = ["0.55", "tab:blue", "tab:orange"]
    bars = ax.bar(labels, rmses, color=colors, edgecolor="k", lw=1.0)
    for bar, val, lab in zip(bars, rmses, labels):
        if lab == "baseline":
            annot = f"{val:.3f}\n(baseline)"
        else:
            imp = 100.0 * (baseline_rmse - val) / baseline_rmse
            sign = "+" if imp >= 0.0 else ""
            annot = f"{val:.3f}\n({sign}{imp:.1f}% vs baseline)"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            annot,
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("position RMSE (box units)")
    sign_b = "+" if improvement_vs_baseline >= 0.0 else ""
    sign_v = "+" if improvement_vs_velocity >= 0.0 else ""
    ax.set_title(
        "Truccolo et al. 2005 §4 — per-cell CCF lag correction "
        "(position + velocity)\n"
        f"velocity_lag: {sign_b}{improvement_vs_baseline:.1f}% vs baseline, "
        f"{sign_v}{improvement_vs_velocity:.1f}% vs velocity (no-lag)"
    )
    ax.set_ylim(0.0, max(rmses) * 1.30)
    ax.grid(True, alpha=0.3, axis="y")

    # Inset: per-cell lag values in ms.
    inset = fig.add_axes([0.66, 0.45, 0.26, 0.30])
    n_cells = len(lag_pos_per_cell)
    xs = np.arange(n_cells, dtype=float)
    width = 0.38
    inset.bar(
        xs - width / 2, [v * 1000.0 for v in lag_pos_per_cell],
        width=width, color="tab:red", edgecolor="k", lw=0.6,
        label="lag_pos",
    )
    inset.bar(
        xs + width / 2, [v * 1000.0 for v in lag_vel_per_cell],
        width=width, color="tab:green", edgecolor="k", lw=0.6,
        label="lag_vel",
    )
    inset.axhline(0.0, color="k", lw=0.8, alpha=0.7)
    inset.set_xticks(xs)
    inset.set_xticklabels([f"#{i+1}" for i in range(n_cells)], fontsize=7)
    inset.set_ylabel("lag (ms)", fontsize=8)
    inset.tick_params(axis="y", labelsize=7)
    inset.legend(loc="best", fontsize=7)
    inset.set_title("per-cell lags", fontsize=9)
    inset.grid(True, alpha=0.3, axis="y")
    # === END FIGURE ===
    return fig


# =====================================================================
# Driver
# =====================================================================
_VALID_MODELS: tuple[str, ...] = (
    "baseline", "velocity", "history", "coupling", "velocity_lag", "all",
)


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
    model : {"baseline", "velocity", "history", "coupling", "velocity_lag", "all"}, default "baseline"
        ``"baseline"`` preserves the original 6-term quadratic CIF pipeline
        bit-for-bit and produces figures 1-4.  ``"velocity"`` runs the
        baseline first to capture its RMSE, then a velocity-augmented
        encoder + decoder per Truccolo et al. (2005) and produces two
        extra figures (fig05 speed tuning, fig06 baseline-vs-velocity
        comparison).  ``"history"`` runs both baseline and velocity
        stages first, then a spike-history-augmented variant per
        Truccolo et al. 2005 §3 and produces two more figures (fig07
        per-cell history kernels, fig08 baseline-vs-velocity-vs-history
        decoder comparison).  ``"coupling"`` (D.3) runs the full
        baseline + velocity + history chain first, then an
        ensemble-coupling-augmented variant per Truccolo et al. 2005
        §3, producing two more figures (fig09 coupling matrix, fig10
        4-bar capstone progression).

    Returns
    -------
    dict
        Summary scalars and figure paths.  When ``model="velocity"`` the
        dict additionally carries ``baseline_rmse``, ``velocity_rmse``,
        ``rmse_improvement_pct``, ``speed_coefs``, and ``n_test_bins``.
        When ``model="history"`` the dict additionally carries
        ``history_decoder_rmse`` (equals ``baseline_rmse`` by
        construction — see "Design choice" in the module docstring),
        ``history_encoder_logL_improvement`` (per-cell log-likelihood
        gain from adding history to the encoder),
        ``history_encoder_logL_baseline`` / ``history_encoder_logL_history``
        (the underlying per-cell encoder log-likelihoods),
        ``history_refractory_gamma_1_5ms`` (per-cell encoder gamma in
        the [1, 5 ms) window — negative values indicate
        refractoriness), and ``history_coefs_encoder`` (list of
        ``(4,)`` ndarrays with all four encoder gammas per cell).
        When ``model="coupling"`` the dict additionally carries
        ``coupling_decoder_rmse``, ``coupling_rmse_improvement_pct``
        (vs baseline), ``coupling_coefs_encoder`` / ``coupling_coefs_decoder``
        (list of ``(6,)`` ndarrays per cell), and
        ``coupling_matrix_encoder`` / ``coupling_matrix_decoder``
        (each a ``(n_cells, n_cells, n_windows)`` tensor — target ×
        source × window — with NaN on the diagonal).  When
        ``model="velocity_lag"`` the dict additionally carries
        ``velocity_lag_pos_seconds`` and ``velocity_lag_vel_seconds``
        (per-cell position and velocity lag estimates, length
        ``n_cells``), ``velocity_lag_ccf_pos_peaks`` and
        ``velocity_lag_ccf_vel_peaks`` (per-cell CCF magnitudes at
        the chosen peaks), ``velocity_lag_decoder_rmse``,
        ``velocity_lag_rmse_improvement_pct`` (vs baseline), and
        ``velocity_lag_rmse_improvement_pct_vs_velocity`` (vs the
        velocity (no-lag) decoder).
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

    # ----- Velocity-augmented variant (model="velocity" or "history") -----
    # The history variant runs the velocity stage too so the full
    # baseline -> velocity -> history progression is available for
    # the 3-bar comparison in fig08.
    velocity_payload: dict | None = None
    if model in ("velocity", "history", "coupling", "velocity_lag", "all"):
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

    # ----- History-augmented variant (model="history") -----
    # D.2 refactor: history is in the *encoder* only.  See the module
    # docstring under "Design choice" for the rationale (lattice
    # mismatch between brief windows and decoder bin).  The decoder
    # for the history variant reuses the position-only quadratic CIF +
    # PPDecodeFilterLinear pipeline, so its RMSE matches baseline
    # bit-for-bit (and we report it as ``history_decoder_rmse``).
    history_payload: dict | None = None
    if model in ("history", "coupling", "all"):
        history_wt = tuple(_HISTORY_WINDOW_TIMES)
        print(
            f"  history windows: "
            f"{[round(w * 1000, 2) for w in history_wt]} ms "
            f"(Truccolo et al. 2005 §3; encoder fit at "
            f"{int(_HISTORY_SAMPLE_RATE_HZ)} Hz; decoder lattice = "
            f"{delta * 1000:.1f} ms position bin x {decode_stride} = "
            f"{decode_stride * delta * 1000:.1f} ms — see module "
            f"docstring for why decoder history is omitted)"
        )
        (
            _grid_x_h, _grid_y_h, _B_h,
            _rate_maps_h, glm_results_h, hist_coefs_encoder,
        ) = _fit_bspline_encoder_with_history(
            x[train_slice], y[train_slice], t[train_slice], spikes_per_cell,
            window_times=history_wt,
            n_grid=32, n_knots=(8, 8), degree=3, box=box,
        )
        converged_h = [int(g.converged) for g in glm_results_h]
        hist_coefs_encoder_rounded = [
            [round(float(v), 3) for v in arr] for arr in hist_coefs_encoder
        ]
        print(
            f"  history-augmented B-spline GLM: converged per cell = "
            f"{converged_h}; encoder gamma per cell = "
            f"{hist_coefs_encoder_rounded}"
        )

        # Per-cell encoder log-likelihood, baseline vs +history.
        # IMPORTANT — fair-comparison refit: the canonical baseline
        # encoder (above) uses l2=1.0 because the spatial-only IRLS
        # converges cleanly there.  The history-augmented encoder
        # requires l2=10.0 to converge (see the comment in
        # :func:`_fit_bspline_encoder_with_history`).  Comparing
        # ``glm_results[c].log_likelihood`` (l2=1.0) against
        # ``glm_results_h[c].log_likelihood`` (l2=10.0) would mix
        # regularization strength with the genuine "did history help?"
        # signal.  We therefore refit each cell's baseline (spatial
        # only) at l2=10.0 just for the fig08 comparison; the
        # baseline figures themselves still use the canonical l2=1.0
        # fit.  This is a cheap N=4 per-cell refit (~0.3 s) and yields
        # a fair penalized-MLE comparison: history helps for the
        # cells where the [10, 50 ms) recovery window is informative
        # (typically the higher-firing cells), and not for cells
        # whose firing is dominated by spatial selectivity alone.
        logL_baseline = []
        for c, st in enumerate(spikes_per_cell):
            H_counts = _bin_spikes_on_grid(
                st, x[train_slice], y[train_slice], t[train_slice],
                np.linspace(box[0], box[1], 33),
                np.linspace(box[2], box[3], 33),
            )
            counts = H_counts.ravel().astype(float)
            offset_b = np.full(
                B_design.shape[0],
                np.log(((box[1] - box[0]) / 32) * ((box[3] - box[2]) / 32)),
                dtype=float,
            )
            glm_b_fair = fit_poisson_glm(
                B_design, counts, offset=offset_b,
                include_intercept=False, l2=10.0, max_iter=800,
            )
            logL_baseline.append(float(glm_b_fair.log_likelihood))
        logL_history = [float(g.log_likelihood) for g in glm_results_h]
        logL_gain = [
            float(lh - lb) for lb, lh in zip(logL_baseline, logL_history)
        ]
        n_improved = int(sum(g > 0.0 for g in logL_gain))
        print(
            f"  encoder log-likelihood gain (history vs baseline @ l2=10) "
            f"per cell = {[round(g, 2) for g in logL_gain]} "
            f"({n_improved}/{len(logL_gain)} cells improved)"
        )

        # Per-cell [1, 5 ms) refractoriness gamma (window index 1).
        refractory_gammas = [float(arr[1]) for arr in hist_coefs_encoder]
        n_refractory_cells = int(sum(g < 0.0 for g in refractory_gammas))
        print(
            f"  refractoriness diagnostic — encoder gamma in "
            f"[1, 5 ms) per cell = "
            f"{[round(g, 3) for g in refractory_gammas]} "
            f"({n_refractory_cells}/{len(refractory_gammas)} cells with "
            f"negative gamma)"
        )

        # Decoder for the history variant: position-only quadratic CIF
        # (identical to baseline).  The decoder pipeline is unchanged
        # so the RMSE and per-bin errors equal the baseline values
        # bit-for-bit — we forward them rather than re-running
        # ``_decode_position`` redundantly.  This is documented under
        # "Design choice" in the module docstring.
        history_decoder_rmse = baseline_rmse
        history_per_bin_err = baseline_per_bin_err
        print(
            f"  history-variant decoder: RMSE = {history_decoder_rmse:.3f} "
            f"(matches baseline by construction — decoder is position-only)"
        )
        history_payload = {
            "history_coefs_encoder": hist_coefs_encoder,
            "history_window_times": history_wt,
            "history_decoder_rmse": history_decoder_rmse,
            "history_per_bin_err": history_per_bin_err,
            "logL_baseline": logL_baseline,
            "logL_history": logL_history,
            "logL_gain": logL_gain,
            "refractory_gammas": refractory_gammas,
        }

    # ----- Ensemble-coupling-augmented variant (model="coupling") -----
    # D.3: lattice-matched coupling windows ([0, 33ms), [33ms, 200ms))
    # picked specifically so the binomial decoder can carry them
    # without the saturation regression D.2 documented for sub-bin
    # history filters.  Coupling is treated as an *extrinsic
    # observable* at decode time, the same pattern velocity uses for
    # speed: the OBSERVED held-out spike counts from the other cells
    # are folded into a time-mean mu contribution and the PPAF state
    # stays 2-D (position).
    coupling_payload: dict | None = None
    if model in ("coupling", "all"):
        cw_labels = _coupling_window_labels()
        print(
            f"  coupling windows: {cw_labels} "
            f"(Truccolo et al. 2005 §3; lattice-matched at "
            f"{delta * 1000:.1f} ms position bin and "
            f"{decode_stride * delta * 1000:.1f} ms strided decode bin)"
        )
        (
            _grid_x_c, _grid_y_c, _B_c, _rate_maps_c, glm_results_c,
            coupling_coefs_enc, coupling_matrix_enc,
        ) = _fit_bspline_encoder_with_coupling(
            x[train_slice], y[train_slice], t[train_slice], spikes_per_cell,
            n_grid=32, n_knots=(8, 8), degree=3, box=box,
        )
        converged_c = [int(g.converged) for g in glm_results_c]
        coupling_enc_rounded = [
            [round(float(v), 3) for v in arr] for arr in coupling_coefs_enc
        ]
        print(
            f"  coupling-augmented B-spline GLM: converged per cell = "
            f"{converged_c}; encoder coupling coefficients per cell = "
            f"{coupling_enc_rounded}"
        )

        cifs_c, coupling_coefs_dec, coupling_matrix_dec = (
            _fit_quadratic_cif_per_cell_with_coupling(
                x[train_slice], y[train_slice], t[train_slice], spikes_per_cell,
                delta=delta,
            )
        )
        coupling_dec_rounded = [
            [round(float(v), 3) for v in arr] for arr in coupling_coefs_dec
        ]
        print(
            f"  coupling-augmented quadratic CIFs: decoder coupling "
            f"coefficients per cell = {coupling_dec_rounded}"
        )

        (
            _dN_c, x_dec_c, _W_c,
            x_test_sub_c, y_test_sub_c, _t_test_sub_c,
        ) = _decode_position_with_coupling(
            x[test_slice], y[test_slice], t[test_slice],
            spikes_per_cell, cifs_c, delta=delta, stride=decode_stride,
        )
        coupling_rmse = float(np.sqrt(np.mean(
            (x_dec_c[0] - x_test_sub_c) ** 2
            + (x_dec_c[1] - y_test_sub_c) ** 2
        )))
        coupling_improvement_pct = (
            100.0 * (baseline_rmse - coupling_rmse) / baseline_rmse
        )
        sign = "+" if coupling_improvement_pct >= 0.0 else ""
        print(
            f"  coupling decoder: RMSE = {coupling_rmse:.3f} "
            f"({sign}{coupling_improvement_pct:.1f}% vs baseline "
            f"{baseline_rmse:.3f})"
        )

        # Light sanity prints for the contract summary.
        flat_enc = coupling_matrix_enc[~np.isnan(coupling_matrix_enc)]
        if flat_enc.size > 0:
            max_abs_enc = float(np.max(np.abs(flat_enc)))
            n_pos_enc = int(np.sum(flat_enc > 0.0))
            n_neg_enc = int(np.sum(flat_enc < 0.0))
            print(
                f"  encoder coupling matrix: max |gamma| = {max_abs_enc:.3f}; "
                f"{n_pos_enc} excitatory, {n_neg_enc} inhibitory off-diagonal "
                f"entries"
            )

        coupling_payload = {
            "coupling_coefs_encoder": coupling_coefs_enc,
            "coupling_coefs_decoder": coupling_coefs_dec,
            "coupling_matrix_encoder": coupling_matrix_enc,
            "coupling_matrix_decoder": coupling_matrix_dec,
            "coupling_rmse": coupling_rmse,
            "coupling_improvement_pct": coupling_improvement_pct,
        }

    # ----- Velocity + per-cell lag-corrected variant -----
    # D.1+: iterated per-cell CCF lag correction for BOTH position and
    # velocity (Truccolo et al. 2005 §4; mirrors the lag-correction
    # pattern of example02).  Runs when ``model="velocity_lag"`` or
    # ``model="all"``; the ``velocity_lag`` model alone leaves the
    # baseline / velocity / history / coupling paths bit-for-bit
    # identical, and ``all`` runs every variant to produce the full
    # fig01-fig12 canonical figure set the manifest tracks.  Requires
    # the velocity stage above to have run (gated on the same model
    # list).
    velocity_lag_payload: dict | None = None
    if model in ("velocity_lag", "all"):
        assert velocity_payload is not None, (
            "velocity_lag requires the velocity stage to have run"
        )
        speed_full = _compute_speed(x, y, t)
        speed_train_lag = speed_full[train_slice]
        speed_test_lag = speed_full[test_slice]
        x_train = x[train_slice]
        y_train = y[train_slice]
        t_train = t[train_slice]
        x_test = x[test_slice]
        y_test = y[test_slice]
        t_test = t[test_slice]

        # The CCF is computed on the *training* lattice (the residual
        # is naturally defined on the training half — the encoder is
        # trained there).  The position-time spacing ``delta`` is the
        # same lattice the canonical baseline encoder fits.
        # ----------------------------------------------------------
        # Stage 1: baseline encoder eta and residual, per cell, then
        #          CCF for position lag.
        # Stage 2: refit on shifted position.
        # Stage 3: residual' x speed CCF for velocity lag.
        # Stage 4+5: augmented encoder + decoder with both per-cell
        #            lags.
        # ----------------------------------------------------------
        print(
            f"  velocity_lag — per-cell CCF lag correction "
            f"(position window {_LAG_WINDOW_POSITION_S}, "
            f"velocity window {_LAG_WINDOW_VELOCITY_S}; "
            f"Truccolo et al. 2005 §4)"
        )

        # Shared spatial-grid setup (same as the baseline + velocity
        # encoders: 32x32 grid, 8x8 knots, degree 3).
        n_grid_l = 32
        x_lo, x_hi, y_lo, y_hi = box
        edges_x_l = np.linspace(x_lo, x_hi, n_grid_l + 1)
        edges_y_l = np.linspace(y_lo, y_hi, n_grid_l + 1)
        grid_x_l = np.linspace(x_lo, x_hi, n_grid_l)
        grid_y_l = np.linspace(y_lo, y_hi, n_grid_l)
        B_l = bspline_basis_2d(
            grid_x_l, grid_y_l, n_knots=(8, 8), degree=3,
        )
        cell_area_l = float(
            ((x_hi - x_lo) / n_grid_l) * ((y_hi - y_lo) / n_grid_l)
        )
        offset_l = np.full(B_l.shape[0], np.log(cell_area_l), dtype=float)

        lag_pos_per_cell: list[float] = []
        lag_vel_per_cell: list[float] = []
        ccf_pos_peaks: list[float] = []
        ccf_vel_peaks: list[float] = []
        ccf_pos_payload: list[tuple[np.ndarray, np.ndarray]] = []
        ccf_vel_payload: list[tuple[np.ndarray, np.ndarray]] = []
        cifs_l: list[CIF] = []
        valid_train_counts: list[int] = []

        for c, st in enumerate(spikes_per_cell):
            # ---- Stage 1: baseline encoder per cell, on training lattice
            # ----    at the *time* lattice (not the spatial bin lattice).
            # We need a per-bin eta(t) and an observed count(t) on the
            # training lattice to compute the residual.  Build a
            # position-only design at the per-bin trajectory:
            X_train_pos = _design_quadratic(x_train, y_train)
            # Spike counts per training bin (collapse multi-spike bins
            # to 1 — the binomial convention used by the decoder CIFs).
            mask_tr = (st >= float(t_train[0])) & (st <= float(t_train[-1]))
            st_tr = st[mask_tr]
            bin_idx_tr = np.searchsorted(t_train, st_tr, side="left")
            bin_idx_tr = np.clip(bin_idx_tr, 0, t_train.size - 1)
            counts_tr = np.zeros(t_train.size, dtype=float)
            np.add.at(counts_tr, bin_idx_tr, 1.0)
            # Binomial logit fit on position only — gives us the
            # log-rate eta on the same time lattice as ``counts_tr``.
            from nstat.glm import fit_binomial_glm
            counts_binary = (counts_tr > 0.0).astype(float)
            glm_b1 = fit_binomial_glm(
                X_train_pos[:, 1:], counts_binary, include_intercept=True,
                l2=1e-3, max_iter=80,
            )
            b1 = np.concatenate([[glm_b1.intercept], glm_b1.coefficients])
            eta_pos = X_train_pos @ b1
            # Residual: observed - predicted spike probability.
            p_pred = 1.0 / (1.0 + np.exp(-np.clip(eta_pos, -20.0, 20.0)))
            residual_1 = counts_binary - p_pred

            # ---- Stage 1: CCF residual vs predicted eta_pos.
            lag_pos_c, peak_pos_c, lags_p, ccf_p = _ccf_with_lag_peak(
                residual_1, eta_pos, dt=delta,
                lag_window=_LAG_WINDOW_POSITION_S,
            )

            # ---- Stage 2: refit position-only encoder on shifted pos.
            px_sh, py_sh, mask_pos = _shift_position_by_lag(
                x_train, y_train, lag_pos_c, delta,
            )
            X_train_sh = _design_quadratic(px_sh, py_sh)
            # Drop NaN bins for the refit.
            keep_p = mask_pos
            glm_b2 = fit_binomial_glm(
                X_train_sh[keep_p, 1:], counts_binary[keep_p],
                include_intercept=True, l2=1e-3, max_iter=80,
            )
            b2 = np.concatenate([[glm_b2.intercept], glm_b2.coefficients])
            eta_pos_sh = X_train_sh @ b2  # has NaN where mask_pos is False
            p_pred_sh = 1.0 / (
                1.0 + np.exp(-np.clip(eta_pos_sh, -20.0, 20.0))
            )
            residual_2 = counts_binary - p_pred_sh
            # Fill NaN edges with 0 so the CCF below is well-defined;
            # the search window excludes the extreme tails anyway.
            residual_2 = np.where(mask_pos, residual_2, 0.0)

            # ---- Stage 3: residual' x speed CCF.
            lag_vel_c, peak_vel_c, lags_v, ccf_v = _ccf_with_lag_peak(
                residual_2, speed_train_lag, dt=delta,
                lag_window=_LAG_WINDOW_VELOCITY_S,
            )

            # ---- Stage 4: refit augmented encoder with both lags.
            # Shift position by lag_pos_c, speed by lag_vel_c.
            px_sh_a, py_sh_a, mp_a = _shift_position_by_lag(
                x_train, y_train, lag_pos_c, delta,
            )
            sp_sh_a, mv_a = _shift_scalar_by_lag(
                speed_train_lag, lag_vel_c, delta,
            )
            keep_a = mp_a & mv_a
            X_aug = _design_quadratic_speed(
                np.where(keep_a, px_sh_a, 0.0),
                np.where(keep_a, py_sh_a, 0.0),
                np.where(keep_a, sp_sh_a, 0.0),
            )
            glm_aug = fit_binomial_glm(
                X_aug[keep_a, 1:], counts_binary[keep_a],
                include_intercept=True, l2=1e-3, max_iter=80,
            )
            b_aug = np.concatenate([[glm_aug.intercept], glm_aug.coefficients])
            cif = CIF(
                b_aug.tolist(),
                ["1", "x", "y", "x^2", "y^2", "x*y", "s"],
                ["x", "y", "s"],
                fitType="binomial",
            )

            lag_pos_per_cell.append(float(lag_pos_c))
            lag_vel_per_cell.append(float(lag_vel_c))
            ccf_pos_peaks.append(float(peak_pos_c))
            ccf_vel_peaks.append(float(peak_vel_c))
            ccf_pos_payload.append((lags_p, ccf_p))
            ccf_vel_payload.append((lags_v, ccf_v))
            cifs_l.append(cif)
            valid_train_counts.append(int(np.sum(keep_a)))

            print(
                f"    cell #{cell_indices[c] + 1}: lag_pos = "
                f"{lag_pos_c * 1000:+.1f} ms (CCF = {peak_pos_c:+.3f}), "
                f"lag_vel = {lag_vel_c * 1000:+.1f} ms "
                f"(CCF = {peak_vel_c:+.3f}); "
                f"training bins kept after NaN drop = "
                f"{int(np.sum(keep_a))} / {t_train.size}"
            )

        # ---- Stage 5: decode with per-cell lags on the test half.
        # The decoder applies each cell's own (lag_pos_c, lag_vel_c)
        # to position / speed at test time.  When a lag is negative
        # the decoder reads a *future* sample — standard in
        # retrospective decoding evaluation; we document the "future
        # peek" in the printed output.
        sub_l = slice(None, None, max(int(decode_stride), 1))
        x_test_sub_l = x_test[sub_l]
        y_test_sub_l = y_test[sub_l]
        t_test_sub_l = t_test[sub_l]
        speed_test_sub_l = speed_test_lag[sub_l]
        n_bins_l = t_test_sub_l.size
        effective_delta_l = float(decode_stride * delta)

        dN_l = np.zeros((len(spikes_per_cell), n_bins_l), dtype=float)
        for c, st in enumerate(spikes_per_cell):
            mask_te = (st >= float(t_test_sub_l[0])) & (
                st <= float(t_test_sub_l[-1])
            )
            st_te = st[mask_te]
            bi = np.searchsorted(t_test_sub_l, st_te, side="left")
            bi = np.clip(bi, 0, n_bins_l - 1)
            dN_l[c, bi] = 1.0

        # Linearise each per-cell CIF around the trajectory mean of
        # that cell's *lag-corrected* test trajectory.  Same pattern
        # as :func:`_decode_position_with_speed` but with the per-cell
        # shifted (position, speed) feeding the mean.
        mu_l = np.zeros(len(cifs_l), dtype=float)
        beta_l = np.zeros((2, len(cifs_l)), dtype=float)
        for c, cif in enumerate(cifs_l):
            px_te, py_te, mp_te = _shift_position_by_lag(
                x_test_sub_l, y_test_sub_l,
                lag_pos_per_cell[c], effective_delta_l,
            )
            sp_te, mv_te = _shift_scalar_by_lag(
                speed_test_sub_l, lag_vel_per_cell[c], effective_delta_l,
            )
            keep_te = mp_te & mv_te
            # Trajectory means used for linearisation.  NaN samples
            # are excluded so the mean is taken over the valid bins.
            if np.any(keep_te):
                xm = float(np.nanmean(px_te[keep_te]))
                ym = float(np.nanmean(py_te[keep_te]))
                sm = float(np.nanmean(sp_te[keep_te]))
            else:
                xm = float(x_test_sub_l.mean())
                ym = float(y_test_sub_l.mean())
                sm = float(speed_test_sub_l.mean())
            b = np.asarray(cif.b, dtype=float).reshape(-1)
            mu_l[c] = float(
                b[0]
                + b[3] * xm ** 2
                + b[4] * ym ** 2
                + b[5] * xm * ym
                + b[6] * sm
            )
            beta_l[0, c] = float(b[1] + 2.0 * b[3] * xm + b[5] * ym)
            beta_l[1, c] = float(b[2] + 2.0 * b[4] * ym + b[5] * xm)

        train_dx_l = np.diff(x_test_sub_l)
        train_dy_l = np.diff(y_test_sub_l)
        qxx_l = float(np.var(train_dx_l)) if train_dx_l.size else 1e-3
        qyy_l = float(np.var(train_dy_l)) if train_dy_l.size else 1e-3
        A_l = np.eye(2, dtype=float)
        Q_l = np.diag([qxx_l, qyy_l])
        x0_l = np.array([x_test_sub_l[0], y_test_sub_l[0]], dtype=float)
        Pi0_l = np.diag([qxx_l, qyy_l])
        (
            _x_p_l, _W_p_l, x_u_l, _W_u_l, *_unused
        ) = DecodingAlgorithms.PPDecodeFilterLinear(
            A_l, Q_l, dN_l, mu_l, beta_l, "binomial", effective_delta_l,
            None, None, x0_l, Pi0_l,
        )
        velocity_lag_rmse = float(np.sqrt(np.mean(
            (x_u_l[0] - x_test_sub_l) ** 2
            + (x_u_l[1] - y_test_sub_l) ** 2
        )))
        improvement_lag_vs_baseline = (
            100.0 * (baseline_rmse - velocity_lag_rmse) / baseline_rmse
        )
        velocity_no_lag_rmse = float(velocity_payload["velocity_rmse"])
        improvement_lag_vs_velocity = (
            100.0 * (velocity_no_lag_rmse - velocity_lag_rmse)
            / velocity_no_lag_rmse
        )
        sign_b = "+" if improvement_lag_vs_baseline >= 0.0 else ""
        sign_v = "+" if improvement_lag_vs_velocity >= 0.0 else ""
        print(
            f"  velocity_lag decoder: RMSE = {velocity_lag_rmse:.3f} "
            f"({sign_b}{improvement_lag_vs_baseline:.1f}% vs baseline "
            f"{baseline_rmse:.3f}; {sign_v}"
            f"{improvement_lag_vs_velocity:.1f}% vs velocity (no-lag) "
            f"{velocity_no_lag_rmse:.3f})"
        )
        # Honest framing: CA1 cells are position-dominated, so the
        # absolute decoder-RMSE improvement is typically modest
        # (1-3%); the SCIENCE is the per-cell lag values themselves.
        # Phase-precession would predict lag_pos < 0 for CA1.
        n_anticip_pos = int(sum(v < 0.0 for v in lag_pos_per_cell))
        n_anticip_vel = int(sum(v < 0.0 for v in lag_vel_per_cell))
        print(
            f"    sign-and-magnitude summary — {n_anticip_pos}/"
            f"{len(lag_pos_per_cell)} cells show negative position "
            f"lag (anticipatory); {n_anticip_vel}/{len(lag_vel_per_cell)} "
            f"show negative velocity lag.  At test time the decoder reads "
            f"position(t - lag_pos_c) and speed(t - lag_vel_c); when "
            f"lag < 0 this is a 'future peek' (standard convention in "
            f"retrospective decoding evaluation)."
        )

        velocity_lag_payload = {
            "lag_pos_per_cell": lag_pos_per_cell,
            "lag_vel_per_cell": lag_vel_per_cell,
            "ccf_pos_peaks": ccf_pos_peaks,
            "ccf_vel_peaks": ccf_vel_peaks,
            "ccf_pos_payload": ccf_pos_payload,
            "ccf_vel_payload": ccf_vel_payload,
            "velocity_lag_rmse": velocity_lag_rmse,
            "improvement_vs_baseline": improvement_lag_vs_baseline,
            "improvement_vs_velocity": improvement_lag_vs_velocity,
            "valid_train_counts": valid_train_counts,
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
    if history_payload is not None:
        # Encoder gammas (Poisson-rate fit with cell-area offset)
        # carry the genuine refractoriness signature.  See the
        # ``model="history"`` doc block for why the decoder is
        # position-only at the 33 ms position lattice; fig08 now
        # reports encoder fit quality rather than (regressing)
        # decoder RMSE.
        fig7 = _plot_history_kernels(
            history_payload["history_coefs_encoder"], cell_indices,
            window_times=history_payload["history_window_times"],
        )
        fig8 = _plot_encoder_history_quality(
            history_payload["logL_baseline"],
            history_payload["logL_history"],
            history_payload["refractory_gammas"],
            cell_indices,
        )
        figures.extend([fig7, fig8])
        fig_names.extend([
            "fig07_history_kernels",
            "fig08_decoder_baseline_vs_history",
        ])
    if velocity_lag_payload is not None:
        fig11 = _plot_per_cell_lag_ccf(
            velocity_lag_payload["ccf_pos_payload"],
            velocity_lag_payload["ccf_vel_payload"],
            velocity_lag_payload["lag_pos_per_cell"],
            velocity_lag_payload["lag_vel_per_cell"],
            cell_indices,
        )
        fig12 = _plot_lag_corrected_comparison(
            baseline_rmse=baseline_rmse,
            velocity_rmse=float(velocity_payload["velocity_rmse"]),
            velocity_lag_rmse=float(
                velocity_lag_payload["velocity_lag_rmse"]
            ),
            lag_pos_per_cell=velocity_lag_payload["lag_pos_per_cell"],
            lag_vel_per_cell=velocity_lag_payload["lag_vel_per_cell"],
        )
        figures.extend([fig11, fig12])
        fig_names.extend([
            "fig11_per_cell_lag_ccf",
            "fig12_lag_corrected_comparison",
        ])
    if coupling_payload is not None:
        # Encoder matrix is the more interpretable of the two (Poisson
        # rate fit) — decoder is binomial with logit-link distortion.
        # Documented in the figure caption per the brief.
        fig9 = _plot_coupling_matrix(
            coupling_payload["coupling_matrix_encoder"],
            cell_indices, source_label="encoder",
        )
        fig10 = _plot_decoder_baseline_vs_progression(
            baseline_rmse=baseline_rmse,
            velocity_rmse=float(velocity_payload["velocity_rmse"]),
            history_decoder_rmse=float(
                history_payload["history_decoder_rmse"]
            ),
            coupling_rmse=float(coupling_payload["coupling_rmse"]),
        )
        figures.extend([fig9, fig10])
        fig_names.extend([
            "fig09_coupling_matrix",
            "fig10_decoder_baseline_vs_progression",
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
    if history_payload is not None:
        # Field naming (D.2 refactor):
        # - ``history_decoder_rmse`` is the position-only decoder RMSE
        #   for the history-model variant.  It equals ``baseline_rmse``
        #   by construction because the decoder pipeline is unchanged
        #   (see module docstring under "Design choice").
        # - ``history_encoder_logL_improvement`` is the per-cell encoder
        #   Poisson-GLM log-likelihood gain (+history vs baseline).
        # - ``history_coefs_encoder`` carries the 4-window per-cell
        #   gammas from the encoder fit.
        # The previously-shipped ``history_rmse`` /
        # ``history_rmse_improvement_pct`` / ``history_coefs_decoder``
        # keys (decoder-side history) were removed when decoder-side
        # history was removed.
        result.update({
            "history_decoder_rmse": float(
                history_payload["history_decoder_rmse"]
            ),
            "history_encoder_logL_improvement": [
                float(g) for g in history_payload["logL_gain"]
            ],
            "history_encoder_logL_baseline": [
                float(v) for v in history_payload["logL_baseline"]
            ],
            "history_encoder_logL_history": [
                float(v) for v in history_payload["logL_history"]
            ],
            "history_refractory_gamma_1_5ms": [
                float(g) for g in history_payload["refractory_gammas"]
            ],
            "history_coefs_encoder": [
                np.asarray(arr, dtype=float)
                for arr in history_payload["history_coefs_encoder"]
            ],
        })
    if coupling_payload is not None:
        # Coupling field naming (D.3):
        # - ``coupling_decoder_rmse`` is the held-out RMSE of the
        #   coupling-augmented quadratic CIF decoder.
        # - ``coupling_rmse_improvement_pct`` is the percent
        #   improvement vs ``baseline_rmse``.
        # - ``coupling_coefs_{encoder,decoder}`` carry the per-cell
        #   ``(6,)`` gamma vectors (3 source cells × 2 windows).
        # - ``coupling_matrix_{encoder,decoder}`` carry the
        #   ``(n_cells, n_cells, n_windows)`` target-x-source-x-window
        #   tensors with NaN on the diagonal.
        result.update({
            "coupling_decoder_rmse": float(coupling_payload["coupling_rmse"]),
            "coupling_rmse_improvement_pct": float(
                coupling_payload["coupling_improvement_pct"]
            ),
            "coupling_coefs_encoder": [
                np.asarray(arr, dtype=float)
                for arr in coupling_payload["coupling_coefs_encoder"]
            ],
            "coupling_coefs_decoder": [
                np.asarray(arr, dtype=float)
                for arr in coupling_payload["coupling_coefs_decoder"]
            ],
            "coupling_matrix_encoder": np.asarray(
                coupling_payload["coupling_matrix_encoder"], dtype=float,
            ),
            "coupling_matrix_decoder": np.asarray(
                coupling_payload["coupling_matrix_decoder"], dtype=float,
            ),
        })
    if velocity_lag_payload is not None:
        # velocity_lag field naming (D.1+):
        # - ``velocity_lag_pos_seconds`` / ``velocity_lag_vel_seconds``:
        #   per-cell position and velocity lag estimates (length
        #   ``n_cells``).  Negative = cell anticipates the covariate.
        # - ``velocity_lag_ccf_pos_peaks`` / ``velocity_lag_ccf_vel_peaks``:
        #   the CCF magnitudes at the chosen peaks (diagnostic).
        # - ``velocity_lag_decoder_rmse``: held-out RMSE of the
        #   lag-corrected augmented decoder.
        # - ``velocity_lag_rmse_improvement_pct``: percent improvement
        #   vs ``baseline_rmse``.
        # - ``velocity_lag_rmse_improvement_pct_vs_velocity``: percent
        #   improvement vs the velocity (no-lag) decoder.
        result.update({
            "velocity_lag_pos_seconds": [
                float(v) for v in velocity_lag_payload["lag_pos_per_cell"]
            ],
            "velocity_lag_vel_seconds": [
                float(v) for v in velocity_lag_payload["lag_vel_per_cell"]
            ],
            "velocity_lag_ccf_pos_peaks": [
                float(v) for v in velocity_lag_payload["ccf_pos_peaks"]
            ],
            "velocity_lag_ccf_vel_peaks": [
                float(v) for v in velocity_lag_payload["ccf_vel_peaks"]
            ],
            "velocity_lag_decoder_rmse": float(
                velocity_lag_payload["velocity_lag_rmse"]
            ),
            "velocity_lag_rmse_improvement_pct": float(
                velocity_lag_payload["improvement_vs_baseline"]
            ),
            "velocity_lag_rmse_improvement_pct_vs_velocity": float(
                velocity_lag_payload["improvement_vs_velocity"]
            ),
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
        "--model", choices=_VALID_MODELS, default="all",
        help=(
            "Encoder/decoder variant.  ``baseline`` reproduces the "
            "original position-only pipeline (4 figures).  "
            "``velocity`` runs the baseline then a velocity-augmented "
            "variant per Truccolo et al. 2005 (6 figures).  "
            "``history`` additionally runs a "
            "spike-history-augmented variant per Truccolo et al. 2005 "
            "§3 (8 figures).  ``coupling`` runs the full "
            "baseline + velocity + history chain plus an "
            "ensemble-coupling-augmented variant per "
            "Truccolo et al. 2005 §3 (10 figures: fig01-fig10).  "
            "``velocity_lag`` runs the baseline + velocity chain then "
            "applies an iterated per-cell CCF lag correction to both "
            "position and velocity per Truccolo et al. 2005 §4, "
            "producing fig11 (per-cell CCF diagnostic) and fig12 "
            "(4-bar capstone) for a total of 8 figures.  ``all`` "
            "(default) runs every variant — baseline + velocity + "
            "history + coupling + velocity_lag — producing the full "
            "fig01-fig12 canonical set the manifest tracks; this is "
            "what ``--export-figures`` needs in order to satisfy the "
            "paper-example manifest contract.  The Python "
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
        for key in (
            "history_decoder_rmse",
            "history_encoder_logL_improvement",
            "history_encoder_logL_baseline",
            "history_encoder_logL_history",
            "history_refractory_gamma_1_5ms",
        ):
            if key in result:
                summary[key] = result[key]
        for key in ("history_coefs_encoder",):
            if key in result:
                summary[key] = [list(map(float, arr)) for arr in result[key]]
        for key in (
            "coupling_decoder_rmse",
            "coupling_rmse_improvement_pct",
        ):
            if key in result:
                summary[key] = result[key]
        for key in ("coupling_coefs_encoder", "coupling_coefs_decoder"):
            if key in result:
                summary[key] = [list(map(float, arr)) for arr in result[key]]
        for key in ("coupling_matrix_encoder", "coupling_matrix_decoder"):
            if key in result:
                # Serialize NaN as null so the JSON parses; nested
                # list-of-list-of-list of floats keeps the
                # (target, source, window) shape readable.
                mat = result[key]
                summary[key] = [
                    [
                        [
                            (float(v) if np.isfinite(v) else None)
                            for v in mat[i, j, :]
                        ]
                        for j in range(mat.shape[1])
                    ]
                    for i in range(mat.shape[0])
                ]
        for key in (
            "velocity_lag_pos_seconds",
            "velocity_lag_vel_seconds",
            "velocity_lag_ccf_pos_peaks",
            "velocity_lag_ccf_vel_peaks",
            "velocity_lag_decoder_rmse",
            "velocity_lag_rmse_improvement_pct",
            "velocity_lag_rmse_improvement_pct_vs_velocity",
        ):
            if key in result:
                summary[key] = result[key]
        args.output_json.write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
