r"""High-level place-field encoding + decoding wrapper.

Encapsulates the canonical pattern from example08_real_place_cells.py:
fit a B-spline Poisson GLM per cell (encoding), refit a quadratic CIF per
cell (so PPDecode_update can call CIF.evalLambdaDelta efficiently), and
run PPDecodeFilterLinear with the history-free fast path (PR #198).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nstat import CIF, DecodingAlgorithms, Trial
from nstat.extras.spatial.basis import bspline_basis_2d
from nstat.glm import fit_binomial_glm, fit_poisson_glm


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceFieldDecoderConfig:
    """Configuration for the place-field encoding + decoding wrapper.

    Parameters
    ----------
    bin_width_s : float
        Time-discretisation bin (seconds). Default 0.020 (20 ms — matches example08).
    n_basis_per_dim : int
        Number of B-spline basis functions per spatial dimension for encoding.
        Default 8 (8x8 = 64 coefficients per cell).
    spline_order : int
        B-spline order (degree + 1) for encoding. Default 4 (cubic).
    cif_kind : {"quadratic", "linear"}
        CIF family for decoding refit. Default "quadratic".
    decode_filter : {"linear", "nonlinear"}
        Which PPDecodeFilter variant to use. "linear" exercises the
        PR #198 O(C*T) fast path; "nonlinear" is provided for parity
        but is O(C*T^2) for history-free CIFs (i.e., always, in this
        wrapper, since none of the place-cell CIFs carry history).
        Default "linear".
    min_n_spikes_per_cell : int
        Skip cells with fewer total spikes; they cannot fit a stable
        place field. Default 10.
    """

    bin_width_s: float = 0.020
    n_basis_per_dim: int = 8
    spline_order: int = 4
    cif_kind: str = "quadratic"
    decode_filter: str = "linear"
    min_n_spikes_per_cell: int = 10

    def __post_init__(self) -> None:
        if not (self.bin_width_s > 0):
            raise ValueError(
                f"bin_width_s must be > 0; got {self.bin_width_s!r}"
            )
        if self.n_basis_per_dim < 2:
            raise ValueError(
                f"n_basis_per_dim must be >= 2; got {self.n_basis_per_dim!r}"
            )
        if self.spline_order < 1:
            raise ValueError(
                f"spline_order must be >= 1; got {self.spline_order!r}"
            )
        if self.cif_kind not in {"quadratic", "linear"}:
            raise ValueError(
                f"cif_kind must be one of {{'quadratic', 'linear'}}; "
                f"got {self.cif_kind!r}"
            )
        if self.decode_filter not in {"linear", "nonlinear"}:
            raise ValueError(
                f"decode_filter must be one of {{'linear', 'nonlinear'}}; "
                f"got {self.decode_filter!r}"
            )
        if self.min_n_spikes_per_cell < 0:
            raise ValueError(
                f"min_n_spikes_per_cell must be >= 0; got "
                f"{self.min_n_spikes_per_cell!r}"
            )


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceFieldDecoderResult:
    """Output of :func:`fit_place_field_decoder`.

    Attributes
    ----------
    decoded_position : (n_time, 2) ndarray
        Posterior mean position from PPAF filter.
    decoded_covariance : (n_time, 2, 2) ndarray
        Per-bin posterior covariance.
    decoding_error : (n_time,) ndarray
        Per-bin Euclidean distance between decoded and true position.
    mean_decoding_error : float
        Average decoding error (the headline summary).
    cell_indices_kept : list[int]
        Indices of cells that contributed (passed min_spike filter, GLM converged).
    cell_indices_skipped : list[int]
        Indices of cells that were dropped (with reason logged via UserWarning).
    spline_coefs : list[ndarray]
        B-spline coefficient vectors for kept cells; shape
        ``(n_basis_per_dim**2 + 1,)`` — first entry is the GLM intercept.
    quadratic_coefs : list[ndarray]
        Quadratic CIF coefficient vectors for kept cells; shape (6,)
        ordered as ``[intercept, x, y, x^2, y^2, x*y]`` (or shape (3,)
        ordered as ``[intercept, x, y]`` when ``cif_kind == "linear"``).
    n_basis_per_dim : int
        Echoed from config.
    bin_width_s : float
        Echoed from config.
    """

    decoded_position: np.ndarray
    decoded_covariance: np.ndarray
    decoding_error: np.ndarray
    mean_decoding_error: float
    cell_indices_kept: list[int]
    cell_indices_skipped: list[int]
    spline_coefs: list[np.ndarray]
    quadratic_coefs: list[np.ndarray]
    n_basis_per_dim: int
    bin_width_s: float


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _quadratic_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[1, x, y, x^2, y^2, x*y] design matrix (T, 6) — see example08."""
    # NOTE: returns columns *without* intercept so fit_*_glm can add one.
    return np.column_stack([x, y, x * x, y * y, x * y])


def _linear_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[x, y] design matrix (T, 2) — fit_*_glm adds the intercept."""
    return np.column_stack([x, y])


def _validate_position(
    position: NDArray[np.float64], expected_n_time: int
) -> NDArray[np.float64]:
    """Coerce + shape-check the position array.

    Parameters
    ----------
    position
        Caller-supplied position array.
    expected_n_time
        Expected number of samples — derived from the trial's covariate
        sample grid by the wrapper, *not* from ``position.shape[0]``.
    """
    pos = np.asarray(position, dtype=float)
    if pos.ndim != 2:
        raise ValueError(
            f"position must be 2-D with shape (n_time, 2); got ndim={pos.ndim}, "
            f"shape={pos.shape}"
        )
    if pos.shape[1] != 2:
        raise ValueError(
            f"position must have 2 columns (x, y); got shape={pos.shape}"
        )
    if pos.shape[0] != expected_n_time:
        raise ValueError(
            f"position length ({pos.shape[0]}) does not match trial length "
            f"({expected_n_time}); the wrapper expects position aligned to "
            f"the trial covariate sample grid"
        )
    return pos


def _bin_spikes_to_bool(
    spike_times: np.ndarray,
    bin_edges: np.ndarray,
) -> NDArray[np.float64]:
    """Binarised spike train on ``bin_edges`` (length = bin_edges.size - 1)."""
    if spike_times.size == 0:
        return np.zeros(bin_edges.size - 1, dtype=float)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return (counts > 0).astype(float)


def _bin_spikes_to_counts(
    spike_times: np.ndarray,
    bin_edges: np.ndarray,
) -> NDArray[np.float64]:
    """Integer spike counts on ``bin_edges`` for Poisson encoding."""
    if spike_times.size == 0:
        return np.zeros(bin_edges.size - 1, dtype=float)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(float)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def fit_place_field_decoder(
    trial: Trial,
    position: NDArray[np.float64],
    *,
    config: PlaceFieldDecoderConfig | None = None,
) -> PlaceFieldDecoderResult:
    """Fit a place-cell encoding + decoding pipeline for a single Trial.

    Wraps the canonical 4-step pipeline from
    ``examples/paper/example08_real_place_cells.py`` (lines 270-420):

    1. Build a tensor-product 2-D B-spline basis on the trial's position
       range and fit a per-cell Poisson GLM (encoding).
    2. Refit a quadratic (or linear) CIF per cell on a position design
       matrix — the decoder needs an analytical CIF whose
       :meth:`CIF.evalLambdaDelta` is cheap.
    3. Build the per-cell :class:`CIF` collection with
       ``Xnames = ["1", "x", "y", "x^2", "y^2", "x*y"]``,
       ``stimNames = ["x", "y"]``, ``fitType = "binomial"``.
    4. Run :func:`DecodingAlgorithms.PPDecodeFilterLinear` (history-free
       fast path) on the same trial; compute per-bin Euclidean error.

    Parameters
    ----------
    trial
        Training trial. Must carry a SpikeTrainCollection.
    position
        Training-trial 2-D position over time, shape ``(n_time, 2)``,
        aligned to the trial's sample grid.
    config
        Configuration. Defaults to :class:`PlaceFieldDecoderConfig`.

    Returns
    -------
    PlaceFieldDecoderResult
    """
    cfg = config if config is not None else PlaceFieldDecoderConfig()

    # ---- 1. Set up the decoding time grid -------------------------------
    min_time = float(trial.minTime)
    max_time = float(trial.maxTime)
    if not (max_time > min_time):
        raise ValueError(
            f"trial has invalid time window [{min_time}, {max_time}]; "
            f"max_time must exceed min_time"
        )
    bin_edges = np.arange(
        min_time, max_time + cfg.bin_width_s, cfg.bin_width_s, dtype=float
    )
    # Guard against degenerate trials where arange overshoots.
    bin_edges = bin_edges[bin_edges <= max_time + 0.5 * cfg.bin_width_s]
    if bin_edges.size < 3:
        raise ValueError(
            f"trial too short for bin_width_s={cfg.bin_width_s} — only "
            f"{bin_edges.size - 1} bin(s) available"
        )
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = bin_centres.size

    # ---- 2. Validate position + resample onto the decoding grid ---------
    # The expected position length is the trial covariate sample grid
    # — after Trial resampling, every covariate shares the same time
    # base.  We compare against that explicitly so a caller passing
    # position at the *original* (pre-resample) rate gets a clear error
    # rather than a silent interpolation onto the wrong grid.
    expected_n_time = (
        int(trial.covarColl.getCov(0).time.size) if trial.covarColl.numCov else 0
    )
    if expected_n_time <= 0:
        raise ValueError(
            "trial.covarColl has no covariates with a populated time grid"
        )
    pos = _validate_position(position, expected_n_time=expected_n_time)
    trial_time = np.asarray(trial.covarColl.getCov(0).time, dtype=float).reshape(-1)
    x_bin = np.interp(bin_centres, trial_time, pos[:, 0])
    y_bin = np.interp(bin_centres, trial_time, pos[:, 1])
    x_lo, x_hi = float(np.min(x_bin)), float(np.max(x_bin))
    y_lo, y_hi = float(np.min(y_bin)), float(np.max(y_bin))
    # Pad slightly so endpoints land inside the spline support
    pad_x = 1e-3 * max(1.0, x_hi - x_lo)
    pad_y = 1e-3 * max(1.0, y_hi - y_lo)
    x_lo -= pad_x
    x_hi += pad_x
    y_lo -= pad_y
    y_hi += pad_y

    # ---- 3. B-spline basis on a coarse rate-map grid (for encoding) -----
    n_grid = max(int(cfg.n_basis_per_dim) * 4, 16)
    grid_x = np.linspace(x_lo, x_hi, n_grid)
    grid_y = np.linspace(y_lo, y_hi, n_grid)
    degree = max(int(cfg.spline_order) - 1, 1)
    B_grid = bspline_basis_2d(
        grid_x, grid_y,
        n_knots=(int(cfg.n_basis_per_dim), int(cfg.n_basis_per_dim)),
        degree=degree,
    )

    # Per-bin basis design (evaluated at the actual trajectory points).
    # ``bspline_basis_2d`` is grid-based, so for per-bin evaluation we
    # nearest-neighbour-snap the trajectory to (grid_x, grid_y) and pull
    # the corresponding row from ``B_grid``.  This preserves the
    # tensor-product basis without a separate per-point evaluator.
    ix = np.clip(
        np.searchsorted(grid_x, x_bin, side="left"), 0, grid_x.size - 1,
    )
    iy = np.clip(
        np.searchsorted(grid_y, y_bin, side="left"), 0, grid_y.size - 1,
    )
    # Pick the closer of (ix-1, ix) and (iy-1, iy).
    for arr, vals, grid in ((ix, x_bin, grid_x), (iy, y_bin, grid_y)):
        left = np.clip(arr - 1, 0, grid.size - 1)
        pick_left = (arr > 0) & (
            np.abs(grid[left] - vals) < np.abs(grid[arr] - vals)
        )
        arr[:] = np.where(pick_left, left, arr)
    flat_idx = ix * grid_y.size + iy
    B_bins = B_grid[flat_idx]  # (n_bins, n_basis_per_dim**2)

    # ---- 4. Per-cell pipeline -------------------------------------------
    spike_collection = trial.nspikeColl
    n_cells = int(spike_collection.numSpikeTrains)
    cifs: list[CIF] = []
    obs_rows: list[NDArray[np.float64]] = []
    spline_coefs: list[NDArray[np.float64]] = []
    quad_coefs: list[NDArray[np.float64]] = []
    kept: list[int] = []
    skipped: list[int] = []

    # Position design for the CIF refit (intercept added inside fit_*_glm).
    if cfg.cif_kind == "quadratic":
        X_pos = _quadratic_design(x_bin, y_bin)
        Xnames = ["1", "x", "y", "x^2", "y^2", "x*y"]
    else:
        X_pos = _linear_design(x_bin, y_bin)
        Xnames = ["1", "x", "y"]

    for c in range(n_cells):
        st = np.asarray(
            spike_collection.getNST(c).getSpikeTimes(min_time, max_time),
            dtype=float,
        ).reshape(-1)
        n_spikes = int(st.size)
        if n_spikes < cfg.min_n_spikes_per_cell:
            warnings.warn(
                f"place_field_decoder: cell {c} has {n_spikes} spike(s) "
                f"< min_n_spikes_per_cell={cfg.min_n_spikes_per_cell}; "
                f"skipping.",
                UserWarning,
                stacklevel=2,
            )
            skipped.append(c)
            continue

        counts = _bin_spikes_to_counts(st, bin_edges)

        # --- B-spline Poisson encoder ---
        try:
            glm = fit_poisson_glm(
                B_bins, counts,
                include_intercept=True,
                l2=1.0,
                max_iter=200,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"place_field_decoder: B-spline GLM failed for cell {c} "
                f"({type(exc).__name__}: {exc}); skipping.",
                UserWarning,
                stacklevel=2,
            )
            skipped.append(c)
            continue
        spline_coefs.append(
            np.concatenate([[glm.intercept], glm.coefficients])
        )

        # --- Quadratic / linear CIF refit (binomial / logistic) ---
        binary = _bin_spikes_to_bool(st, bin_edges)
        try:
            cif_glm = fit_binomial_glm(
                X_pos, binary,
                include_intercept=True,
                l2=1e-3,
                max_iter=120,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"place_field_decoder: quadratic CIF GLM failed for cell "
                f"{c} ({type(exc).__name__}: {exc}); skipping.",
                UserWarning,
                stacklevel=2,
            )
            # The B-spline coefs were already appended; pop them so the
            # ``spline_coefs`` and ``quadratic_coefs`` lists stay aligned
            # with ``cell_indices_kept``.
            spline_coefs.pop()
            skipped.append(c)
            continue
        b = np.concatenate([[cif_glm.intercept], cif_glm.coefficients])
        quad_coefs.append(b.copy())

        cif = CIF(b.tolist(), Xnames, ["x", "y"], fitType="binomial")
        cifs.append(cif)
        obs_rows.append(binary)
        kept.append(c)

    if not cifs:
        raise ValueError(
            "no cells kept after min_n_spikes filter "
            f"(min_n_spikes_per_cell={cfg.min_n_spikes_per_cell}); "
            f"cannot run decoder"
        )

    # ---- 5. Build observation matrix + decoder state-space --------------
    dN = np.vstack(obs_rows)  # (C_kept, n_bins)
    # 2-D random-walk state x_{k+1} = x_k + w_k, Q = empirical step covariance
    train_dx = np.diff(x_bin)
    train_dy = np.diff(y_bin)
    qxx = float(np.var(train_dx)) if train_dx.size else 1e-3
    qyy = float(np.var(train_dy)) if train_dy.size else 1e-3
    qxx = max(qxx, 1e-9)
    qyy = max(qyy, 1e-9)
    A = np.eye(2, dtype=float)
    Q = np.diag([qxx, qyy])
    x0 = np.array([x_bin[0], y_bin[0]], dtype=float)
    Pi0 = np.diag([qxx, qyy])

    # ---- 6. Dispatch to PPDecodeFilter{,Linear} -------------------------
    if cfg.decode_filter == "linear":
        # Extract (mu, beta) from each CIF — when cif_kind == "quadratic"
        # we Taylor-linearize the binomial log-odds at the trajectory mean
        # so that ``mu + beta @ x = f(m) + grad f(m) @ (x - m)`` matches
        # the second-order CIF at x = m.  Example08's _decode_position
        # uses ``mu = f(m)`` (not subtracting the linear correction); this
        # is fine when the trajectory mean is the origin (its [-1,1] box)
        # but biases away from origin, so we use the consistent expansion.
        state_mean = np.array(
            [float(x_bin.mean()), float(y_bin.mean())], dtype=float
        )
        mx, my = float(state_mean[0]), float(state_mean[1])
        n_cells_kept = len(cifs)
        mu = np.zeros(n_cells_kept, dtype=float)
        beta_mat = np.zeros((2, n_cells_kept), dtype=float)
        for c_idx, cif in enumerate(cifs):
            b = np.asarray(cif.b, dtype=float).reshape(-1)
            if cfg.cif_kind == "quadratic":
                # b = [b0, bx, by, bxx, byy, bxy]
                # f(x, y) = b0 + bx*x + by*y + bxx*x^2 + byy*y^2 + bxy*x*y
                f_m = (
                    b[0] + b[1] * mx + b[2] * my
                    + b[3] * mx * mx + b[4] * my * my + b[5] * mx * my
                )
                grad_x = b[1] + 2.0 * b[3] * mx + b[5] * my
                grad_y = b[2] + 2.0 * b[4] * my + b[5] * mx
                mu[c_idx] = float(f_m - grad_x * mx - grad_y * my)
                beta_mat[0, c_idx] = float(grad_x)
                beta_mat[1, c_idx] = float(grad_y)
            else:
                # b = [b0, bx, by]
                mu[c_idx] = float(b[0])
                beta_mat[0, c_idx] = float(b[1])
                beta_mat[1, c_idx] = float(b[2])

        result = DecodingAlgorithms.PPDecodeFilterLinear(
            A, Q, dN, mu, beta_mat, "binomial", float(cfg.bin_width_s),
            None, None, x0, Pi0,
        )
        x_u = result[2]
        W_u = result[3]
    else:
        # nonlinear (CIF object) branch
        result = DecodingAlgorithms.PPDecodeFilter(
            A, Q, Pi0, dN, cifs, float(cfg.bin_width_s),
            x0, Pi0,
        )
        x_u = result[2]
        W_u = result[3]

    # x_u is shape (state_dim, n_bins); transpose for the result container.
    decoded_position = np.ascontiguousarray(x_u.T)  # (n_bins, 2)
    # W_u is (state_dim, state_dim, n_bins) → transpose to (n_bins, 2, 2).
    decoded_covariance = np.ascontiguousarray(np.moveaxis(W_u, 2, 0))
    true_position = np.column_stack([x_bin, y_bin])
    err = np.sqrt(np.sum((decoded_position - true_position) ** 2, axis=1))
    mean_err = float(np.mean(err)) if err.size else float("nan")

    return PlaceFieldDecoderResult(
        decoded_position=decoded_position,
        decoded_covariance=decoded_covariance,
        decoding_error=err,
        mean_decoding_error=mean_err,
        cell_indices_kept=kept,
        cell_indices_skipped=skipped,
        spline_coefs=spline_coefs,
        quadratic_coefs=quad_coefs,
        n_basis_per_dim=int(cfg.n_basis_per_dim),
        bin_width_s=float(cfg.bin_width_s),
    )


__all__ = [
    "PlaceFieldDecoderConfig",
    "PlaceFieldDecoderResult",
    "fit_place_field_decoder",
]
