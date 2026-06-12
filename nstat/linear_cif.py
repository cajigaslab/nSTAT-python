"""Closed-form canonical-link conditional intensity function.

``LinearCIF`` is a Python port of MATLAB ``LinearCIF.m`` (added in
upstream v1.4.0).  It implements the same 5-method interface as
:class:`nstat.cif.CIF` (``evalLambdaDelta``, ``evalGradient``,
``evalGradientLog``, ``evalJacobian``, ``evalJacobianLog``) but only
supports the two canonical-link cases — Poisson (log link) and
binomial (logit link).  For these, derivatives of ``lambda*delta`` and
``log(lambda*delta)`` w.r.t. the stimulus variables have a closed form
and **require no symbolic-math toolkit** (no SymPy dependency).

Closed-form derivatives:

  Poisson:  ld       = exp(X·beta + H·gamma)
            grad     = ld · beta_stim
            gradlog  = beta_stim
            jacobian = ld · (beta_stim^T · beta_stim)
            jaclog   = 0_(nStim × nStim)

  Binomial: ld       = sigma(X·beta + H·gamma)
            grad     = ld·(1−ld) · beta_stim
            gradlog  = (1−ld) · beta_stim
            jacobian = ld·(1−ld)·(1−2·ld) · (beta_stim^T · beta_stim)
            jaclog   = −ld·(1−ld) · (beta_stim^T · beta_stim)

See Also
--------
nstat.cif.CIF : Full symbolic CIF, used when the link is non-canonical
    or when symbolic differentiation is required for fitting.

Matlab cross-reference
----------------------
This class mirrors MATLAB ``LinearCIF.m`` 1:1 for the public API.  The
storage layout (``b``, ``stimIdx``, ``bStim``, ``historyMat``) and the
private helpers (``_compute_eta``, ``_resolve_hist_val``,
``_expand_stim_to_var_in``, ``_link_inverse``) match the MATLAB
counterparts; only Python-idiomatic ``snake_case`` is used for private
helpers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .history import History
from ._spike_train_impl import nspikeTrain


_VALID_FIT_TYPES = frozenset({"poisson", "binomial"})


class LinearCIF:
    """Conditional intensity function with closed-form derivatives.

    Drop-in compatible with :class:`nstat.cif.CIF` for the 5 eval methods
    used by :class:`nstat.decoding_algorithms.DecodingAlgorithms.PPDecode_update`:
    ``evalLambdaDelta``, ``evalGradient``, ``evalGradientLog``,
    ``evalJacobian``, ``evalJacobianLog``.

    Parameters
    ----------
    beta : array_like
        Numeric regression coefficients (length ``len(Xnames)``).  Must be
        numeric — symbolic input is unsupported here because the entire
        point of ``LinearCIF`` is closed-form numeric differentiation.
    Xnames : sequence of str
        Names of every design-matrix column (length equal to ``beta``).
    stimNames : sequence of str
        Subset of ``Xnames`` that are stimulus/state variables.  Every
        element must appear in ``Xnames``.
    fitType : {"poisson", "binomial"}, optional
        Canonical link family.  Default ``"poisson"``.
    histCoeffs : array_like or None
        History coefficients (1-D, length ``history.numWindows``).  When
        ``None`` or empty, the linear predictor has no history term.
    historyObj : History or array_like, optional
        History object or a window-times vector (which is converted into
        a :class:`History` internally).
    nst : nspikeTrain, optional
        Spike train used to pre-compute the history matrix when both
        ``historyObj`` and ``nst`` are provided.

    Attributes
    ----------
    b : ndarray, shape (n_var,)
        Regression coefficients (row-vector semantics for left-multiplies).
    varIn : tuple of str
        All variable names (preserves order).
    stimVars : tuple of str
        Stimulus-variable subset of ``varIn``.
    stimIdx : ndarray of int, shape (n_stim,)
        0-based indices of ``stimVars`` inside ``varIn``.
    bStim : ndarray, shape (n_stim,)
        Stimulus subset of ``b`` (row-vector semantics).
    fitType : str
        ``"poisson"`` or ``"binomial"``.
    histCoeffs : ndarray or None
        History coefficient row-vector.
    history : History or None
        History object (or ``None`` if not supplied).
    spikeTrain : nspikeTrain or None
        Spike train used to pre-compute ``historyMat`` (or ``None``).
    historyMat : ndarray or None
        Pre-computed history matrix
        (shape ``(n_time, n_hist)``) when both ``history`` and
        ``spikeTrain`` were supplied at construction time.
    """

    def __init__(
        self,
        beta,
        Xnames,
        stimNames,
        fitType: str = "poisson",
        histCoeffs=None,
        historyObj=None,
        nst: nspikeTrain | None = None,
    ) -> None:
        # --- spike train (optional) ------------------------------------
        if nst is None:
            self.spikeTrain: nspikeTrain | None = None
        else:
            self.spikeTrain = nst.nstCopy()

        # --- history object (optional) ---------------------------------
        if historyObj is None:
            self.history: History | None = None
        else:
            self.setHistory(historyObj)

        # --- history coefficients (optional) ---------------------------
        if histCoeffs is None or (
            hasattr(histCoeffs, "__len__") and len(histCoeffs) == 0
        ):
            self.histCoeffs: np.ndarray | None = None
        else:
            arr = np.asarray(histCoeffs, dtype=float)
            if arr.ndim > 2 or (arr.ndim == 2 and 1 not in arr.shape):
                raise ValueError(
                    "LinearCIF: histCoeffs must have one dimension equal to 1; "
                    f"got shape {arr.shape}"
                )
            self.histCoeffs = arr.reshape(-1)

        # --- fitType ---------------------------------------------------
        if fitType is None or fitType == "":
            fitType = "poisson"
        if fitType not in _VALID_FIT_TYPES:
            raise ValueError(
                f"LinearCIF: fitType must be 'poisson' or 'binomial', got {fitType!r}"
            )
        self.fitType: str = fitType

        # --- Xnames -> varIn -------------------------------------------
        names = list(_normalize_string_sequence(Xnames, "Xnames"))
        self.varIn: tuple[str, ...] = tuple(names)

        # --- stimNames -> stimVars + stimIdx ---------------------------
        stim_names_list = list(_normalize_string_sequence(stimNames, "stimNames"))
        self.stimVars: tuple[str, ...] = tuple(stim_names_list)

        # --- beta normalization ---------------------------------------
        beta_arr = np.asarray(beta, dtype=float)
        if beta_arr.ndim > 2 or (beta_arr.ndim == 2 and 1 not in beta_arr.shape):
            raise ValueError(
                "LinearCIF: beta must have one dimension equal to 1; "
                f"got shape {beta_arr.shape}"
            )
        self.b: np.ndarray = beta_arr.reshape(-1)
        if self.b.size != len(self.varIn):
            raise ValueError(
                f"LinearCIF: len(beta)={self.b.size} does not match "
                f"len(Xnames)={len(self.varIn)}"
            )

        # --- stim indices into varIn -----------------------------------
        stim_idx = np.empty(len(self.stimVars), dtype=int)
        for k, sv in enumerate(self.stimVars):
            try:
                stim_idx[k] = self.varIn.index(sv)
            except ValueError:
                raise ValueError(
                    f"LinearCIF: stimulus variable {sv!r} not found in Xnames"
                ) from None
        self.stimIdx: np.ndarray = stim_idx
        self.bStim: np.ndarray = self.b[self.stimIdx]

        # --- pre-compute history matrix -------------------------------
        if self.spikeTrain is not None and self.history is not None:
            self.historyMat = np.asarray(
                self.history.computeHistory(self.spikeTrain).dataToMatrix(),
                dtype=float,
            )
        else:
            self.historyMat: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def setSpikeTrain(self, spikeTrain: nspikeTrain) -> None:
        """Replace the stored spike train and refresh ``historyMat``.

        Parameters
        ----------
        spikeTrain : nspikeTrain
            Spike train to use for history evaluation.  A deep copy is
            taken via ``spikeTrain.nstCopy()`` so subsequent mutations
            of the source do not affect this CIF.
        """
        self.spikeTrain = spikeTrain.nstCopy()
        if self.history is not None:
            self.historyMat = np.asarray(
                self.history.computeHistory(self.spikeTrain).dataToMatrix(),
                dtype=float,
            )
        else:
            self.historyMat = None

    def setHistory(self, histObj) -> None:
        """Replace the stored history object.

        Parameters
        ----------
        histObj : History or array_like
            A :class:`History` instance (its ``windowTimes`` are copied
            into a fresh ``History``) **or** a 1-D vector of window
            times.

        Notes
        -----
        Mirrors MATLAB ``LinearCIF.setHistory`` exactly — accepts either
        a ``History`` object or a numeric ``windowTimes`` array.
        """
        if isinstance(histObj, History):
            self.history = History(histObj.windowTimes)
            return
        try:
            arr = np.asarray(histObj, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "LinearCIF: history can only be set by passing in a "
                "History object or a vector of windowTimes"
            ) from exc
        if arr.ndim != 1:
            raise ValueError(
                "LinearCIF: history can only be set by passing in a "
                "History object or a vector of windowTimes"
            )
        self.history = History(arr)

    # ------------------------------------------------------------------
    # Public eval API (matches CIF.eval* signatures exactly)
    # ------------------------------------------------------------------

    def evalLambdaDelta(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
    ) -> float:
        """Evaluate scalar ``lambda*delta`` at the given stimulus values.

        Parameters
        ----------
        stimVal : array_like
            Either the full design vector (length ``len(varIn)``) or the
            stimulus-only subset (length ``len(stimVars)``).
        time_index : int or None
            0-based time index into ``historyMat``.  Mirrors MATLAB's
            0-based indexing (``nst.dataToMatrix(time_index)``).
        nst : nspikeTrain or None
            On-the-fly history computation for the supplied train.

        Returns
        -------
        float
            Scalar value of ``lambda*delta``.
        """
        eta = self._compute_eta(stimVal, time_index, nst)
        return self._link_inverse(eta)

    def evalGradient(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
    ) -> np.ndarray:
        """Gradient of ``lambda*delta`` w.r.t. the stimulus variables.

        Returns
        -------
        ndarray, shape (1, n_stim)
            Row vector of partial derivatives.
        """
        ld = self.evalLambdaDelta(stimVal, time_index, nst)
        if self.fitType == "poisson":
            scale = ld
        else:  # binomial
            scale = ld * (1.0 - ld)
        return (scale * self.bStim).reshape(1, -1)

    def evalGradientLog(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
    ) -> np.ndarray:
        """Gradient of ``log(lambda*delta)`` w.r.t. the stimulus variables.

        Returns
        -------
        ndarray, shape (1, n_stim)
            Row vector of partial derivatives.
        """
        if self.fitType == "poisson":
            return self.bStim.reshape(1, -1).copy()
        # binomial
        ld = self.evalLambdaDelta(stimVal, time_index, nst)
        return ((1.0 - ld) * self.bStim).reshape(1, -1)

    def evalJacobian(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
    ) -> np.ndarray:
        """Hessian of ``lambda*delta`` w.r.t. the stimulus variables.

        Returns
        -------
        ndarray, shape (n_stim, n_stim)
            Second-derivative matrix.
        """
        ld = self.evalLambdaDelta(stimVal, time_index, nst)
        outer = np.outer(self.bStim, self.bStim)
        if self.fitType == "poisson":
            return ld * outer
        # binomial — the canonical sigmoid third-derivative factor.
        return ld * (1.0 - ld) * (1.0 - 2.0 * ld) * outer

    def evalJacobianLog(
        self,
        stimVal,
        time_index: int | None = None,
        nst: nspikeTrain | None = None,
    ) -> np.ndarray:
        """Hessian of ``log(lambda*delta)`` w.r.t. the stimulus variables.

        Returns
        -------
        ndarray, shape (n_stim, n_stim)
            For Poisson: zero matrix (canonical-link log-likelihood is
            linear in the stimulus, so its Hessian vanishes).
            For binomial: ``-ld·(1-ld)·outer(bStim, bStim)``.
        """
        n_stim = len(self.stimVars)
        if self.fitType == "poisson":
            return np.zeros((n_stim, n_stim), dtype=float)
        # binomial
        ld = self.evalLambdaDelta(stimVal, time_index, nst)
        return -ld * (1.0 - ld) * np.outer(self.bStim, self.bStim)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_eta(
        self,
        stimVal,
        time_index: int | None,
        nst: nspikeTrain | None,
    ) -> float:
        """Linear predictor ``eta = X·beta + H·gamma`` at the given inputs."""
        hist_val = self._resolve_hist_val(time_index, nst)
        x_full = self._expand_stim_to_var_in(stimVal)
        eta = float(self.b @ x_full)
        if hist_val is not None and self.histCoeffs is not None:
            eta += float(self.histCoeffs @ hist_val)
        return eta

    def _resolve_hist_val(
        self,
        time_index: int | None,
        nst: nspikeTrain | None,
    ) -> np.ndarray | None:
        """Pick the right history column for the current eval call.

        Mirrors MATLAB ``LinearCIF.resolveHistVal``: when ``nst`` is
        absent and a pre-computed ``historyMat`` exists, slice at
        ``time_index`` (0-based; the Python port flipped the indexing
        convention in v0.5.0).  When ``nst`` is provided, compute fresh
        history and use the last row.
        """
        if nst is None:
            if time_index is not None and self.historyMat is not None:
                idx = int(time_index)
                return np.asarray(self.historyMat[idx, :], dtype=float).reshape(-1)
            return None
        if not isinstance(nst, nspikeTrain):
            raise TypeError(
                "LinearCIF._resolve_hist_val: nst must be an nspikeTrain "
                f"instance, got {type(nst).__name__}"
            )
        if self.history is None:
            return None
        hist_data = np.asarray(
            self.history.computeHistory(nst).dataToMatrix(),
            dtype=float,
        )
        return hist_data[-1, :].reshape(-1)

    def _expand_stim_to_var_in(self, stimVal) -> np.ndarray:
        """Fill the full design vector from the stimulus subset.

        If ``stimVal`` already has length ``len(varIn)``, it is passed
        through unchanged.  If it has length ``len(stimVars)``, the
        non-stimulus positions (typically intercept / constant columns)
        are filled with 1.0 — matching MATLAB ``CIF.expandStimToVarIn``.
        """
        arr = np.asarray(stimVal, dtype=float).reshape(-1)
        n_var = len(self.varIn)
        n_stim = len(self.stimVars)
        if arr.size == n_var:
            return arr
        if arr.size == n_stim:
            full = np.ones(n_var, dtype=float)
            full[self.stimIdx] = arr
            return full
        raise ValueError(
            f"LinearCIF: stimVal must have length {n_var} (all vars) or "
            f"{n_stim} (stim vars only), got {arr.size}."
        )

    def _link_inverse(self, eta: float) -> float:
        """Apply the canonical inverse link to the linear predictor.

        For ``fitType='binomial'`` this is a logistic sigmoid σ(η);
        for ``'poisson'`` it is ``exp(η)``.  The sigmoid is the shared
        :func:`nstat.cif._sigmoid` helper (audit finding H1 — unified
        with the cif.py path so both CIF classes produce numerically
        identical λ values).
        """
        if self.fitType == "poisson":
            return float(np.exp(eta))
        # binomial — delegate to the shared two-branch stable sigmoid.
        from .cif import _sigmoid as _shared_sigmoid
        return float(_shared_sigmoid(np.asarray(eta, dtype=float)))


# ----------------------------------------------------------------------
# Module-private helpers
# ----------------------------------------------------------------------

def _normalize_string_sequence(seq: Any, label: str) -> list[str]:
    """Coerce a name vector into a Python ``list[str]``.

    Mirrors MATLAB ``LinearCIF.__init__``'s normalization step that
    accepted either a sym column vector, a cell array of chars, or a
    row vector and produced a column.  In Python we just return the
    canonical list-of-str representation.
    """
    if isinstance(seq, str):
        return [seq]
    if seq is None:
        raise ValueError(f"LinearCIF: {label} must be a sequence of strings")
    if isinstance(seq, np.ndarray):
        if seq.dtype.kind not in ("U", "S", "O"):
            raise ValueError(
                f"LinearCIF: {label} must be a sequence of strings; "
                f"got numeric ndarray with dtype {seq.dtype}"
            )
        seq = seq.tolist()
    try:
        return [str(item) for item in seq]
    except TypeError as exc:
        raise ValueError(
            f"LinearCIF: {label} must be a sequence of strings"
        ) from exc


__all__ = ["LinearCIF"]
