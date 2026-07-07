r"""Continuous-time conditional-intensity point-process simulator.

Native-Python port of the MATLAB Simulink model
``PointProcessSimulationCont.slx``: the stimulus and ensemble drives are
realised as *continuous* LTI filters (``S``, ``E``) integrated with
:func:`scipy.signal.lsim`, while the self-history feedback (``H``) is
discretized to the spike-generation sample time via
:func:`scipy.signal.cont2discrete` (zero-order hold) and stepped once per
bin.  This is a Python-only ``nstat.extras`` feature — the discrete
``CIF.simulateCIF`` in core ``nstat`` has no continuous-transfer-function
path, so there is no MATLAB-parity obligation to keep this in ``nstat.cif``.

Model
-----
At each sample time :math:`t_k = k \, T_s`::

    eta(t)         = mu + S(s)*stim(t) + E(s)*ens(t) + H(s)*pp_delayed(t)
    lambdaDelta(t) = exp(eta)               (simType='poisson')
                   = exp(eta) / (1+exp(eta))  (simType='binomial')
    spike(t)       = 1  iff  U(0,1) < lambdaDelta(t)
    rate_hz(t)     = lambdaDelta(t) / Ts

``pp_delayed`` is the *previous* bin's realised spike indicator (a
one-step delay, ``z^-1``), fed through the discretized ``H`` filter —
mirroring the injected-uniform, step-loop pattern of
:func:`nstat.simulators.simulate_two_neuron_network`.

Validation
----------
With ``H = 0`` (no self-history feedback), ``eta`` — and therefore
``lambda_delta`` — does not depend on the realised spikes, so it is
bit-close deterministic and is validated against the MATLAB gold fixture
``tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat``.  With ``H !=
0``, ``lambda_delta`` depends on the stochastic spike sequence and is
instead validated Python-side using injected ``uniform_values`` (MATLAB's
DSP Random Source RNG is not reproducible in NumPy).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete, lsim, tf2ss

from ..cif import _sigmoid
from ..simulators import PointProcessSimulation
from ..spikes import SpikeTrain

_ETA_CLIP = 20.0


def _as_num_den(system_like) -> tuple[np.ndarray, np.ndarray]:
    """Coerce an LTI-like spec into ``(num, den)`` 1-D float arrays.

    Accepts ``(num, den)`` tuples, objects exposing ``.num``/``.den``
    (e.g. :class:`scipy.signal.lti` / ``TransferFunction`` instances), or
    a plain array-like / scalar, which is treated as an FIR numerator
    over ``den = [1]`` (a bare ``0.0`` therefore yields the degenerate
    "zero system"; a bare nonzero scalar yields a static gain).
    """
    if isinstance(system_like, tuple) and len(system_like) == 2:
        num, den = system_like
        return (
            np.atleast_1d(np.asarray(num, dtype=float)),
            np.atleast_1d(np.asarray(den, dtype=float)),
        )
    if hasattr(system_like, "num") and hasattr(system_like, "den"):
        return (
            np.atleast_1d(np.asarray(system_like.num, dtype=float)),
            np.atleast_1d(np.asarray(system_like.den, dtype=float)),
        )
    num = np.atleast_1d(np.asarray(system_like, dtype=float))
    return num, np.array([1.0])


def _extract_time_series(source, argname: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(time, values)`` 1-D float arrays out of a Covariate-like or tuple input."""
    if hasattr(source, "time") and hasattr(source, "values"):
        t = np.asarray(source.time, dtype=float).reshape(-1)
        v = np.asarray(source.values, dtype=float).reshape(-1)
        return t, v
    if isinstance(source, tuple) and len(source) == 2:
        t = np.asarray(source[0], dtype=float).reshape(-1)
        v = np.asarray(source[1], dtype=float).reshape(-1)
        if t.shape[0] != v.shape[0]:
            raise ValueError(f"{argname}: time and value arrays must have matching length")
        return t, v
    raise TypeError(
        f"{argname} must be a Covariate-like object (with .time/.values attributes) "
        "or a (time, values) tuple"
    )


def _lsim_drive(num: np.ndarray, den: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Continuous LTI drive ``lsim((num, den), u, t)``; degenerate systems fall out for free."""
    _, y, _ = lsim((num, den), u, t)
    return np.asarray(y, dtype=float).reshape(-1)


def _discretize_history(num: np.ndarray, den: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-order-hold discretization of the continuous history filter H to sample time Ts."""
    A, B, C, D = tf2ss(num, den)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), Ts, method="zoh")
    return Ad, Bd, Cd, Dd


def simulate_cif_continuous(
    mu: float,
    stim,
    ens,
    hist,
    input_stim,
    input_ens,
    *,
    Ts: float,
    simType: str = "binomial",
    seed: int | None = None,
    uniform_values: np.ndarray | None = None,
    return_details: bool = False,
) -> PointProcessSimulation:
    """Simulate a continuous-time CIF point process (Matlab ``PointProcessSimulationCont.slx``).

    Parameters
    ----------
    mu : float
        Baseline log-odds / log-rate offset (constant term of ``eta``).
    stim, ens, hist : (num, den) tuple | scipy.signal.lti | array-like | float
        Continuous LTI specs for the stimulus filter ``S``, ensemble
        filter ``E``, and self-history feedback filter ``H``.  A bare
        array/scalar is treated as an FIR numerator over ``den = [1]``
        (so ``0.0`` gives the degenerate zero system and a nonzero
        scalar gives a static gain).  ``hist`` has no external input —
        it filters the one-step-delayed realised spike indicator.
    input_stim, input_ens : Covariate | (time, values) tuple
        Stimulus / ensemble drive time series.  Must share the same time
        grid.  The simulation runs on this grid.
    Ts : float
        Spike-generation bin / random-source sample time (seconds); also
        the sample time used to discretize ``hist``.
    simType : {'binomial', 'poisson'}, default 'binomial'
        Link function: logistic (``sigmoid(eta)``) for ``'binomial'``,
        exponential (``exp(eta)``, clipped to ``|eta| <= 20`` for
        numerical safety) for ``'poisson'``.  The binomial link uses the
        numerically-stable :func:`nstat.cif._sigmoid` (branches on the
        sign of ``eta`` internally, no ``eta`` clipping needed); the
        poisson link clips ``|eta| <= 20`` before exponentiating instead.

        .. note::
           **Poisson-link footgun.** For ``simType='poisson'``,
           ``lambdaDelta = exp(eta)`` is unbounded above and can exceed 1
           whenever ``eta > 0``.  Since the per-bin spike test is ``U(0,1)
           < lambdaDelta``, a ``lambdaDelta >= 1`` fires on *every* draw,
           producing deterministic (non-random) spiking every bin. This is
           faithful to the MATLAB Simulink model (no clamping to ``[0,
           1]`` is applied), but it is easy to trip over. Users who want a
           genuinely rate-limited process should keep ``eta`` negative
           (small ``mu`` / gains so ``exp(eta) < 1``) or use
           ``simType='binomial'``, whose logistic link is bounded in
           ``(0, 1)`` for any ``eta``.
    seed : int or None, optional
        Seed for :func:`numpy.random.default_rng` when ``uniform_values``
        is not supplied.
    uniform_values : ndarray, optional
        Pre-drawn ``U(0,1)`` variates, one per sample, for exact spike
        reproducibility (bypasses the RNG entirely).
    return_details : bool, default False
        When ``True``, attach diagnostic arrays (``eta``, ``stim_drive``,
        ``ens_drive``, ``history_effect``) onto the returned object as
        extra attributes (not part of the :class:`PointProcessSimulation`
        dataclass fields).

    Returns
    -------
    PointProcessSimulation
        ``time`` is the shared input grid; ``rate_hz = lambda_delta /
        Ts``; ``spikes`` holds the realised :class:`~nstat.spikes.SpikeTrain`;
        ``lambda_delta``, ``spike_indicator``, and ``uniform_values`` are
        the per-bin traces.

    Raises
    ------
    ValueError
        If ``Ts <= 0``, ``simType`` is not ``'poisson'``/``'binomial'``,
        the stimulus/ensemble grids disagree, or ``uniform_values`` does
        not match the time grid length.
    TypeError
        If ``input_stim`` or ``input_ens`` is neither a Covariate-like
        object (exposing ``.time``/``.values``) nor a ``(time, values)``
        tuple (see :func:`_extract_time_series`).
    """
    if Ts <= 0:
        raise ValueError("Ts must be > 0")
    if simType not in ("poisson", "binomial"):
        raise ValueError("simType must be 'poisson' or 'binomial'")

    t_stim, u_stim = _extract_time_series(input_stim, "input_stim")
    t_ens, u_ens = _extract_time_series(input_ens, "input_ens")
    if t_stim.shape != t_ens.shape or not np.allclose(t_stim, t_ens):
        raise ValueError("input_stim and input_ens must share the same time grid")

    t = t_stim
    n = t.shape[0]
    if n < 1:
        raise ValueError("input_stim must have at least one sample")

    S_num, S_den = _as_num_den(stim)
    E_num, E_den = _as_num_den(ens)
    H_num, H_den = _as_num_den(hist)

    stim_drive = _lsim_drive(S_num, S_den, u_stim, t)
    ens_drive = _lsim_drive(E_num, E_den, u_ens, t)
    Ad, Bd, Cd, Dd = _discretize_history(H_num, H_den, Ts)
    n_states = Ad.shape[0]
    x_h = np.zeros((n_states, 1), dtype=float)
    d_scalar = float(Dd[0, 0])

    if uniform_values is None:
        rng = np.random.default_rng(seed)
        draws = rng.random(n)
    else:
        draws = np.asarray(uniform_values, dtype=float).reshape(-1)
        if draws.shape[0] != n:
            raise ValueError("uniform_values must match the length of time")

    eta = np.zeros(n, dtype=float)
    lambda_delta = np.zeros(n, dtype=float)
    spike = np.zeros(n, dtype=float)
    history_effect = np.zeros(n, dtype=float)

    pp_delayed = 0.0
    for k in range(n):
        h_effect = float((Cd @ x_h)[0, 0]) + d_scalar * pp_delayed
        x_h = Ad @ x_h + Bd * pp_delayed

        eta_k = mu + stim_drive[k] + ens_drive[k] + h_effect
        eta[k] = eta_k
        history_effect[k] = h_effect

        if simType == "poisson":
            lam = float(np.exp(np.clip(eta_k, -_ETA_CLIP, _ETA_CLIP)))
        else:
            lam = float(_sigmoid(np.asarray([eta_k], dtype=float))[0])
        lambda_delta[k] = lam

        spike[k] = 1.0 if draws[k] < lam else 0.0
        pp_delayed = spike[k]

    rate_hz = lambda_delta / Ts
    result = PointProcessSimulation(
        time=t,
        rate_hz=rate_hz,
        spikes=SpikeTrain(t[spike > 0.5]),
        lambda_delta=lambda_delta,
        spike_indicator=spike,
        uniform_values=draws,
    )
    if return_details:
        result.eta = eta  # type: ignore[attr-defined]
        result.stim_drive = stim_drive  # type: ignore[attr-defined]
        result.ens_drive = ens_drive  # type: ignore[attr-defined]
        result.history_effect = history_effect  # type: ignore[attr-defined]
    return result


__all__ = ["simulate_cif_continuous"]
