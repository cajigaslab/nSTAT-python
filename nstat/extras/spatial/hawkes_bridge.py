r"""Multivariate Hawkes process bridge via ``tick`` (optional).

Lazy bridge to the Hawkes estimators of `tick
<https://github.com/X-DataInitiative/tick>`_ (Bacry et al. 2018, JMLR 18;
BSD-3) for self / mutually-exciting temporal point processes —
functional-connectivity and triggering-kernel estimation for a
multi-electrode ensemble.

Install
-------

.. code-block:: bash

    pip install nstat-toolbox[hawkes]

The fitted :math:`C \times C` triggering matrix is the cell-averaged
continuous triggering kernel: a propagating kernel
:math:`\varphi(\tau, r) = \psi(\tau)\,\delta(r - v\tau)` gives
off-diagonal entries peaked at :math:`\tau \approx \|r_{cc'}\|/\|v\|`,
the signature of a traveling wave at speed :math:`v`.  This bridge is
intentionally thin: it wraps ``tick``'s exponential-kernel MLE and
returns plain NumPy.

.. note::

   Estimating inhibition (negative triggering) is hard and the
   exponential-kernel MLE here assumes excitation; see Bonnet (2023) for
   the inhibition-kernel difficulty.  *Confidence: high for excitatory
   exponential kernels; lower for inhibition.*
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstat.extras._lazy import require_optional


@dataclass(frozen=True)
class HawkesFit:
    """Fitted multivariate Hawkes parameters (plain NumPy).

    Attributes
    ----------
    baseline
        ``(C,)`` background rates :math:`\\mu_c`.
    adjacency
        ``(C, C)`` branching / triggering matrix (integrated kernel
        norms) — entry ``[c, c']`` is the expected number of events in
        channel ``c`` triggered by one event in channel ``c'``.
    decays
        The exponential decay(s) used in the parametric kernel.
    n_channels
        Number of channels ``C``.
    """

    baseline: np.ndarray
    adjacency: np.ndarray
    decays: np.ndarray
    n_channels: int


def fit_hawkes_exp(
    event_times,
    decay: float | np.ndarray = 1.0,
    *,
    penalty: str = "l2",
    C: float = 1e3,
    max_iter: int = 200,
) -> HawkesFit:
    r"""Fit a multivariate Hawkes process with exponential kernels via ``tick``.

    Parameters
    ----------
    event_times
        Sequence of length ``C`` — one increasing 1-D array of event
        times (seconds) per channel.
    decay
        Exponential decay :math:`\beta` (scalar, shared across channels).
    penalty
        Regularization passed to ``tick`` (``"l2"`` default; ``"l1"`` for
        sparse connectivity à la ADM4).
    C
        Inverse regularization strength (``tick`` convention).
    max_iter
        Solver iterations.

    Returns
    -------
    HawkesFit

    Raises
    ------
    ImportError
        If ``tick`` is not installed (with the ``[hawkes]`` install hint).
    """
    tick_hawkes = require_optional("tick.hawkes", install_key="hawkes")
    HawkesExpKern = tick_hawkes.HawkesExpKern

    events = [np.asarray(e, dtype=float) for e in event_times]
    n_channels = len(events)
    learner = HawkesExpKern(
        decays=float(np.asarray(decay).ravel()[0]),
        penalty=penalty,
        C=C,
        max_iter=max_iter,
    )
    learner.fit(events)
    return HawkesFit(
        baseline=np.asarray(learner.baseline, dtype=float),
        adjacency=np.asarray(learner.adjacency, dtype=float),
        decays=np.atleast_1d(np.asarray(decay, dtype=float)),
        n_channels=n_channels,
    )


__all__ = ["HawkesFit", "fit_hawkes_exp"]
