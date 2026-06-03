"""Clusterless / trajectory-classification decoding via ``replay_trajectory_classification``.

This bridge connects nSTAT's point-process decoding lineage (PPAF / PPHF
in :class:`nstat.DecodingAlgorithms`) with the modern marked
point-process state-space decoders of Denovellis et al. 2021, eLife —
implemented in `replay_trajectory_classification
<https://github.com/Eden-Kramer-Lab/replay_trajectory_classification>`_
(MIT).  Two capabilities that the MATLAB nSTAT toolbox does **not**
provide:

- **Clusterless decoding** — the observation model is a marked point
  process where each spike carries the multi-dim waveform features that
  would have been used for spike sorting; this avoids the
  cluster-quality dependence of sorted decoders.  See the seminal Kloosterman
  et al. (2014) / Deng et al. (2015) clusterless framework and Denovellis
  2021 for the state-space integration.
- **Trajectory classification** — a discrete latent state (e.g.
  *local*, *forward replay*, *reverse replay*, *fragmented*) sits on
  top of the continuous position decode, letting the model identify
  what kind of trajectory is being represented at each moment.

Scope of this bridge
--------------------

- :func:`fit_clusterless_decoder` — wraps :class:`~replay_trajectory_classification.ClusterlessDecoder`
  (single continuous state, one movement model).
- :func:`fit_clusterless_classifier` — wraps :class:`~replay_trajectory_classification.ClusterlessClassifier`
  (multiple discrete states / movement-model mixtures).

Both accept plain NumPy inputs (the position trajectory and the
multiunit-mark cube) and return :class:`ClusterlessDecoderResult` /
:class:`ClusterlessClassifierResult` — frozen dataclasses with plain
NumPy fields, so downstream code never sees xarray.  The actual
state-space inference, parameter fitting, and likelihood machinery is
all delegated to the upstream library; this bridge is intentionally
thin (data conversion, sensible defaults, plain-NumPy outputs).

Install
-------

.. code-block:: bash

    pip install nstat-toolbox[clusterless]

Pulls ``replay_trajectory_classification`` and its JAX-based numerical
stack (~200 MB).  Like ``[dynamax]``, this group is **not** rolled into
``[all-extras]``; install it explicitly when you need clusterless
decoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstat.extras._lazy import require_optional


def _require_clusterless():
    """Lazy-import the upstream library; raise the canonical install hint."""
    return require_optional(
        "replay_trajectory_classification", install_key="clusterless"
    )


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterlessDecoderResult:
    """Output of :func:`fit_clusterless_decoder`.

    Plain-NumPy view of the upstream :class:`xarray.Dataset` returned by
    :meth:`~replay_trajectory_classification.ClusterlessDecoder.predict`.

    Attributes
    ----------
    posterior
        Smoothed (acausal) posterior over the decoded variable, shape
        ``(n_time, n_position_bins[0], ...)`` — the leading dim is time,
        the trailing dims are the position-bin grid for each spatial
        dimension.  Sums to 1 along the position axes at each time.
    map_position
        Posterior-mode (MAP) position bin index at each time, shape
        ``(n_time, n_position_dims)``.  Use ``position_bin_centers`` to
        convert to physical units.
    position_bin_centers
        Bin centres along each position dimension; tuple of 1-D arrays.
    causal_posterior
        Filter (causal) posterior — same shape as ``posterior`` but
        conditioning only on the past at each time.
    """

    posterior: np.ndarray
    map_position: np.ndarray
    position_bin_centers: tuple[np.ndarray, ...]
    causal_posterior: np.ndarray


@dataclass(frozen=True)
class ClusterlessClassifierResult:
    """Output of :func:`fit_clusterless_classifier`.

    Plain-NumPy view of the upstream :class:`xarray.Dataset` returned by
    :meth:`~replay_trajectory_classification.ClusterlessClassifier.predict`.

    Attributes
    ----------
    posterior
        Smoothed joint posterior over ``(state, position)``, shape
        ``(n_time, n_states, n_position_bins[0], ...)``.
    state_probabilities
        Marginal smoothed posterior over the discrete state, shape
        ``(n_time, n_states)``.  Sums to 1 along the state axis.
    state_names
        Human-readable name for each discrete state, length ``n_states``.
    position_bin_centers
        Bin centres along each position dimension; tuple of 1-D arrays.
    """

    posterior: np.ndarray
    state_probabilities: np.ndarray
    state_names: list[str]
    position_bin_centers: tuple[np.ndarray, ...]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _validate_inputs(position: np.ndarray, multiunits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce + shape-check the two principal inputs."""
    position = np.asarray(position, dtype=float)
    multiunits = np.asarray(multiunits, dtype=float)
    if position.ndim == 1:
        position = position[:, None]
    if position.ndim != 2:
        raise ValueError(
            f"position must be shape (n_time, n_position_dims); got {position.shape}"
        )
    if multiunits.ndim != 3:
        raise ValueError(
            f"multiunits must be shape (n_time, n_marks, n_electrodes); "
            f"got {multiunits.shape}"
        )
    if position.shape[0] != multiunits.shape[0]:
        raise ValueError(
            f"position and multiunits must share the same n_time; got "
            f"{position.shape[0]} vs {multiunits.shape[0]}"
        )
    return position, multiunits


def _extract_position_bin_centers(dataset) -> tuple[np.ndarray, ...]:
    """Pull the position-bin centre arrays out of an upstream xarray Dataset.

    The library names the spatial coordinates ``position`` (1-D track) or
    ``x_position`` / ``y_position`` (2-D).  Be defensive about which.
    """
    centres = []
    for name in ("position", "x_position", "y_position"):
        if name in getattr(dataset, "coords", {}):
            centres.append(np.asarray(dataset.coords[name].values, dtype=float))
    return tuple(centres)


def _extract_posterior(dataset, *, prefer_acausal: bool = True):
    """Return ``(posterior, causal_posterior)`` plain-NumPy arrays.

    Upstream emits ``acausal_posterior`` when ``is_compute_acausal=True``
    (default) and always ``causal_posterior``; either may be absent in
    edge configurations.  We prefer the smoothed one as the primary
    ``posterior`` field.
    """
    has_acausal = "acausal_posterior" in dataset
    has_causal = "causal_posterior" in dataset
    if not (has_acausal or has_causal):
        raise RuntimeError(
            "upstream predict() Dataset contains neither acausal_posterior "
            "nor causal_posterior — cannot extract the decode posterior."
        )
    causal = np.asarray(dataset["causal_posterior"].values) if has_causal else None
    if prefer_acausal and has_acausal:
        primary = np.asarray(dataset["acausal_posterior"].values)
    else:
        primary = causal if causal is not None else np.asarray(dataset["acausal_posterior"].values)
    if causal is None:
        causal = primary.copy()
    return primary, causal


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def fit_clusterless_decoder(
    position: np.ndarray,
    multiunits: np.ndarray,
    *,
    place_bin_size: float = 2.0,
    movement_var: float | None = None,
    is_training: np.ndarray | None = None,
    is_compute_acausal: bool = True,
) -> ClusterlessDecoderResult:
    """Fit + decode a single-state clusterless point-process state-space decoder.

    Wraps :class:`replay_trajectory_classification.ClusterlessDecoder`
    with sensible defaults: a single :class:`~replay_trajectory_classification.Environment`
    with the given ``place_bin_size`` and a single
    :class:`~replay_trajectory_classification.RandomWalk` continuous-state
    transition.  For multi-state (e.g. replay) classification, use
    :func:`fit_clusterless_classifier` instead.

    Parameters
    ----------
    position
        Animal position on the training trajectory, shape
        ``(n_time, n_position_dims)`` (1-D arrays are reshaped).
    multiunits
        Multiunit mark cube, shape ``(n_time, n_marks, n_electrodes)``.
        ``NaN`` entries denote "no spike on this electrode at this
        time" — the upstream convention.
    place_bin_size
        Spatial discretization in the same units as ``position``.
    movement_var
        Variance of the random-walk transition between consecutive
        position bins.  ``None`` (default) lets the library use its
        own data-driven estimate via ``estimate_movement_var``.
    is_training
        Boolean mask of which time bins to use for *encoding* (model
        fitting); decoding is then run over **all** time bins.  If
        ``None``, every time bin is used for both.
    is_compute_acausal
        If ``True`` (default), also run the backward smoother and use
        its (acausal) posterior as the primary output.

    Returns
    -------
    ClusterlessDecoderResult

    References
    ----------
    Denovellis EL, Frank LM, Eden UT (2021), eLife.
    Kloosterman F et al. (2014), J Neural Eng (clusterless framework).
    """
    # Validate shapes BEFORE the optional dep is touched, so callers
    # without the library installed still get the actionable shape
    # error rather than an install-hint for a problem they don't have.
    position, multiunits = _validate_inputs(position, multiunits)
    rtc = _require_clusterless()

    env = rtc.Environment(place_bin_size=float(place_bin_size))
    if movement_var is None:
        transition = rtc.RandomWalk()
    else:
        transition = rtc.RandomWalk(movement_var=float(movement_var))

    decoder = rtc.ClusterlessDecoder(
        environment=env,
        transition_type=transition,
    )
    decoder.fit(position, multiunits, is_training=is_training)
    dataset = decoder.predict(multiunits, is_compute_acausal=bool(is_compute_acausal))

    posterior, causal = _extract_posterior(dataset, prefer_acausal=is_compute_acausal)
    centres = _extract_position_bin_centers(dataset)
    # Posterior shape: (n_time, n_position_bins[0], n_position_bins[1], ...).
    # MAP: argmax over the position axes (everything past axis 0).
    flat = posterior.reshape(posterior.shape[0], -1)
    map_flat = np.argmax(flat, axis=1)
    map_index = np.array(np.unravel_index(map_flat, posterior.shape[1:])).T
    return ClusterlessDecoderResult(
        posterior=posterior,
        map_position=map_index,
        position_bin_centers=centres,
        causal_posterior=causal,
    )


def fit_clusterless_classifier(
    position: np.ndarray,
    multiunits: np.ndarray,
    *,
    place_bin_size: float = 2.0,
    state_names: list[str] | None = None,
    discrete_diagonal: float = 0.98,
    is_training: np.ndarray | None = None,
    is_compute_acausal: bool = True,
) -> ClusterlessClassifierResult:
    """Fit + classify with the clusterless trajectory-classification model.

    Wraps :class:`replay_trajectory_classification.ClusterlessClassifier`
    with a default 2-state setup (``"continuous"`` = local random walk,
    ``"fragmented"`` = uniform) controlled by a diagonal discrete-state
    transition (high self-persistence).  Override ``state_names`` to fit
    a custom number of discrete states; the matching
    ``continuous_transition_types`` is built proportionally.

    Parameters
    ----------
    position, multiunits, is_training
        See :func:`fit_clusterless_decoder`.
    place_bin_size
        Spatial discretization in the same units as ``position``.
    state_names
        Names of the discrete states.  ``None`` (default) gives
        ``["continuous", "fragmented"]``.  Pass at least one name; the
        continuous-transition cell defaults to ``RandomWalk`` per state,
        and the discrete transition is a diagonal matrix with
        self-persistence ``discrete_diagonal``.
    discrete_diagonal
        Self-persistence probability on the discrete-state transition
        matrix (off-diagonals split the remainder).
    is_compute_acausal
        Compute the backward smoother as well as the forward filter.

    Returns
    -------
    ClusterlessClassifierResult

    References
    ----------
    Denovellis EL, Frank LM, Eden UT (2021), eLife — for the
    classification framework integrated with clusterless decoding.
    """
    position, multiunits = _validate_inputs(position, multiunits)
    if state_names is None:
        state_names = ["continuous", "fragmented"]
    if len(state_names) < 1:
        raise ValueError("state_names must contain at least one name")
    rtc = _require_clusterless()

    env = rtc.Environment(place_bin_size=float(place_bin_size))
    # ``continuous_transition_types`` must be a square (n_states x n_states)
    # matrix: entry [i][j] is the continuous-state transition model applied
    # when moving between discrete states i and j.  The diagonal carries
    # each state's own movement model — random walk for the "continuous"-like
    # states, uniform for the "fragmented"-like ones — while the off-diagonal
    # cells use a uniform spatial transition, mirroring the upstream default
    # layout (e.g. ``[[RandomWalk, Uniform], [Uniform, Uniform]]``).
    def _transition_for(name: str):
        return rtc.Uniform() if name.lower() == "fragmented" else rtc.RandomWalk()
    n_states = len(state_names)
    cont_transitions = [
        [
            _transition_for(state_names[i]) if i == j else rtc.Uniform()
            for j in range(n_states)
        ]
        for i in range(n_states)
    ]

    classifier = rtc.ClusterlessClassifier(
        environments=env,
        continuous_transition_types=cont_transitions,
        discrete_transition_type=rtc.DiagonalDiscrete(float(discrete_diagonal)),
    )
    classifier.fit(position, multiunits, is_training=is_training)
    dataset = classifier.predict(
        multiunits,
        is_compute_acausal=bool(is_compute_acausal),
        state_names=state_names,
    )

    posterior, _ = _extract_posterior(dataset, prefer_acausal=is_compute_acausal)
    # Marginal state probabilities: sum out the position axes (everything
    # past axes 0 [time] and 1 [state]).
    if posterior.ndim < 2:
        raise RuntimeError(
            f"classifier posterior has unexpected shape {posterior.shape}; "
            f"expected (n_time, n_states, *position)."
        )
    state_probs = posterior.reshape(posterior.shape[0], posterior.shape[1], -1).sum(axis=-1)
    centres = _extract_position_bin_centers(dataset)
    return ClusterlessClassifierResult(
        posterior=posterior,
        state_probabilities=state_probs,
        state_names=list(state_names),
        position_bin_centers=centres,
    )


__all__ = [
    "ClusterlessDecoderResult",
    "ClusterlessClassifierResult",
    "fit_clusterless_decoder",
    "fit_clusterless_classifier",
]
