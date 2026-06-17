"""Gaussian-Process Factor Analysis bridge via Elephant.

Wraps :class:`elephant.gpfa.GPFA` (Yu et al. 2009, *J. Neurophysiol.*
102(1)) — smooth low-dimensional latent trajectories inferred from
simultaneous multi-trial spike trains.

Scope
-----
- :class:`GPFAConfig` — frozen dataclass capturing the fit hyper-parameters.
- :func:`fit_gpfa` — main entry point.  Accepts ``list[Trial]`` (native
  nstat) or ``list[list[neo.SpikeTrain]]`` (Elephant's native shape).
- :class:`GPFAResult` — frozen dataclass with per-trial latent
  trajectories ``(n_bins, x_dim)`` plus the underlying
  ``elephant.gpfa.GPFA`` instance (exposed so callers can re-
  ``transform()`` held-out data).

GPFA has no MATLAB nSTAT counterpart; this is a pure Python-only
extension and is opt-in via ``pip install nstat-toolbox[latents]``.

Install
-------

.. code-block:: bash

    pip install nstat-toolbox[latents]

Pulls Elephant (>=1.2) + Neo + quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nstat.extras._lazy import require_optional, require_optionals

if TYPE_CHECKING:
    import neo

    from nstat.trial import Trial


@dataclass(frozen=True)
class GPFAConfig:
    """Configuration for a GPFA fit.

    Parameters
    ----------
    x_dim : int
        Number of latent dimensions to recover.
    bin_size_s : float
        Discretisation bin size in seconds.  Default 0.02 (20 ms — the
        Yu et al. 2009 default).
    em_max_iter : int
        Maximum EM iterations.  Default 500.
    em_tol : float
        Relative log-likelihood tolerance for convergence.  Default 1e-8.
    min_var_frac : float
        Minimum variance per latent dimension (numerical stability
        floor).  Default 0.01.
    """

    x_dim: int
    bin_size_s: float = 0.02
    em_max_iter: int = 500
    em_tol: float = 1e-8
    min_var_frac: float = 0.01

    def __post_init__(self) -> None:
        if int(self.x_dim) < 1:
            raise ValueError(f"x_dim must be >= 1; got {self.x_dim}")
        if float(self.bin_size_s) <= 0.0:
            raise ValueError(
                f"bin_size_s must be > 0 (seconds); got {self.bin_size_s}"
            )
        if int(self.em_max_iter) < 1:
            raise ValueError(
                f"em_max_iter must be >= 1; got {self.em_max_iter}"
            )
        if float(self.em_tol) <= 0.0:
            raise ValueError(f"em_tol must be > 0; got {self.em_tol}")
        if not (0.0 < float(self.min_var_frac) < 1.0):
            raise ValueError(
                "min_var_frac must satisfy 0 < min_var_frac < 1; "
                f"got {self.min_var_frac}"
            )


@dataclass(frozen=True)
class GPFAResult:
    """Fitted GPFA model and latent trajectories.

    Attributes
    ----------
    latent_trajectories : list of (n_time_bins, x_dim) ndarrays
        Per-trial latent trajectories.  Axis order is
        ``(time, latent)`` — the nstat population-data convention.
        Elephant's native order is ``(latent, time)``; this bridge
        transposes for consistency with other nstat extras.
    x_dim : int
        Number of recovered latent dimensions.
    bin_size_s : float
        Bin size used for fitting (seconds).
    n_trials : int
        Number of trials the model was fit on.
    log_likelihood : float or None
        Final EM log-likelihood, or ``None`` if elephant did not surface
        a finite value.  (Elephant reports the LL only every
        ``freq_ll`` iterations; intermediate entries are ``NaN`` — the
        bridge picks the last finite entry.)
    elephant_model : object
        The underlying ``elephant.gpfa.GPFA`` fitted instance, exposed
        so callers can re-``transform()`` held-out data via
        ``result.elephant_model.transform(new_spiketrains)``.
    """

    latent_trajectories: list[np.ndarray]
    x_dim: int
    bin_size_s: float
    n_trials: int
    log_likelihood: float | None
    elephant_model: object  # opaque; type avoided to keep elephant lazy


def _trials_to_neo_spiketrains(
    trials: list["Trial"],
) -> list[list["neo.SpikeTrain"]]:
    """Convert nstat :class:`Trial` objects to elephant's expected shape.

    Each trial maps to one Neo Segment via the existing
    :func:`nstat.extras.interop.neo.to_neo_segment` helper; we then
    extract its ``spiketrains`` list.
    """
    # Lazy import to avoid a hard dependency on the interop subpackage at
    # module load time (and so callers without ``Trial`` inputs never
    # pay the import cost).
    from nstat.extras.interop.neo import to_neo_segment

    out: list[list["neo.SpikeTrain"]] = []
    for trial in trials:
        segment = to_neo_segment(trial.nspikeColl)
        # Neo's ``Segment.spiketrains`` is a ``SpikeTrainList``; coerce
        # to a plain Python list so elephant's union-typed signature
        # accepts it on every supported Neo version.
        out.append(list(segment.spiketrains))
    return out


def _is_list_of_neo_spiketrain_lists(obj: Any) -> bool:
    """Heuristic: ``obj`` is a non-empty list whose first element is a
    list whose first element looks like a neo SpikeTrain.

    Avoids importing ``neo`` until the lazy gate fires.  We only check
    the FIRST inner-element duck type — sufficient because elephant
    itself does no per-element validation either.
    """
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    if not isinstance(first, list) or not first:
        return False
    inner = first[0]
    # Neo SpikeTrain duck-typing: carries times + t_start + t_stop.
    return all(hasattr(inner, attr) for attr in ("times", "t_start", "t_stop"))


def _is_list_of_trials(obj: Any) -> bool:
    """Heuristic: ``obj`` is a non-empty list whose first element is an
    nstat ``Trial`` (has the ``nspikeColl`` attribute).
    """
    if not isinstance(obj, list) or not obj:
        return False
    return hasattr(obj[0], "nspikeColl")


def fit_gpfa(
    spike_trains: Any,
    *,
    config: GPFAConfig | None = None,
    seed: int | None = None,
) -> GPFAResult:
    """Fit a GPFA model to multi-trial spike data.

    Parameters
    ----------
    spike_trains
        Either a ``list[Trial]`` of native nstat trial objects (each
        carrying an ``nspikeColl`` :class:`SpikeTrainCollection`), or a
        ``list[list[neo.SpikeTrain]]`` matching Elephant's native input
        shape.  Any other shape raises :class:`TypeError`.
    config
        Fit configuration.  ``None`` defaults to
        ``GPFAConfig(x_dim=3)`` (the Elephant default).
    seed
        Optional integer seed.  When provided, the legacy numpy global
        seed is temporarily set so the FA-init + EM fit are reproducible;
        the caller's previous RNG state is restored on exit.

    Returns
    -------
    GPFAResult

    Raises
    ------
    TypeError
        If ``spike_trains`` is not one of the accepted shapes.
    ValueError
        If ``len(spike_trains) < 2`` — GPFA's EM covariance estimation
        is degenerate on a single trial (Yu et al. 2009 §2.3).
    ImportError
        If the ``elephant`` / ``neo`` / ``quantities`` optional
        dependency is missing.  The error names the install command.

    Notes
    -----
    Elephant's :class:`elephant.gpfa.GPFA` consumes the module-level
    numpy random state (no ``rng=`` parameter).  When ``seed`` is
    provided, this bridge wraps the fit in
    ``np.random.get_state()`` / ``np.random.set_state()`` so reproducibility
    works AND the caller's RNG context is preserved — a deliberate,
    isolated exception to the "default_rng-only" convention.
    """
    if config is None:
        config = GPFAConfig(x_dim=3)

    # ----- Input-shape dispatch (no neo import before the lazy gate) -----
    if _is_list_of_trials(spike_trains):
        n_trials = len(spike_trains)
        _input_kind = "trials"
    elif _is_list_of_neo_spiketrain_lists(spike_trains):
        n_trials = len(spike_trains)
        _input_kind = "neo"
    else:
        raise TypeError(
            "fit_gpfa expects either a list[Trial] (native nstat) or a "
            "list[list[neo.SpikeTrain]] (Elephant native shape); got "
            f"{type(spike_trains).__name__}"
        )

    if n_trials < 2:
        raise ValueError(
            "GPFA requires >= 2 trials; covariance estimation is "
            "degenerate on a single trial (Yu et al. 2009 §2.3)"
        )

    # ----- Lazy-import the optional stack ---------------------------------
    # Gate elephant first so the error message names that package (rather
    # than the neo/quantities transitive deps) when it's the missing one.
    require_optional("elephant", install_key="latents")
    _neo, pq = require_optionals(
        "neo", "quantities", install_key="latents"
    )
    from elephant.gpfa import GPFA

    # ----- Build neo input ------------------------------------------------
    if _input_kind == "trials":
        neo_input = _trials_to_neo_spiketrains(spike_trains)
    else:
        neo_input = spike_trains

    # ----- Instantiate the elephant model --------------------------------
    gpfa = GPFA(
        bin_size=config.bin_size_s * pq.s,
        x_dim=int(config.x_dim),
        em_max_iters=int(config.em_max_iter),
        em_tol=float(config.em_tol),
        min_var_frac=float(config.min_var_frac),
        tau_init=100.0 * pq.ms,
        eps_init=1.0e-3,
        verbose=False,
    )

    # ----- Fit + transform, with optional reproducible seed wrapper ------
    #
    # Elephant's GPFA consumes the module-level numpy random state (no
    # ``rng=`` parameter), so we save/restore the legacy state around
    # the call — the caller's RNG context is preserved.  This is the
    # SOLE legacy-``np.random.seed`` use in nstat; default_rng-only is
    # the convention everywhere else.
    if seed is not None:
        old_state = np.random.get_state()
        np.random.seed(int(seed))
        try:
            gpfa.fit(neo_input)
            latents_native = gpfa.transform(neo_input)
        finally:
            np.random.set_state(old_state)
    else:
        gpfa.fit(neo_input)
        latents_native = gpfa.transform(neo_input)

    # ----- Transpose latents to (n_bins, x_dim) --------------------------
    # Elephant returns a length-n_trials object-array of (x_dim, n_bins)
    # ndarrays.  The nstat convention is (time, feature); transpose each.
    latent_trajectories = [
        np.asarray(traj, dtype=float).T for traj in latents_native
    ]

    # ----- Pull the final log-likelihood from fit_info -------------------
    ll_value: float | None = None
    fit_info = getattr(gpfa, "fit_info", None)
    if isinstance(fit_info, dict):
        ll_trace = fit_info.get("log_likelihoods")
        if ll_trace is not None:
            arr = np.asarray(ll_trace, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                ll_value = float(finite[-1])

    return GPFAResult(
        latent_trajectories=latent_trajectories,
        x_dim=int(config.x_dim),
        bin_size_s=float(config.bin_size_s),
        n_trials=int(n_trials),
        log_likelihood=ll_value,
        elephant_model=gpfa,
    )


__all__ = ["GPFAConfig", "GPFAResult", "fit_gpfa"]
