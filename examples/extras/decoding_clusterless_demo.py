"""Demo: clusterless vs. sorted point-process decoding as spike sorting degrades.

**Question** (design spec S4.1, intracortical motor-BCI thread): can we decode
kinematics/position directly from **unsorted** multiunit events + their
waveform "marks", and does clusterless decoding match/beat a conventional
sorted-spike decoder as spike-sorting quality degrades?

Ground truth
------------
Four place-/kinematic-tuned units ride a 1-D back-and-forth trajectory --
the same substrate as nSTAT's own point-process state-space decoding
lineage (PPAF/PPHF, Eden et al. 2004).  A single tetrode detects at most
one multiunit event per time bin; which unit "generated" that event is
drawn from each unit's instantaneous place-tuned rate, and the event's
4-channel waveform mark is a noisy draw around that unit's mark centroid.
Mark centroids are spaced close enough to overlap realistically --
exactly the ambiguous-cluster situation clusterless decoding targets
(Kloosterman et al. 2014; Deng et al. 2015).  The per-event "true unit"
label is tracked *only* to build the comparator's sorted spike trains
below -- the clusterless decoder never sees it.

Recovery
--------
1. ``fit_clusterless_decoder`` decodes position directly from the raw,
   never-sorted marks.  It never learns which unit produced an event, so
   its accuracy cannot depend on how well spikes are subsequently sorted.
2. A **sorted-spike comparator**, built from
   ``replay_trajectory_classification.SortedSpikesDecoder`` (the same
   upstream state-space framework, Denovellis et al. 2021 -- differing
   from the clusterless decoder only in its observation model), decodes
   from per-unit spike counts obtained by corrupting the true unit-identity
   labels at a tunable **sort-error rate**: a stand-in for cumulative
   merge/split spike-sorting mistakes.  As the sort-error rate climbs, the
   sorted decoder degrades gracefully at first and then collapses once
   corrupted labels dominate, while the clusterless decoder is unaffected
   (its inputs never depended on unit identity in the first place).

References
----------
- Eden UT, Frank LM, Barbieri R, Solo V, Brown EN (2004), Neural Comput
  16:971 -- the point-process state-space decoding lineage (PPAF/PPHF)
  both decoders compared here descend from.
- Deng X, Liu DF, Kay K, Frank LM, Eden UT (2015), Neural Comput 27:1438 --
  clusterless marked point-process decoding.
- Kloosterman F, Layton SP, Chen Z, Wilson MA (2014), J Neurophysiol
  111:217 -- Bayesian decoding from unsorted spikes.
- Denovellis EL, Gillespie AK, Coulter ME, Sosa M, Chung JE, Eden UT,
  Frank LM (2021), eLife 10:e64505 -- the state-space clusterless/sorted
  decoding framework wrapped here (``replay_trajectory_classification``).

Cross-links
-----------
``examples/extras/decoding_place_field_demo.py`` -- the fully-sorted,
core-``nstat`` PPAF decode that this comparator's "sorted" baseline
echoes.  ``examples/extras/validation_pykalman_demo.py`` -- the clinical
velocity-Kalman decoder's own degrade-under-drift story (chronic
recording drift rather than acute sort-error).

Run::

    pip install nstat-toolbox[clusterless]   # pulls JAX (~200 MB)
    python examples/extras/decoding_clusterless_demo.py
    python examples/extras/decoding_clusterless_demo.py --export-figures --no-display
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Sort-error-rate sweep shared by the console summary and the figures.
_ERROR_RATES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
_N_REPEATS = 6  # corruption draws averaged per error-rate point


def _make_synthetic_place_tuned_units(
    n_time: int = 400,
    n_marks: int = 4,
    n_units: int = 4,
    seed: int = 0,
    sigma_pf: float = 18.0,
    peak_rate: float = 0.5,
    mark_sep: float = 2.5,
    mark_noise: float = 1.0,
):
    """Place-/kinematic-tuned units emitting ambiguous unsorted events.

    A 1-D back-and-forth trajectory drives ``n_units`` Gaussian-tuned
    place fields.  A single tetrode detects at most one multiunit event
    per raw time bin; the responsible unit is drawn in proportion to each
    unit's instantaneous rate (so nearby units genuinely compete for the
    same detected event, as on a real tetrode), and the event's waveform
    mark is a noisy draw around that unit's mark centroid.  Centroids are
    spaced by ``mark_sep`` with per-channel noise ``mark_noise`` -- close
    enough that adjacent units' clusters overlap in mark-space, mirroring
    genuine tetrode cluster overlap (Kloosterman 2014; Deng 2015).

    Returns
    -------
    position : ndarray, shape (n_time, 1)
    multiunits : ndarray, shape (n_time, n_marks, 1)
        NaN denotes "no event this bin" (the upstream convention).
    true_unit : ndarray of int, shape (n_time,)
        Which unit generated the event at each bin; -1 if none.  Used
        only to build/corrupt the sorted-decoder comparator below -- the
        clusterless decoder is never given this array.
    field_centers : ndarray, shape (n_units,)
        Each unit's preferred position (Gaussian tuning-curve centre).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    position = 50.0 + 45.0 * np.sin(2 * np.pi * t / n_time)

    field_centers = np.linspace(15.0, 85.0, n_units)
    rates = peak_rate * np.exp(
        -((position[:, None] - field_centers[None, :]) ** 2) / (2 * sigma_pf ** 2)
    )  # (n_time, n_units)
    total_rate = np.clip(rates.sum(axis=1), 0.0, 0.95)

    mark_centroids = mark_sep * np.arange(n_units)[:, None] * np.ones((n_units, n_marks))

    multiunits = np.full((n_time, n_marks, 1), np.nan)
    true_unit = np.full(n_time, -1, dtype=int)
    has_event = rng.random(n_time) < total_rate
    for t_i in np.flatnonzero(has_event):
        probs = rates[t_i] / rates[t_i].sum()
        unit = rng.choice(n_units, p=probs)
        true_unit[t_i] = unit
        multiunits[t_i, :, 0] = rng.normal(loc=mark_centroids[unit], scale=mark_noise)
    return position.reshape(-1, 1), multiunits, true_unit, field_centers


def _corrupt_sorted_labels(
    true_unit: np.ndarray, n_units: int, sort_error_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate a spike sorter's cluster assignment at a given error rate.

    With probability ``sort_error_rate`` the assigned label is replaced by
    a label drawn uniformly at random from all ``n_units`` (independent of
    the true label) -- a simple, standard "symmetric label noise" model
    for cumulative merge/split misassignment.  At ``sort_error_rate=0``
    every event keeps its true label (a perfect sorter); at
    ``sort_error_rate=1`` labels carry zero information about identity.
    No-event bins (``true_unit == -1``) are left untouched.
    """
    labels = true_unit.copy()
    has_event = true_unit >= 0
    idx = np.flatnonzero(has_event)
    corrupt = rng.random(idx.shape[0]) < sort_error_rate
    random_labels = rng.integers(0, n_units, size=idx.shape[0])
    labels[idx] = np.where(corrupt, random_labels, labels[idx])
    return labels


def _labels_to_binary_spikes(labels: np.ndarray, n_units: int) -> np.ndarray:
    """(n_time,) unit-label array -> (n_time, n_units) binary spike matrix."""
    n_time = labels.shape[0]
    spikes = np.zeros((n_time, n_units))
    has_event = labels >= 0
    spikes[has_event, labels[has_event]] = 1.0
    return spikes


def _decode_from_dataset(dataset):
    """Extract ``(posterior, bin_centers, map_position)`` from an upstream
    ``predict()`` xarray.Dataset (shared shape convention across
    ``ClusterlessDecoder`` and ``SortedSpikesDecoder``)."""
    has_acausal = "acausal_posterior" in dataset
    posterior = np.asarray(
        dataset["acausal_posterior" if has_acausal else "causal_posterior"].values
    )
    bin_centers = np.asarray(dataset.coords["position"].values)
    flat = posterior.reshape(posterior.shape[0], -1)
    map_position = bin_centers[np.argmax(flat, axis=1)]
    return posterior, bin_centers, map_position


def _demo_decoder() -> int:
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    position, multiunits, _, _ = _make_synthetic_place_tuned_units(n_time=200)
    result = fit_clusterless_decoder(position, multiunits, place_bin_size=5.0)
    sums = result.posterior.reshape(position.shape[0], -1).sum(axis=1)
    print(f"[Decoder]    posterior shape={result.posterior.shape} "
          f"row-sums in [{sums.min():.4f}, {sums.max():.4f}] (≈1)")
    return 0 if np.all(np.isfinite(result.posterior)) else 1


def _demo_classifier() -> int:
    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_classifier

    position, multiunits, _, _ = _make_synthetic_place_tuned_units(n_time=150)
    result = fit_clusterless_classifier(
        position, multiunits,
        place_bin_size=5.0,
        state_names=["continuous", "fragmented"],
    )
    marginal_means = result.state_probabilities.mean(axis=0)
    print(f"[Classifier] states={result.state_names} "
          f"mean P(state) over time={np.round(marginal_means, 3).tolist()}")
    return 0 if np.allclose(result.state_probabilities.sum(axis=1), 1.0, atol=1e-3) else 1


def _run_sorted_vs_clusterless_comparison(seed: int = 0) -> dict:
    """Core comparator: one clusterless decode vs. a sort-error-rate sweep
    of sorted decodes on the *same* synthetic ground truth.

    The clusterless decoder is fit exactly once -- its accuracy cannot
    depend on the (synthetic, decoy) sort-error rate, since it never uses
    the per-event unit labels at all.  The sorted decoder is re-fit at
    each ``sort_error_rate`` in ``_ERROR_RATES`` on labels corrupted by
    :func:`_corrupt_sorted_labels`, averaged over ``_N_REPEATS`` draws.
    """
    import replay_trajectory_classification as rtc

    from nstat.extras.decoding.clusterless_bridge import fit_clusterless_decoder

    place_bin_size = 2.0
    movement_var = 3.0

    position, multiunits, true_unit, field_centers = _make_synthetic_place_tuned_units(
        n_time=400, seed=seed,
    )
    n_units = field_centers.shape[0]
    true_position = position.reshape(-1)
    n_events = int((true_unit >= 0).sum())

    # --- clusterless: fit + decode once; never sees true_unit ---
    cl_result = fit_clusterless_decoder(
        position, multiunits, place_bin_size=place_bin_size, movement_var=movement_var,
    )
    cl_bin_centers = cl_result.position_bin_centers[0]
    cl_map_position = cl_bin_centers[cl_result.map_position[:, 0]]
    clusterless_error = float(np.mean(np.abs(cl_map_position - true_position)))

    # --- sorted comparator: sweep the simulated sort-error rate ---
    rng_corrupt = np.random.default_rng(seed + 1)
    sorted_errors: list[float] = []
    example_posteriors: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    p_low, p_high = _ERROR_RATES[0], _ERROR_RATES[-1]
    for p in _ERROR_RATES:
        errs = []
        for rep in range(_N_REPEATS):
            labels = _corrupt_sorted_labels(true_unit, n_units, p, rng_corrupt)
            spikes = _labels_to_binary_spikes(labels, n_units)
            sorted_decoder = rtc.SortedSpikesDecoder(
                environment=rtc.Environment(place_bin_size=place_bin_size),
                transition_type=rtc.RandomWalk(movement_var=movement_var),
            )
            sorted_decoder.fit(position, spikes)
            dataset = sorted_decoder.predict(spikes, is_compute_acausal=True)
            posterior, bin_centers, map_position = _decode_from_dataset(dataset)
            errs.append(float(np.mean(np.abs(map_position - true_position))))
            if rep == 0 and p in (p_low, p_high):
                example_posteriors[p] = (posterior, bin_centers, map_position)
        sorted_errors.append(float(np.mean(errs)))

    recovery_ok = clusterless_error < sorted_errors[-1]

    print(
        f"[Comparison] {n_events}/{position.shape[0]} bins carry an event "
        f"({n_events / position.shape[0]:.0%})"
    )
    print(f"  clusterless MAE (sort-error independent) = {clusterless_error:.3f}")
    for p, e in zip(_ERROR_RATES, sorted_errors):
        flag = "  <-- exceeds clusterless" if e > clusterless_error else ""
        print(f"  sorted MAE @ sort-error={p:4.2f}  = {e:6.3f}{flag}")
    print(
        f"  recovery: clusterless < sorted at sort-error={_ERROR_RATES[-1]:g} "
        f"-> {recovery_ok}"
    )

    return {
        "position": position,
        "true_position": true_position,
        "field_centers": field_centers,
        "cl_posterior": cl_result.posterior.reshape(position.shape[0], -1),
        "cl_bin_centers": cl_bin_centers,
        "error_rates": _ERROR_RATES,
        "clusterless_error": clusterless_error,
        "sorted_errors": sorted_errors,
        "example_posteriors": example_posteriors,
        "p_low": p_low,
        "p_high": p_high,
        "recovery_ok": recovery_ok,
    }


def _plot_trajectory_and_posteriors(comparison: dict):
    import matplotlib.pyplot as plt

    position = comparison["true_position"]
    n_time = position.shape[0]
    t_axis = np.arange(n_time)
    p_low, p_high = comparison["p_low"], comparison["p_high"]
    post_low, bins_low, _ = comparison["example_posteriors"][p_low]
    post_high, bins_high, _ = comparison["example_posteriors"][p_high]

    # === FIGURE: fig01_trajectory_posteriors.png ===
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharey=True)

    axes[0].pcolormesh(
        t_axis, comparison["cl_bin_centers"], comparison["cl_posterior"].T,
        shading="auto", cmap="viridis",
    )
    axes[0].plot(t_axis, position, color="white", lw=1.3, label="true position")
    axes[0].set_title("Clusterless posterior\n(unsorted marks -- never depends on sort quality)")
    axes[0].set_xlabel("time [bin]")
    axes[0].set_ylabel("position")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].pcolormesh(t_axis, bins_low, post_low.T, shading="auto", cmap="viridis")
    axes[1].plot(t_axis, position, color="white", lw=1.3)
    axes[1].set_title(f"Sorted posterior\n(sort-error rate = {p_low:g}, perfect sort)")
    axes[1].set_xlabel("time [bin]")

    axes[2].pcolormesh(t_axis, bins_high, post_high.T, shading="auto", cmap="viridis")
    axes[2].plot(t_axis, position, color="white", lw=1.3)
    axes[2].set_title(f"Sorted posterior\n(sort-error rate = {p_high:g}, scrambled sort)")
    axes[2].set_xlabel("time [bin]")

    fig.suptitle(
        "True trajectory vs. decoded posterior: clusterless vs. corrupted sorted spikes"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_decode_error_curve(comparison: dict):
    import matplotlib.pyplot as plt

    # === FIGURE: fig02_decode_error_vs_sort_rate.png ===
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(
        comparison["error_rates"], comparison["sorted_errors"], "o-",
        color="tab:red", label="sorted decoder (SortedSpikesDecoder)",
    )
    ax.axhline(
        comparison["clusterless_error"], color="tab:blue", ls="--",
        label="clusterless decoder (fit_clusterless_decoder)",
    )
    ax.set_xlabel("simulated sort-error rate (merge/split)")
    ax.set_ylabel("mean |decode error| [position units]")
    ax.set_title("Decode error vs. spike-sorting error rate")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Demo of nstat.extras.decoding.clusterless_bridge vs. a "
            "sort-error-corrupted sorted-spike decoder"
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument(
        "--export-dir", type=Path,
        default=Path("docs/figures/extras/decoding_clusterless"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    try:
        import nstat.extras.decoding.clusterless_bridge  # noqa: F401
    except ImportError as exc:
        print(f"Install required: {exc}")
        return 1

    print(
        "nstat.extras.decoding.clusterless_bridge — clusterless vs. sorted "
        "decoding under simulated spike-sorting error\n"
    )
    try:
        rc = _demo_decoder()
        rc |= _demo_classifier()
        comparison = _run_sorted_vs_clusterless_comparison(seed=args.seed)
        rc |= 0 if comparison["recovery_ok"] else 1

        if args.export_figures or args.show or not args.no_display:
            fig_traj = _plot_trajectory_and_posteriors(comparison)
            fig_err = _plot_decode_error_curve(comparison)
            if args.export_figures:
                args.export_dir.mkdir(parents=True, exist_ok=True)
                fig_traj.savefig(
                    args.export_dir / "fig01_trajectory_posteriors.png", dpi=160
                )
                fig_err.savefig(
                    args.export_dir / "fig02_decode_error_vs_sort_rate.png", dpi=160
                )
                print(f"  saved figures under {args.export_dir}")
            if args.show:
                import matplotlib.pyplot as plt
                plt.show()
            elif args.no_display:
                import matplotlib.pyplot as plt
                plt.close("all")
    except ImportError as exc:
        print(f"replay_trajectory_classification missing: {exc}")
        return 1

    print("\nAll clusterless routines ran." if rc == 0
          else "\nA routine reported a problem.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
