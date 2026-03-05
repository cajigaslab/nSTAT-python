import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import CIF, Covariate, nspikeTrain, nstColl


def _build_lambda() -> tuple[np.ndarray, np.ndarray, float]:
    duration_s = 10.0
    dt = 0.001
    t = np.arange(0.0, duration_s + dt, dt, dtype=float)

    f_hz = 0.5
    baseline_hz = 15.0
    amp_hz = 10.0
    lam = np.clip(baseline_hz + amp_hz * np.sin(2.0 * np.pi * f_hz * t), 0.2, None)
    return t, lam, dt


def _ensure_collection(spikes_obj: object, t_start: float, t_end: float) -> nstColl:
    if isinstance(spikes_obj, nstColl):
        return spikes_obj
    if hasattr(spikes_obj, "plot") and hasattr(spikes_obj, "trains"):
        return spikes_obj

    trains: list[nspikeTrain] = []
    if hasattr(spikes_obj, "trains"):
        raw_trains = getattr(spikes_obj, "trains")
    elif isinstance(spikes_obj, (list, tuple)):
        raw_trains = spikes_obj
    else:
        raw_trains = []

    for i, raw in enumerate(raw_trains):
        if hasattr(raw, "spike_times"):
            sp = np.asarray(raw.spike_times, dtype=float).reshape(-1)
        else:
            sp = np.asarray(raw, dtype=float).reshape(-1)
        trains.append(nspikeTrain(spike_times=sp, t_start=t_start, t_end=t_end, name=f"unit_{i+1}"))

    return nstColl(trains)


def main() -> None:
    np.random.seed(0)

    t, lam, dt = _build_lambda()
    lambda_cov = Covariate(time=t, data=lam, name="Lambda", units="spikes/s", labels=["lambda"])

    n_units = 20
    spikes_coll = CIF.simulateCIFByThinningFromLambda(lambda_cov, n_units, dt)
    coll = _ensure_collection(spikes_coll, t_start=float(t[0]), t_end=float(t[-1]))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plt.sca(ax)
    coll.plot()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit index")
    ax.set_title("Spike-train collection raster (nstColl.plot)")
    ax.set_ylim(0.5, n_units + 0.5)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "readme_example3_nstcoll_raster.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
