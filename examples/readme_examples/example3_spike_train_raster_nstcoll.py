import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nstat.compat.matlab import CIF, Covariate


def main() -> None:
    np.random.seed(0)

    dt = 0.001
    duration_s = 2.0
    n_units = 20
    time = np.arange(0.0, duration_s + 0.5 * dt, dt, dtype=float)

    lambda_t = (
        9.0
        + 4.0 * np.sin(2.0 * np.pi * 1.5 * time)
        + 2.0 * np.sin(2.0 * np.pi * 4.0 * time + 0.25)
    )
    lambda_t = np.clip(lambda_t, 0.1, None)

    lambda_cov = Covariate(
        time=time,
        data=lambda_t,
        name="Lambda(t)",
        units="spikes/s",
        labels=["lambda"],
    )

    coll = CIF.simulateCIFByThinningFromLambda(lambda_cov, n_units, dt)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plt.sca(ax)
    coll.plot()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit index")
    ax.set_title("Spike-train raster (nstColl.plot)")
    ax.set_ylim(0.5, n_units + 0.5)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "readme_example3_spike_train_raster_nstcoll.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
