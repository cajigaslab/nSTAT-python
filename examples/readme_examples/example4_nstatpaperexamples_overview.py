import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from nstat.paper_examples import run_paper_examples


def main() -> None:
    repo_root = PKG_ROOT
    _, payloads = run_paper_examples(repo_root, return_plot_data=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    e2 = payloads["experiment2"]
    ax = axes[0, 0]
    ax.plot(e2["time_binned_s"], e2["obs_rate_hz"], color="tab:blue", lw=1.2, label="Observed")
    ax.plot(e2["time_binned_s"], e2["rate3_binned_hz"], color="tab:red", lw=1.2, label="Stimulus + history GLM")
    ax.set_title("Whisker stimulus example")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    e3 = payloads["experiment3"]
    ax = axes[0, 1]
    ax.plot(e3["time_s"], e3["true_rate_hz"], color="tab:gray", lw=1.2, label="True rate")
    ax.step(
        e3["psth_bin_centers_s"],
        e3["psth_rate_hz"],
        where="mid",
        color="tab:orange",
        lw=1.6,
        label="PSTH",
    )
    ax.set_title("PSTH example")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (Hz)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    e4 = payloads["experiment4"]
    ax = axes[1, 0]
    ax.plot(e4["x_pos"], e4["y_pos"], color="tab:blue", alpha=0.35, lw=0.8)
    spike_x = np.interp(e4["first_cell_spike_times_s"], e4["time_s"], e4["x_pos"])
    spike_y = np.interp(e4["first_cell_spike_times_s"], e4["time_s"], e4["y_pos"])
    ax.scatter(spike_x, spike_y, s=5, color="tab:red", alpha=0.5)
    ax.set_title("Place-cell trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    e5 = payloads["experiment5"]
    ax = axes[1, 1]
    ax.plot(e5["time_s"], e5["stimulus"], color="tab:blue", lw=1.4, label="True stimulus")
    ax.plot(e5["time_s"], e5["decoded"], color="tab:orange", lw=1.2, label="Decoded")
    ax.fill_between(e5["time_s"], e5["ci_low"], e5["ci_high"], color="tab:orange", alpha=0.2, label="95% CI")
    ax.set_title("Point-process decoding example")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("stimulus")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    out_dir = THIS_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "readme_example4_nstatpaperexamples_overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
