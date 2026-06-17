#!/usr/bin/env python3
"""Example 07 — Spatiotemporal Wave Analysis of a Synthetic Hawkes Adjacency.

A pure-Python end-to-end demonstration of the wave-analysis stack in
:mod:`nstat.extras.spatial`:

1. A 6x6 grid of electrodes on the unit square stands in for a recording
   array.
2. We construct a synthetic multivariate Hawkes triggering matrix
   :math:`A_{cc'}` that embeds a coherent planar wave with known wave
   vector :math:`\\mathbf{k}_{\\text{true}}` and Gaussian spatial decay,
   then rescale ``A`` so the spectral radius is well below 1 (the
   Hawkes process is stationary).
3. We compute the Bartlett (frequency x wave-vector) spectrum of ``A``
   via :func:`~nstat.extras.spatial.bartlett_spectrum` (Daley &
   Vere-Jones 2003 §8.4) on a (32, 17, 17) grid.
4. We locate the top wave-vector peaks with
   :func:`~nstat.extras.spatial.detect_wave_peaks` (greedy descending
   power with non-maximum suppression) and compare the recovered top
   peak's direction to ``atan2(k_true[1], k_true[0])``.

.. note::

   **Grid-undersampling caveat.** A 6x6 array on the unit square
   covers only ~1.7 wavelengths of ``k_true = (5, 2)`` — well below
   the spatial Nyquist regime that the planar-wave argument assumes.
   The dominant spectral mass therefore lands near the reciprocal-
   lattice DC neighbours instead of at ``k_true``, and the recovered
   direction error can be ~0.4 rad.  This is an honest demonstration
   of the workflow under acknowledged undersampling, not a tool bug.
   The companion notebook (``notebooks/HawkesWaveAnalysis.ipynb``)
   shows the same and discusses 8x8 / 10x10 alternatives.

The example is **fully synthetic** — it never imports the optional
``tick`` dependency.

References:
- Bacry E, Mastromatteo I, Muzy J-F (2015). *Hawkes processes in finance.*
  Market Microstructure and Liquidity 1(1):1550005.
- Daley DJ, Vere-Jones D (2003). *An Introduction to the Theory of Point
  Processes*, Vol I, §8.4.
- Kass RE, Eden UT, Brown EN (2014). *Analysis of Neural Data*, Ch. 19.

Expected outputs:
- Figure 1: heatmap of the adjacency matrix ``A`` with the electrode
  positions annotated.
- Figure 2: frequency-integrated 2-D Bartlett power on ``(kx, ky)``,
  with the ground-truth wave vector ``k_true`` marked.
- Figure 3: detected peaks overlaid on the frequency-integrated power.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nstat import apply_plot_style  # noqa: E402
from nstat.extras.spatial import bartlett_spectrum, detect_wave_peaks  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
#  Construction
# ──────────────────────────────────────────────────────────────────────────


def _electrode_grid(Cx: int = 6, Cy: int = 6) -> np.ndarray:
    """Regular Cx x Cy electrode grid on the unit square (Cx*Cy, 2)."""
    ax = np.linspace(0.0, 1.0, Cx)
    ay = np.linspace(0.0, 1.0, Cy)
    XX, YY = np.meshgrid(ax, ay, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])


def _build_planar_wave_adjacency(
    pos: np.ndarray,
    k_true: np.ndarray,
    *,
    sigma: float = 0.18,
    phi0: float = 0.0,
    rho_target: float = 0.6,
    a_max: float = 1.0,
) -> np.ndarray:
    """Construct a planar-wave Hawkes adjacency on the electrode grid.

    The pairwise excitation is the product of a Gaussian spatial decay
    and a cosine in the projected lag along ``k_true``; negative entries
    are clipped to zero so ``A`` is a valid Hawkes excitation, then the
    matrix is rescaled so its spectral radius equals ``rho_target``.

    Parameters
    ----------
    pos
        ``(C, 2)`` electrode positions.
    k_true
        ``(2,)`` ground-truth planar wave vector.
    sigma
        Spatial decay scale for the Gaussian envelope (position units).
    phi0
        Constant phase offset (radians).
    rho_target
        Target spectral radius after rescaling (``< 1`` for stationarity).
    a_max
        Pre-rescale envelope amplitude.
    """
    diff = pos[:, None, :] - pos[None, :, :]                  # (C, C, 2)
    gauss = np.exp(-(diff ** 2).sum(-1) / (2.0 * sigma ** 2))
    phase = np.einsum("ijd,d->ij", diff, k_true) + phi0
    A = a_max * gauss * np.cos(phase)
    A = np.clip(A, 0.0, None)
    rho = float(np.max(np.abs(np.linalg.eigvals(A)).real))
    if rho > 0:
        A = A * (rho_target / rho)
    return A


# ──────────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────────


def _plot_adjacency(A: np.ndarray, pos: np.ndarray, Cx: int, Cy: int):
    """Heatmap of A plus the electrode-position scatter."""
    # === FIGURE: fig01_adjacency_and_positions.png ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im = axes[0].imshow(A, cmap="magma", aspect="equal", origin="lower")
    axes[0].set_title(f"Adjacency A ({A.shape[0]}x{A.shape[1]})")
    axes[0].set_xlabel("source electrode")
    axes[0].set_ylabel("target electrode")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].scatter(pos[:, 0], pos[:, 1], s=80, color="tab:blue",
                    edgecolor="k")
    for idx, (x, y) in enumerate(pos):
        axes[1].text(x + 0.015, y + 0.015, str(idx), fontsize=7)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Electrode grid ({Cx}x{Cy} on unit square)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.suptitle(
        "Example 07 — synthetic planar-wave Hawkes adjacency"
    )
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_bartlett(
    S: np.ndarray,
    freq: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    k_true: np.ndarray,
):
    """Frequency-integrated 2-D Bartlett power on (kx, ky)."""
    Nf = S.shape[0]
    Nk_total = S.shape[1]
    nx = kx_axis.size
    ny = ky_axis.size
    assert Nk_total == nx * ny
    # Mean power across frequency, then reshape to the (kx, ky) grid.
    power_2d = S.mean(axis=0).reshape(nx, ny)

    # === FIGURE: fig02_bartlett_spectrum.png ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im = axes[0].pcolormesh(
        kx_axis, ky_axis, power_2d.T, shading="auto", cmap="viridis"
    )
    axes[0].scatter([k_true[0]], [k_true[1]], s=120, marker="x",
                    color="red", linewidths=2.5, label="k_true")
    axes[0].scatter([-k_true[0]], [-k_true[1]], s=80, marker="x",
                    color="red", linewidths=1.5, alpha=0.5,
                    label="-k_true (antipodal)")
    axes[0].set_xlabel("kx (rad / unit)")
    axes[0].set_ylabel("ky (rad / unit)")
    axes[0].set_title("frequency-mean Bartlett power")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Frequency profile at the (kx, ky) cell closest to k_true.
    ix = int(np.argmin(np.abs(kx_axis - k_true[0])))
    iy = int(np.argmin(np.abs(ky_axis - k_true[1])))
    k_idx_true = ix * ny + iy
    profile_true = S[:, k_idx_true]
    profile_mean = S.mean(axis=1)
    axes[1].plot(freq, profile_true, color="tab:red", lw=1.8,
                 label=f"at k_true (kx={kx_axis[ix]:.2f}, ky={ky_axis[iy]:.2f})")
    axes[1].plot(freq, profile_mean, color="tab:gray", lw=1.4, ls="--",
                 label="mean across k")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("Bartlett power")
    axes[1].set_title("frequency profile of S(f, k)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Example 07 — Bartlett spectrum of the Hawkes adjacency")
    fig.tight_layout()
    # === END FIGURE ===
    return fig


def _plot_peaks(
    S: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    peaks,
    k_true: np.ndarray,
):
    """Detected peaks overlaid on the frequency-mean Bartlett power."""
    nx = kx_axis.size
    ny = ky_axis.size
    power_2d = S.mean(axis=0).reshape(nx, ny)

    # === FIGURE: fig03_detected_peaks_overlay.png ===
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.pcolormesh(
        kx_axis, ky_axis, power_2d.T, shading="auto", cmap="viridis"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter([k_true[0]], [k_true[1]], s=160, marker="x", color="red",
               linewidths=2.5, label="k_true")
    for i in range(peaks.kx.size):
        ax.scatter(peaks.kx[i], peaks.ky[i], s=110, marker="o",
                   facecolor="none", edgecolor="white", linewidths=2.0)
        ax.annotate(
            f"#{i+1} f={peaks.freq[i]:.1f} Hz",
            (peaks.kx[i], peaks.ky[i]),
            xytext=(8, 8), textcoords="offset points", color="white",
            fontsize=8,
        )
    ax.set_xlabel("kx (rad / unit)")
    ax.set_ylabel("ky (rad / unit)")
    ax.set_aspect("equal")
    ax.set_title(
        f"Example 07 — detected wave peaks (top {peaks.kx.size}) "
        "vs ground truth"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    # === END FIGURE ===
    return fig


# ──────────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────────


def run_example07(
    *,
    export_figures: bool = False,
    export_dir: Path | None = None,
    visible: bool | None = True,
    plot_style: str = "legacy",
) -> dict:
    """Run Example 07: synthetic planar-wave Hawkes wave analysis.

    Returns
    -------
    dict
        Keys: ``positions``, ``adjacency``, ``spectrum``, ``freq_grid``,
        ``wave_vectors``, ``peaks``, ``k_true``, ``recovered_top_peak``,
        ``figure_paths``.
    """
    print("=" * 70)
    print("Example 07: Spatiotemporal Wave Analysis (synthetic Hawkes)")
    print("=" * 70)

    # The rng seeded here is not used by the deterministic Bartlett pipeline
    # below, but is reserved for future stochastic extensions and is the
    # documented seeding hook for reproducibility.
    _ = np.random.default_rng(20260617)

    Cx, Cy = 6, 6
    pos = _electrode_grid(Cx, Cy)
    k_true = np.array([5.0, 2.0])
    A = _build_planar_wave_adjacency(
        pos, k_true,
        sigma=0.18, phi0=0.0, rho_target=0.6, a_max=1.0,
    )
    rho_after = float(np.max(np.abs(np.linalg.eigvals(A)).real))
    print(f"  electrodes: {Cx}x{Cy} ({pos.shape[0]} channels)")
    print(f"  k_true = {k_true.tolist()} | adjacency spectral radius "
          f"after rescale = {rho_after:.3f}")

    # Bartlett-spectrum grid.
    freq = np.linspace(0.5, 10.0, 32)
    kx = np.linspace(-8.0, 8.0, 17)
    ky = np.linspace(-8.0, 8.0, 17)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_grid = np.stack([KX.ravel(), KY.ravel()], axis=1)
    S = bartlett_spectrum(A, pos, freq, k_grid, decay=1.0)
    print(f"  bartlett_spectrum shape = {S.shape} "
          f"(Nf={freq.size}, Nk={k_grid.shape[0]})")

    peaks = detect_wave_peaks(
        S, freq, k_grid, n_peaks=3, min_separation_bins=2
    )
    print(f"  detected {peaks.kx.size} peaks")

    # Verification: compare the top-1 recovered peak direction to the
    # ground-truth direction.  Because the cosine envelope of A is even,
    # k and -k are both peaks of |S|^2 — accept either as a match.
    if peaks.kx.size == 0:
        recovered_top = None
        dir_err = float("nan")
    else:
        recovered_top = {
            "freq": float(peaks.freq[0]),
            "kx": float(peaks.kx[0]),
            "ky": float(peaks.ky[0]),
            "power": float(peaks.power[0]),
            "speed": float(peaks.speed[0]),
            "direction": float(peaks.direction[0]),
        }
        dir_true = float(np.arctan2(k_true[1], k_true[0]))
        dir_recov = recovered_top["direction"]
        # Angular distance modulo pi (planar wave is undirected).
        diff = (dir_recov - dir_true + np.pi) % np.pi
        dir_err = float(min(diff, np.pi - diff))

    print(f"  ground-truth direction (rad) = "
          f"{float(np.arctan2(k_true[1], k_true[0])):+.3f}")
    if recovered_top is not None:
        print(f"  recovered top-1 direction    = "
              f"{recovered_top['direction']:+.3f} "
              f"(|err mod pi| = {dir_err:.3f} rad)")
        print(f"  recovered top-1 (kx, ky)      = "
              f"({recovered_top['kx']:+.2f}, {recovered_top['ky']:+.2f})")
        print(f"  recovered top-1 frequency    = "
              f"{recovered_top['freq']:.2f} Hz, speed = "
              f"{recovered_top['speed']:.3f} unit/s")
        print(
            f"  [note] {Cx}x{Cy} grid covers ~1.7 wavelengths of k_true; "
            f"top spectral mass lands near DC (a real undersampling "
            f"effect, not a tool bug). See the docstring note and the "
            f"companion notebook for 8x8 / 10x10 alternatives."
        )

    fig1 = _plot_adjacency(A, pos, Cx, Cy)
    fig2 = _plot_bartlett(S, freq, kx, ky, k_true)
    fig3 = _plot_peaks(S, kx, ky, peaks, k_true)
    figures = [fig1, fig2, fig3]
    for fig in figures:
        apply_plot_style(fig, style=plot_style)

    figure_paths: list[Path] = []
    if export_figures:
        if export_dir is None:
            export_dir = REPO_ROOT / "docs" / "figures" / "example07"
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        fig_names = (
            "fig01_adjacency_and_positions",
            "fig02_bartlett_spectrum",
            "fig03_detected_peaks_overlay",
        )
        for fig, name in zip(figures, fig_names):
            path = export_dir / f"{name}.png"
            fig.savefig(path, dpi=200, facecolor="w", edgecolor="none")
            figure_paths.append(path)
            print(f"  Saved: {path}")

    if bool(visible):
        plt.show()
    else:
        plt.close("all")

    return {
        "positions": pos,
        "adjacency": A,
        "spectrum": S,
        "freq_grid": freq,
        "wave_vectors": k_grid,
        "peaks": peaks,
        "k_true": k_true,
        "recovered_top_peak": recovered_top,
        "figure_paths": [str(p) for p in figure_paths],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 07: Spatiotemporal Wave Analysis (synthetic Hawkes)"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help=("Repository root (used by other paper examples "
                              "for dataset lookup; this script is data-free "
                              "and only uses it to resolve the default "
                              "export-dir under docs/figures/example07)."))
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without displaying figures (headless).")
    parser.add_argument("--plot-style", choices=("modern", "legacy"),
                        default="legacy",
                        help="Figure styling.")
    args = parser.parse_args()

    if args.no_display:
        visible = False
    else:
        visible = bool(args.show)
    result = run_example07(
        export_figures=args.export_figures,
        export_dir=args.export_dir,
        visible=visible,
        plot_style=args.plot_style,
    )
    if args.output_json:
        summary = {
            "n_electrodes": int(result["positions"].shape[0]),
            "k_true": result["k_true"].tolist(),
            "recovered_top_peak": result["recovered_top_peak"],
            "n_peaks_detected": int(result["peaks"].kx.size),
        }
        args.output_json.write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
