"""Deterministic figure tracking for notebook examples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure


# MATLAB R2014b+ "lines" color order
MATLAB_LINES: list[str] = [
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # light blue
    "#A2142F",  # dark red
]

# Backwards-compatible private alias — pre-existing call sites within this
# module use the underscored name. Keep both pointing at the same list so
# callers that mutate or compare either name stay in sync.
_MATLAB_LINES = MATLAB_LINES


def matlab_palette(n: int = 7) -> list[str]:
    """Return the first ``n`` colors of MATLAB's default ``lines()`` palette.

    Matplotlib's default cycle (``"C0"``, ``"C1"``, ...) and MATLAB's default
    line colors differ subtly — Python defaults are ``#1f77b4`` / ``#ff7f0e``
    while MATLAB's ``lines(2)`` returns ``#0072BD`` / ``#D95319``. For figures
    that need to look-alike against a MATLAB helpfile reference, use this
    helper instead of the matplotlib defaults::

        from nstat.notebook_figures import matlab_palette
        ax.bar(["A", "B"], [1, 2], color=matlab_palette(2))

    Parameters
    ----------
    n : int, default 7
        Number of colors to return (1 <= n <= 7).

    Returns
    -------
    list[str]
        A list of ``n`` hex color strings.

    Raises
    ------
    ValueError
        If ``n`` is outside the supported range [1, 7].

    Notes
    -----
    Source: MATLAB R2014b+ default ``lines(7)`` colormap. The 7 colors are the
    canonical line color order; index 7+ wraps around modulo 7 in MATLAB.
    """
    if not 1 <= n <= 7:
        raise ValueError(f"n must be in [1, 7], got {n}")
    return list(MATLAB_LINES[:n])


def _matlab_style() -> dict:
    """Return a dict of matplotlib rcParams matching MATLAB defaults."""
    return {
        "axes.prop_cycle": mpl.cycler(color=_MATLAB_LINES),
        "image.cmap": "jet",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#1f1f1f",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": "#bfbfbf",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "xtick.color": "#1f1f1f",
        "ytick.color": "#1f1f1f",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.family": "Helvetica",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.fontsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 5,
    }


def apply_matlab_style() -> None:
    """Apply the MATLAB-default rcParams to the current matplotlib session."""
    mpl.rcParams.update(_matlab_style())


def _matlab_axes(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = False,
    tick_in: bool = True,
) -> None:
    """Apply MATLAB-style axis decoration in one call.

    Sets axis labels and title using the MATLAB default font (Helvetica,
    size 10), routes tick marks inward (MATLAB default), turns the grid off
    unless explicitly requested, and forces spine line widths to 0.8 to
    match MATLAB's default axis line weight.
    """
    label_kwargs = {"fontfamily": "Helvetica", "fontsize": 10}
    if xlabel is not None:
        ax.set_xlabel(xlabel, **label_kwargs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **label_kwargs)
    if title is not None:
        ax.set_title(title, **label_kwargs)
    if tick_in:
        ax.tick_params(direction="in")
    ax.grid(bool(grid))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# MATLAB-parity subplot layouts.
#
# Each entry describes how a MATLAB plotting method tiles its figure: the
# overall (rows, cols) gridspec, the per-row/col size ratios, and a list of
# (panel_name, row_start, col_start, rowspan, colspan) tuples in 0-indexed
# gridspec coordinates. The names are the logical roles the caller plots
# into, decoupled from MATLAB subplot indices.
_MATLAB_SUBPLOT_LAYOUTS: dict[str, dict] = {
    "plotResults": {
        "grid": (2, 4),
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1, 1],
        "panels": [
            ("KSPlot", 0, 0, 1, 2),
            ("plotInvGausTrans", 0, 2, 1, 1),
            ("plotSeqCorr", 0, 3, 1, 1),
            ("plotCoeffs", 1, 0, 1, 2),
            ("plotResidual", 1, 2, 1, 2),
        ],
    },
    "plotSummary": {
        "grid": (2, 4),
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1, 1],
        "panels": [
            ("plotAllCoeffs", 0, 0, 2, 2),
            ("KSStatsBoxplot", 0, 2, 1, 2),
            ("getDiffAIC", 1, 2, 1, 1),
            ("getDiffBIC", 1, 3, 1, 1),
        ],
    },
    "plotIC": {
        "grid": (3, 1),
        "width_ratios": [1],
        "height_ratios": [1, 1, 1],
        "panels": [
            ("getDiffAIC", 0, 0, 1, 1),
            ("getDiffBIC", 1, 0, 1, 1),
            ("getDifflogLL", 2, 0, 1, 1),
        ],
    },
    "RunAnalysisForNeuron": {
        "grid": (2, 4),
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1, 1],
        "panels": [
            ("KSPlot", 0, 0, 1, 2),
            ("plotInvGausTrans", 0, 2, 1, 1),
            ("plotSeqCorr", 0, 3, 1, 1),
            ("plotCoeffs", 1, 0, 1, 2),
            ("plotFitResidual", 1, 2, 1, 2),
        ],
    },
    # Alias for the diagnostic panel produced by computeHistLag's caller
    # (RunAnalysisForNeuron in MATLAB nSTAT). Same layout.
    "computeHistLag": {
        "grid": (2, 4),
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1, 1],
        "panels": [
            ("KSPlot", 0, 0, 1, 2),
            ("plotInvGausTrans", 0, 2, 1, 1),
            ("plotSeqCorr", 0, 3, 1, 1),
            ("plotCoeffs", 1, 0, 1, 2),
            ("plotFitResidual", 1, 2, 1, 2),
        ],
    },
    "CovColl.plot.nCov2": {
        "grid": (2, 1),
        "width_ratios": [1],
        "height_ratios": [1, 1],
        "panels": [
            ("covariate_1", 0, 0, 1, 1),
            ("covariate_2", 1, 0, 1, 1),
        ],
    },
    "CovColl.plot.nCov3": {
        "grid": (3, 1),
        "width_ratios": [1],
        "height_ratios": [1, 1, 1],
        "panels": [
            ("covariate_1", 0, 0, 1, 1),
            ("covariate_2", 1, 0, 1, 1),
            ("covariate_3", 2, 0, 1, 1),
        ],
    },
    "CovColl.plot.nCov4": {
        "grid": (2, 2),
        "width_ratios": [1, 1],
        "height_ratios": [1, 1],
        "panels": [
            ("covariate_1", 0, 0, 1, 1),
            ("covariate_2", 0, 1, 1, 1),
            ("covariate_3", 1, 0, 1, 1),
            ("covariate_4", 1, 1, 1, 1),
        ],
    },
    "plotCoeffsWithoutHistory": {
        "grid": (1, 1),
        "width_ratios": [1],
        "height_ratios": [1],
        "panels": [
            ("coefficients_with_CI", 0, 0, 1, 1),
        ],
    },
    "plotHistCoeffs": {
        "grid": (1, 1),
        "width_ratios": [1],
        "height_ratios": [1],
        "panels": [
            ("coefficients_with_CI", 0, 0, 1, 1),
        ],
    },
    # Convenience aliases (no MATLAB analog, but mirror common call sites).
    "plotVariability": {
        "grid": (1, 1),
        "width_ratios": [1],
        "height_ratios": [1],
        "panels": [
            ("variability", 0, 0, 1, 1),
        ],
    },
}


def _matlab_subplot_layout(
    fig: Figure, *, kind: str
) -> dict[str, mpl.axes.Axes]:
    """Lay out a figure's panels to match MATLAB's convention for ``kind``.

    Builds a ``gridspec`` on ``fig`` whose rows/columns and panel spans mirror
    the corresponding MATLAB plotting method, then creates one Axes per
    logical panel and returns a name → Axes mapping.

    Parameters
    ----------
    fig
        The matplotlib Figure to subdivide. Existing axes are not cleared.
    kind
        Layout key. Supported values:

        - ``"plotResults"`` — FitResult.plotResults (2x4)
        - ``"plotSummary"`` — FitResSummary.plotSummary (2x4)
        - ``"plotIC"`` — FitResSummary.plotIC (3x1)
        - ``"RunAnalysisForNeuron"`` / ``"computeHistLag"`` —
          Analysis.RunAnalysisForNeuron diagnostic panel (2x4)
        - ``"CovColl.plot.nCov2"``, ``"CovColl.plot.nCov3"``,
          ``"CovColl.plot.nCov4"`` — CovColl.plot variants
        - ``"plotCoeffsWithoutHistory"`` / ``"plotHistCoeffs"`` (1x1)
        - ``"plotVariability"`` (1x1; Python-only convenience)

    Returns
    -------
    dict[str, matplotlib.axes.Axes]
        Logical panel name → Axes. Callers plot into each panel by name.

    Raises
    ------
    ValueError
        If ``kind`` is not a known layout.
    """
    try:
        spec = _MATLAB_SUBPLOT_LAYOUTS[kind]
    except KeyError as exc:
        valid = ", ".join(sorted(_MATLAB_SUBPLOT_LAYOUTS))
        raise ValueError(
            f"Unknown subplot layout kind: {kind!r}. Valid kinds: {valid}"
        ) from exc

    rows, cols = spec["grid"]
    gs = fig.add_gridspec(
        rows,
        cols,
        width_ratios=spec["width_ratios"],
        height_ratios=spec["height_ratios"],
    )
    panels: dict[str, mpl.axes.Axes] = {}
    for name, r0, c0, rspan, cspan in spec["panels"]:
        ax = fig.add_subplot(gs[r0 : r0 + rspan, c0 : c0 + cspan])
        panels[name] = ax
    return panels


@dataclass
class FigureTracker:
    """Track figure creation order and save deterministic notebook images."""

    topic: str
    output_root: Path
    expected_count: int
    count: int = 0
    _active_fig: plt.Figure | None = field(default=None, init=False, repr=False)
    _active_ax: plt.Axes | None = field(default=None, init=False, repr=False)
    _note_y: float = field(default=0.95, init=False, repr=False)

    def __post_init__(self) -> None:
        # Apply the MATLAB-default theme to the current matplotlib session so
        # every notebook that constructs a FigureTracker automatically picks up
        # the MATLAB color cycle, jet colormap, and font/tick conventions —
        # without each notebook having to call apply_matlab_style() itself.
        apply_matlab_style()
        topic_dir = self._topic_dir()
        for img_path in topic_dir.glob("fig_*.png"):
            img_path.unlink()
        manifest_path = topic_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        # Inside an IPython kernel (nbclient / jupyter), flush the active figure
        # at the end of each cell so the last figure of a cell renders inline
        # under that cell. Combined with the eager save in new_figure(), every
        # figure is embedded under its producing cell — so figures show in the
        # committed notebook and on GitHub, not just the gallery PNGs.
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None:
                ip.events.register("post_run_cell", self._on_post_run_cell)
        except Exception:
            pass

    def _on_post_run_cell(self, *_args, **_kwargs) -> None:
        self._save_active()

    def _topic_dir(self) -> Path:
        out = self.output_root / self.topic
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_active(self) -> None:
        if self._active_fig is None:
            return
        out = self._topic_dir() / f"fig_{self.count:03d}.png"
        self._active_fig.tight_layout()
        self._active_fig.savefig(out, dpi=180)
        self._display_inline(self._active_fig)
        plt.close(self._active_fig)
        self._active_fig = None
        self._active_ax = None
        self._note_y = 0.95

    def _display_inline(self, fig) -> None:
        """Embed the figure as a cell output so it renders in the committed
        notebook (and on GitHub), in addition to the saved gallery PNG.

        Renders to PNG bytes and displays an ``image/png`` payload explicitly,
        so it works under the Agg backend (without ``%matplotlib inline``).
        No-op outside an IPython kernel, so gallery generation / tests are
        unaffected.
        """
        try:
            from IPython import get_ipython
            from IPython.display import Image, display

            if get_ipython() is None:
                return
            import io

            # Lower DPI for the embedded copy (the gallery PNG stays 180) so
            # notebooks with many figures stay well under the repo file-size
            # guard while remaining crisp for on-screen / GitHub viewing.
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            display(Image(data=buf.getvalue(), format="png"))
        except Exception:
            pass

    def new_figure(self, matlab_line: str | None = None) -> plt.Figure:
        """Start a new figure while preserving deterministic numbering."""

        if self.count >= int(self.expected_count):
            return self._active_fig if self._active_fig is not None else Figure()

        self._save_active()
        self.count += 1
        fig = plt.figure(figsize=(8.0, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{self.topic} :: Figure {self.count:03d}")
        ax.set_axis_off()
        self._active_fig = fig
        self._active_ax = ax
        self._note_y = 0.95
        if matlab_line:
            self.annotate(matlab_line)
        return fig

    def annotate(self, matlab_line: str) -> None:
        """Record plotting notes on the active placeholder figure."""

        if self._active_fig is None or self._active_ax is None:
            if self.count >= int(self.expected_count):
                return
            self.new_figure(matlab_line=None)
        assert self._active_ax is not None
        self._active_ax.text(
            0.02,
            self._note_y,
            matlab_line[:160],
            transform=self._active_ax.transAxes,
            fontsize=8,
            va="top",
            family="monospace",
        )
        self._note_y -= 0.08
        if self._note_y < 0.08:
            self._note_y = 0.95

    def finalize(self) -> None:
        """Save the active figure and enforce the expected figure count."""

        self._save_active()
        while self.count < int(self.expected_count):
            self.count += 1
            out = self._topic_dir() / f"fig_{self.count:03d}.png"
            fig = plt.figure(figsize=(8.0, 4.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"{self.topic} :: Figure {self.count:03d}")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(out, dpi=180)
            plt.close(fig)
        if self.count != int(self.expected_count):
            raise AssertionError(
                f"{self.topic}: produced {self.count} figure(s), expected {self.expected_count}"
            )
        topic_dir = self._topic_dir()
        images = [path.name for path in sorted(topic_dir.glob("fig_*.png"))]
        (topic_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "topic": self.topic,
                    "expected_count": int(self.expected_count),
                    "produced_count": self.count,
                    "images": images,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# MATLAB-parity drawing helpers
#
# Small, focused helpers shared by the notebook gallery. Each helper
# centralizes a MATLAB-matching convention (thin raster ticks, open
# Start/Finish circles, jet/gouraud heatmaps, 0.046/0.04 colorbar magic
# numbers, equal-aspect trajectory axes, [0,1] probability ticks, neuron
# y-tick labels, 45-deg KS reference line with a 95% confidence band) so
# notebooks stop drifting on per-cell styling.
# ---------------------------------------------------------------------------


def matlab_raster(
    ax: mpl.axes.Axes,
    spike_times: Any,
    *,
    row: float = 0.0,
    half_height: float = 0.4,
    color: str = "k",
    linewidth: float = 0.3,
) -> None:
    """Draw a row (or rows) of raster ticks in the MATLAB style.

    Centralizes the per-cell convention `ax.vlines(spikes, row-h, row+h,
    color='k', linewidth=0.3)` that recurs across notebooks. ``spike_times``
    may be either a single 1-D array-like of spike times (single row) or
    an iterable of 1-D arrays — in the multi-row case each subsequent
    array is drawn at ``row + i`` (matching the row index convention used
    by :func:`matlab_neuron_yticklabels`).

    Parameters
    ----------
    ax
        Target axes.
    spike_times
        Either a single 1-D array of spike times, or an iterable of such
        arrays (one per neuron / trial).
    row
        Row index for the first (or only) raster row.
    half_height
        Half-height of each tick mark. MATLAB default ~0.4.
    color
        Tick color (MATLAB default ``'k'``).
    linewidth
        Line width for each tick. MATLAB-thin default ``0.3``.
    """
    arr = np.asarray(spike_times, dtype=object) if not isinstance(spike_times, np.ndarray) else spike_times
    # Detect "list of rows" by checking for a nested sequence.
    rows: list[np.ndarray]
    if isinstance(spike_times, (list, tuple)) and len(spike_times) > 0 and hasattr(spike_times[0], "__len__"):
        rows = [np.asarray(r, dtype=float) for r in spike_times]
    elif isinstance(arr, np.ndarray) and arr.dtype == object:
        rows = [np.asarray(r, dtype=float) for r in arr]
    else:
        rows = [np.asarray(spike_times, dtype=float).ravel()]
    for i, times in enumerate(rows):
        if times.size == 0:
            continue
        y = row + i
        ax.vlines(
            times,
            y - half_height,
            y + half_height,
            color=color,
            linewidth=linewidth,
        )


def matlab_start_finish_markers(
    ax: mpl.axes.Axes,
    x: Any,
    y: Any,
    *,
    size: float = 80.0,
    zorder: int = 3,
    start_color: str = "tab:blue",
    finish_color: str = "tab:red",
    linewidth: float = 1.5,
) -> None:
    """Mark the start and finish of a 2D trajectory with MATLAB-style open circles.

    Draws an open circle (white face, colored edge) at the first sample
    and a matching circle at the last sample, using the MATLAB-faithful
    convention that was flagged as "less distinct than MATLAB" in reviewer
    iterations 16-21.

    Parameters
    ----------
    ax
        Target axes.
    x, y
        Trajectory coordinates (1-D array-likes, same length).
    size
        Marker size (matplotlib ``s`` for ``scatter``).
    zorder
        Z-order; default ``3`` keeps markers above heatmaps and trajectory lines.
    start_color
        Edge color for the Start marker.
    finish_color
        Edge color for the Finish marker.
    linewidth
        Edge width.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.size == 0 or ys.size == 0:
        return
    ax.scatter(
        xs[0],
        ys[0],
        s=size,
        facecolor="white",
        edgecolor=start_color,
        linewidth=linewidth,
        zorder=zorder,
        label="Start",
    )
    ax.scatter(
        xs[-1],
        ys[-1],
        s=size,
        facecolor="white",
        edgecolor=finish_color,
        linewidth=linewidth,
        zorder=zorder,
        label="Finish",
    )


def matlab_heatmap(
    ax: mpl.axes.Axes,
    gx: Any,
    gy: Any,
    Z: Any,
    *,
    cmap: str = "jet",
    shading: str = "gouraud",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = False,
) -> QuadMesh:
    """Render a MATLAB ``pcolor`` + ``shading interp`` equivalent heatmap.

    Wraps ``ax.pcolormesh`` with jet/gouraud defaults to match MATLAB's
    2D place-field and posterior plots. When ``add_colorbar`` is true,
    a colorbar is attached using the standard MATLAB-matching aspect
    ratio (fraction=0.046, pad=0.04) via :func:`matlab_colorbar`.

    Parameters
    ----------
    ax
        Target axes.
    gx, gy
        Grid edges/centers for the X and Y axes (1-D or 2-D, as accepted
        by ``pcolormesh``).
    Z
        2-D array of values.
    cmap
        Colormap name. MATLAB default ``'jet'``.
    shading
        ``pcolormesh`` shading. MATLAB ``shading interp`` ≈ ``'gouraud'``.
    vmin, vmax
        Optional color limits.
    add_colorbar
        If true, attach a colorbar (via :func:`matlab_colorbar`).

    Returns
    -------
    matplotlib.collections.QuadMesh
        The QuadMesh, so callers can attach their own colorbar / pass it
        to a colorbar utility.
    """
    mesh = ax.pcolormesh(
        gx,
        gy,
        np.asarray(Z),
        cmap=cmap,
        shading=shading,
        vmin=vmin,
        vmax=vmax,
    )
    if add_colorbar:
        matlab_colorbar(mesh, ax)
    return mesh


def matlab_colorbar(
    mappable: Any,
    ax: mpl.axes.Axes,
    *,
    fraction: float = 0.046,
    pad: float = 0.04,
    shrink: float = 1.0,
    label: str | None = None,
) -> Colorbar:
    """Attach a colorbar using the MATLAB-matching aspect ratio.

    Centralizes the repeated ``fig.colorbar(im, ax=ax, fraction=0.046,
    pad=0.04)`` call that keeps a colorbar visually proportional to its
    parent axes. The ``shrink`` keyword provides the iter-17 fallback for
    the overlap fix reviewer flagged on 2D posterior panels.

    Parameters
    ----------
    mappable
        Any colorable artist (e.g. the QuadMesh returned by
        :func:`matlab_heatmap`).
    ax
        The parent axes the colorbar attaches to.
    fraction
        Colorbar width as fraction of parent axes. MATLAB-magic ``0.046``.
    pad
        Spacing between colorbar and parent. MATLAB-magic ``0.04``.
    shrink
        Optional shrink factor for the colorbar height (iter-17 overlap
        fallback).
    label
        Optional colorbar label.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar instance.
    """
    fig = ax.figure
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=fraction,
        pad=pad,
        shrink=shrink,
    )
    if label is not None:
        cbar.set_label(label)
    return cbar


def matlab_trajectory_axes(
    ax: mpl.axes.Axes,
    *,
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    adjustable: str = "box",
) -> None:
    """Apply equal-aspect MATLAB-style axes for a 2D trajectory plot.

    Standardizes the ``ax.set_aspect('equal', adjustable=...)`` + label
    pair that was flagged as inconsistent ("aspect ratio and Start/Finish
    marker size differ") on HippocampalPlaceCellExample figure 1.

    Parameters
    ----------
    ax
        Target axes.
    xlabel, ylabel
        Axis labels (defaults match the canonical place-cell convention).
    adjustable
        Argument forwarded to ``set_aspect``. MATLAB default ``'box'``.
    """
    ax.set_aspect("equal", adjustable=adjustable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def matlab_probability_axis(
    ax: mpl.axes.Axes,
    axis: str = "y",
    *,
    ticks: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    limits: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Set the X or Y axis to MATLAB's ``[0, 1]`` probability convention.

    Applies the ``[0, 1]`` limits and 0.2-step ticks used across KS, ROC,
    and CDF plots in the MATLAB toolbox.

    Parameters
    ----------
    ax
        Target axes.
    axis
        Which axis to format — ``'y'`` (default), ``'x'``, or ``'both'``.
    ticks
        Tick positions; MATLAB default is ``(0, 0.2, 0.4, 0.6, 0.8, 1.0)``.
    limits
        Axis limits; MATLAB default is ``(0.0, 1.0)``.
    """
    if axis not in {"x", "y", "both"}:
        raise ValueError(f"axis must be 'x', 'y', or 'both'; got {axis!r}")
    ticks_list = list(ticks)
    if axis in {"y", "both"}:
        ax.set_yticks(ticks_list)
        ax.set_ylim(*limits)
    if axis in {"x", "both"}:
        ax.set_xticks(ticks_list)
        ax.set_xlim(*limits)


def matlab_neuron_yticklabels(
    ax: mpl.axes.Axes,
    n_neurons: int,
    *,
    prefix: str = "Neuron",
    start: int = 1,
    fontsize: float = 8.0,
) -> None:
    """Set the Y axis tick labels to ``"Neuron 1"``, ``"Neuron 2"``, ... .

    Centralizes the naming scheme (``"Neuron1"`` vs ``"N1"`` vs
    ``"neuron 1"``) so ensemble/raster plots stay consistent. Y-tick
    positions are set to integer row indices that match the multi-row
    convention used by :func:`matlab_raster`.

    Parameters
    ----------
    ax
        Target axes.
    n_neurons
        Number of neurons / rows.
    prefix
        Label prefix (default ``"Neuron"``).
    start
        First neuron number (default ``1``; pass ``0`` for zero-indexed).
    fontsize
        Tick-label font size.
    """
    if n_neurons < 0:
        raise ValueError(f"n_neurons must be >= 0; got {n_neurons}")
    positions = list(range(n_neurons))
    labels = [f"{prefix} {start + i}" for i in range(n_neurons)]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=fontsize)


def matlab_ks_reference_line(
    ax: mpl.axes.Axes,
    *,
    n: int | None = None,
    alpha: float = 0.05,
    draw_band: bool = True,
    line_color: str = "k",
    band_color: str = "0.7",
) -> None:
    """Draw the 45-deg KS reference line and (optionally) a 95% confidence band.

    Adds the MATLAB-style ``y = x`` reference line plus the Kolmogorov
    confidence band ``±c/√n`` (with ``c = 1.36`` for the default
    ``alpha = 0.05``). The band is rendered as a light gray fill between
    ``y = x − c/√n`` and ``y = x + c/√n``, clipped to ``[0, 1]``.

    Parameters
    ----------
    ax
        Target axes.
    n
        Sample size (number of rescaled times). Required when
        ``draw_band`` is true.
    alpha
        Significance level. Supports ``0.10``, ``0.05`` (default),
        ``0.01`` via the standard Kolmogorov critical values; other
        values fall back to ``c = 1.36``.
    draw_band
        Whether to shade the confidence band.
    line_color
        Color of the 45-deg reference line.
    band_color
        Fill color (matplotlib color string) for the confidence band.
    """
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color=line_color, linewidth=1.0)
    if not draw_band:
        return
    if n is None or n <= 0:
        return
    # Kolmogorov critical values for common alpha levels.
    critical = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    c = critical.get(round(float(alpha), 2), 1.36)
    delta = c / float(np.sqrt(n))
    grid = np.linspace(0.0, 1.0, 200)
    lower = np.clip(grid - delta, 0.0, 1.0)
    upper = np.clip(grid + delta, 0.0, 1.0)
    ax.fill_between(grid, lower, upper, color=band_color, alpha=0.4, linewidth=0)
