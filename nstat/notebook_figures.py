"""Deterministic figure tracking for notebook examples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# MATLAB R2014b+ "lines" color order
_MATLAB_LINES = [
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # light blue
    "#A2142F",  # dark red
]


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
