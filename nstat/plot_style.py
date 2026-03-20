"""Lightweight Python analogue of the upstream MATLAB plot-style helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
from matplotlib.collections import PathCollection


_STYLE_FILE = Path(__file__).resolve().with_name(".plot_style")


def _validate_style(style: str) -> str:
    norm = str(style).strip().lower()
    if norm not in {"legacy", "modern"}:
        raise ValueError('Invalid plot style. Valid styles: "legacy", "modern".')
    return norm


def get_plot_style(default: str = "modern") -> str:
    """Return the persisted global plotting style."""

    fallback = _validate_style(default)
    if not _STYLE_FILE.exists():
        return fallback
    try:
        return _validate_style(_STYLE_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return fallback


def set_plot_style(style: str = "modern") -> str:
    """Persist the plotting style for future sessions."""

    norm = _validate_style(style)
    _STYLE_FILE.write_text(norm + "\n", encoding="utf-8")
    return norm


def apply_plot_style(target=None, *, style: str = ""):
    """Apply the current nSTAT plot style to a matplotlib figure or axes."""

    chosen = get_plot_style() if not style else _validate_style(style)
    if chosen == "legacy":
        return target

    if target is None:
        return target
    if isinstance(target, matplotlib.figure.Figure):
        figure = target
        axes_list = list(target.axes)
    elif isinstance(target, matplotlib.axes.Axes):
        figure = target.figure
        axes_list = [target]
    else:
        figure = getattr(target, "figure", None)
        axes = getattr(target, "axes", None)
        axes_list = [axes] if isinstance(axes, matplotlib.axes.Axes) else []
        if figure is None:
            return target

    if figure is not None:
        figure.set_facecolor("white")

    for ax in axes_list:
        # Match MATLAB nstat.applyPlotStyle: Helvetica 10pt, ticks out, layer top
        ax.tick_params(direction="out", top=True, right=True, length=6, width=1)
        ax.set_axisbelow(False)  # layer = 'top' equivalent
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        # Set font on existing labels and title
        for label in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if label.get_fontfamily() == ["sans-serif"] or not label.get_text():
                label.set_fontfamily("Helvetica")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(10)
            label.set_fontfamily("Helvetica")
        for line in ax.get_lines():
            if isinstance(line, matplotlib.lines.Line2D) and float(line.get_linewidth()) < 1.25:
                line.set_linewidth(1.25)
            if line.get_marker() == ".":
                line.set_markersize(max(float(line.get_markersize()), 9.0))
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                sizes = coll.get_sizes()
                if sizes.size:
                    coll.set_sizes(sizes.clip(min=30.0))
        # Fix per-axes legends
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            for text in leg.get_texts():
                text.set_fontsize(10)
    # Fix figure-level legends
    legend = None if figure is None else figure.legends
    for item in legend or []:
        item.set_frame_on(False)
        item.prop.set_size(10)
    return target
