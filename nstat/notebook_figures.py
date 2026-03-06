"""Deterministic figure tracking for notebook examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
        topic_dir = self._topic_dir()
        for img_path in topic_dir.glob("fig_*.png"):
            img_path.unlink()

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
        plt.close(self._active_fig)
        self._active_fig = None
        self._active_ax = None
        self._note_y = 0.95

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
