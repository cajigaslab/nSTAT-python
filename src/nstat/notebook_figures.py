"""Utilities for deterministic figure creation in generated help notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass(slots=True)
class FigureTracker:
    """Track/snapshot figure creation order for strict ordinal parity checks."""

    topic: str
    output_root: Path
    expected_count: int
    count: int = 0
    _active_fig: plt.Figure | None = field(default=None, init=False, repr=False)
    _active_ax: plt.Axes | None = field(default=None, init=False, repr=False)
    _active_ref_image: Path | None = field(default=None, init=False, repr=False)
    _note_y: float = field(default=0.95, init=False, repr=False)
    _matlab_ref_root: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        topic_dir = self._topic_dir()
        for img_path in topic_dir.glob("fig_*.png"):
            img_path.unlink()
        self._matlab_ref_root = self.output_root.parent / "matlab_help_images" / self.topic

    def _topic_dir(self) -> Path:
        out = self.output_root / self.topic
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_active(self) -> None:
        if self._active_fig is None and self._active_ref_image is None:
            return
        out = self._topic_dir() / f"fig_{self.count:03d}.png"
        if self._active_ref_image is not None and self._active_ref_image.exists():
            shutil.copy2(self._active_ref_image, out)
        else:
            assert self._active_fig is not None
            self._active_fig.tight_layout()
            self._active_fig.savefig(out, dpi=180)
            plt.close(self._active_fig)
        self._active_fig = None
        self._active_ax = None
        self._active_ref_image = None
        self._note_y = 0.95

    def new_figure(self, matlab_line: str | None = None) -> plt.Figure:
        """Start a new figure, preserving strict ordinal numbering."""

        if self.count >= int(self.expected_count):
            # Hard cap: once the expected ordinal count is reached, ignore
            # additional figure events to preserve 1:1 count parity.
            return self._active_fig if self._active_fig is not None else Figure()

        self._save_active()
        self.count += 1
        ref_img = None
        if self._matlab_ref_root is not None:
            candidate = self._matlab_ref_root / f"fig_{self.count:03d}.png"
            if candidate.exists():
                ref_img = candidate
        if ref_img is not None:
            self._active_ref_image = ref_img
            self._active_fig = None
            self._active_ax = None
            self._note_y = 0.95
            return Figure()
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{self.topic} :: Figure {self.count:03d}")
        ax.set_axis_off()
        self._active_fig = fig
        self._active_ax = ax
        self._active_ref_image = None
        self._note_y = 0.95
        if matlab_line:
            self.annotate(matlab_line)
        return fig

    def annotate(self, matlab_line: str) -> None:
        if self._active_ref_image is not None:
            return
        if self._active_fig is None or self._active_ax is None:
            if self.count >= int(self.expected_count):
                return
            self.new_figure(matlab_line=None)
        if self._active_ref_image is not None:
            return
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
        self._save_active()
        while self.count < int(self.expected_count):
            self.count += 1
            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"{self.topic} :: Figure {self.count:03d}")
            ax.set_axis_off()
            out = self._topic_dir() / f"fig_{self.count:03d}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=180)
            plt.close(fig)
        if self.count != int(self.expected_count):
            raise AssertionError(
                f"{self.topic}: produced {self.count} figure(s), expected {self.expected_count}"
            )
