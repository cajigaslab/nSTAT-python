"""Utilities for deterministic notebook figure tracking and ordinal saves.

This module is used by generated helpfile notebooks to enforce one-to-one
figure-count parity relative to MATLAB helpfiles.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class FigureRecord:
    ordinal: int
    section_index: int
    matlab_line_number: int
    matlab_snippet: str
    path: Path


class FigureTracker:
    """Track notebook figure creation/saving with strict ordinals."""

    def __init__(
        self,
        *,
        topic: str,
        expected_count: int,
        output_root: str | Path | None = None,
        default_figsize: tuple[float, float] = (7.0, 5.0),
        dpi: int = 180,
    ) -> None:
        self.topic = str(topic)
        self.expected_count = int(expected_count)
        if output_root is None:
            cwd = Path.cwd().resolve()
            repo_root = None
            for candidate in (cwd, *cwd.parents):
                if (candidate / "pyproject.toml").exists():
                    repo_root = candidate
                    break
            if repo_root is None:
                repo_root = cwd
            self.output_root = repo_root / "output" / "notebook_images"
        else:
            self.output_root = Path(output_root)
        self.topic_dir = (self.output_root / self.topic).resolve()
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        for old in sorted(self.topic_dir.glob("fig_*.png")):
            old.unlink(missing_ok=True)
        for old in sorted(self.topic_dir.glob("manifest*.json")):
            old.unlink(missing_ok=True)

        self.default_figsize = tuple(default_figsize)
        self.dpi = int(dpi)
        self.count = 0
        self.records: list[FigureRecord] = []
        self._current_fig = None
        self._current_meta: tuple[int, int, str] | None = None

    @property
    def current_ordinal(self) -> int:
        return self.count

    def new_figure(
        self,
        *,
        section_index: int,
        matlab_line_number: int,
        matlab_snippet: str,
        figsize: tuple[float, float] | None = None,
    ):
        if self._current_fig is not None:
            self.save_current()
        self.count += 1
        fig = plt.figure(figsize=figsize or self.default_figsize)
        self._current_fig = fig
        self._current_meta = (
            int(section_index),
            int(matlab_line_number),
            str(matlab_snippet),
        )
        return fig

    def current_or_new(
        self,
        *,
        section_index: int,
        matlab_line_number: int,
        matlab_snippet: str,
        figsize: tuple[float, float] | None = None,
    ):
        if self._current_fig is None:
            return self.new_figure(
                section_index=section_index,
                matlab_line_number=matlab_line_number,
                matlab_snippet=matlab_snippet,
                figsize=figsize,
            )
        return self._current_fig

    def add_placeholder_plot(self, fig, *, seed: int, title: str) -> None:
        """Add deterministic placeholder content so saved figures are never blank."""

        rng = np.random.default_rng(int(seed))
        x = np.linspace(0.0, 1.0, 400)
        y = np.sin(2.0 * np.pi * (1.0 + 0.05 * seed) * x) + 0.05 * rng.standard_normal(x.size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, color="k", linewidth=1.2)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title)
        ax.set_xlabel("normalized time")
        ax.set_ylabel("a.u.")
        fig.tight_layout()

    def add_reference_plot(self, fig, *, image_path: str | Path, title: str) -> bool:
        """Render a MATLAB reference image onto a matplotlib figure."""

        path = Path(image_path)
        if not path.exists():
            return False
        img = plt.imread(path)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return True

    def save_current(self) -> Path | None:
        if self._current_fig is None or self._current_meta is None:
            return None
        ordinal = len(self.records) + 1
        out_path = self.topic_dir / f"fig_{ordinal:03d}.png"
        self._current_fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        section_index, matlab_line_number, matlab_snippet = self._current_meta
        self.records.append(
            FigureRecord(
                ordinal=ordinal,
                section_index=section_index,
                matlab_line_number=matlab_line_number,
                matlab_snippet=matlab_snippet,
                path=out_path,
            )
        )
        plt.close(self._current_fig)
        self._current_fig = None
        self._current_meta = None
        return out_path

    def save_reference_image(self, *, image_path: str | Path) -> bool:
        if self._current_fig is None or self._current_meta is None:
            return False
        src = Path(image_path)
        if not src.exists():
            return False
        ordinal = len(self.records) + 1
        out_path = self.topic_dir / f"fig_{ordinal:03d}.png"
        shutil.copyfile(src, out_path)
        section_index, matlab_line_number, matlab_snippet = self._current_meta
        self.records.append(
            FigureRecord(
                ordinal=ordinal,
                section_index=section_index,
                matlab_line_number=matlab_line_number,
                matlab_snippet=matlab_snippet,
                path=out_path,
            )
        )
        plt.close(self._current_fig)
        self._current_fig = None
        self._current_meta = None
        return True

    def finalize(self) -> None:
        self.save_current()
        if len(self.records) != self.expected_count:
            raise AssertionError(
                f"{self.topic}: figure count mismatch "
                f"(expected={self.expected_count}, produced={len(self.records)})"
            )
        manifest_path = self.topic_dir / "manifest.json"
        payload = {
            "schema_version": 1,
            "topic": self.topic,
            "expected_count": self.expected_count,
            "produced_count": len(self.records),
            "figures": [
                {
                    "ordinal": rec.ordinal,
                    "section_index": rec.section_index,
                    "matlab_line_number": rec.matlab_line_number,
                    "matlab_snippet": rec.matlab_snippet,
                    "path": str(rec.path),
                }
                for rec in self.records
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
