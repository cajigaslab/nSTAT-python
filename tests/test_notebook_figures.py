"""Unit tests for helpers in :mod:`nstat.notebook_figures`."""

from __future__ import annotations

import pytest

from nstat.notebook_figures import MATLAB_LINES, matlab_palette


def test_matlab_palette_default_returns_seven_colors():
    assert len(matlab_palette()) == 7
    assert matlab_palette() == MATLAB_LINES


def test_matlab_palette_n_2_matches_matlab_lines2():
    assert matlab_palette(2) == ["#0072BD", "#D95319"]


def test_matlab_palette_n_1_returns_blue():
    assert matlab_palette(1) == ["#0072BD"]


def test_matlab_palette_invalid_n_raises():
    with pytest.raises(ValueError):
        matlab_palette(0)
    with pytest.raises(ValueError):
        matlab_palette(8)


def test_matlab_lines_constant_has_seven_unique_hex_colors():
    assert len(MATLAB_LINES) == 7
    assert all(c.startswith("#") and len(c) == 7 for c in MATLAB_LINES)
    assert len(set(MATLAB_LINES)) == 7  # all unique


def test_matlab_palette_returns_copy_not_module_alias():
    """Mutating the returned list must not corrupt the module-level constant."""
    pal = matlab_palette(3)
    pal[0] = "#000000"
    assert MATLAB_LINES[0] == "#0072BD"
