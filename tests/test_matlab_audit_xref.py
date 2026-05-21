"""Regression tests for MATLAB-side audit findings (Phase 4.4).

Each test pins a Python-side bug discovered while cross-referencing the
MATLAB ``AUDIT_REPORT.md`` against the Python port (see
``parity/matlab_audit_xref.md``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
XREF_PATH = REPO_ROOT / "parity" / "matlab_audit_xref.md"


# ----------------------------------------------------------------------
# MATLAB §1.3: ExplambdaDeltaCubed uses ld^3, not ld^2
# ----------------------------------------------------------------------

def test_explambdadeltacubed_uses_cube_not_square() -> None:
    """The Python port mirrored MATLAB's old ``ld.^2`` bug at one site.

    MATLAB nSTAT v1.4.0 audit point §1.3 fixed
    ``DecodingAlgorithms.m`` line 5483/5537/8071/8125 from ``ld.^2`` to
    ``ld.^3`` — the variable name was always ``ExplambdaDeltaCubed`` but
    the code computed the square.  The Python port at
    ``nstat/decoding_algorithms.py:5092`` had the same bug with a
    comment ``# Matlab uses ld.^2 here``.

    This test confirms the corrected ``ld ** 3`` form is present.  It
    scans the source rather than executing a numerical example because
    the relevant code path is deep inside an EM iteration that's hard
    to exercise in a unit test; the source-level pin is sufficient to
    catch a regression to ``ld ** 2``.
    """
    source = (REPO_ROOT / "nstat" / "decoding_algorithms.py").read_text(
        encoding="utf-8"
    )
    # The fixed line should use ld ** 3, and the obsolete "Matlab uses
    # ld.^2 here" comment should be gone.
    assert "ExplambdaDeltaCubed = 1.0 / McExp * np.sum(ld ** 3)" in source, (
        "ExplambdaDeltaCubed should compute the cube (ld ** 3); MATLAB "
        "v1.4.0 audit §1.3 fixed this.  See parity/matlab_audit_xref.md."
    )
    assert "Matlab uses ld.^2 here" not in source, (
        "Obsolete comment 'Matlab uses ld.^2 here' should be removed; "
        "MATLAB v1.4.0 no longer uses ld.^2 at that site."
    )


# ----------------------------------------------------------------------
# Documentation contract
# ----------------------------------------------------------------------

def test_matlab_audit_xref_document_exists_and_is_current() -> None:
    """``parity/matlab_audit_xref.md`` must exist and reference v1.4.0.

    This document is the canonical record of which MATLAB-side bugs
    are/aren't in the Python port.  CI's drift gate doesn't cover it
    because it's hand-maintained — but its presence is required.
    """
    assert XREF_PATH.exists(), "parity/matlab_audit_xref.md is missing"
    text = XREF_PATH.read_text(encoding="utf-8")
    assert "MATLAB nSTAT v1.4.0" in text or "MATLAB v1.4.0" in text, (
        "matlab_audit_xref.md should reference the current MATLAB "
        "version (v1.4.0)."
    )
    # The document should at least mention the highest-severity finding.
    assert "ExplambdaDeltaCubed" in text, (
        "matlab_audit_xref.md should document the §1.3 ld^3 fix."
    )
