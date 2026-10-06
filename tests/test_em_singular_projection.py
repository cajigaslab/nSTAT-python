"""EM SE routines: nearestSPD projection when the observed information is singular.

MATLAB issue nSTAT#136 / PR #137: ``nearestSPD`` does not return on an exactly
singular matrix (its shift ``-mineig*k^2 + eps(mineig)`` turns negative when
``min(eig)`` is a tiny positive rounding value), so projecting the full
pseudo-inverse of a singular observed information never returned.  Only the
identifiable block is projected now (``_em_project_covariance``).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from nstat.decoding_algorithms import _em_project_covariance, _matlab_nearest_spd

REPO = Path(__file__).resolve().parents[1]


def test_nonsingular_is_plain_nearest_spd():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((5, 5))
    inv = np.linalg.inv(B @ B.T + 0.1 * np.eye(5))
    expected = _matlab_nearest_spd(inv)
    np.testing.assert_array_equal(_em_project_covariance(inv, None, _matlab_nearest_spd), expected)
    np.testing.assert_array_equal(
        _em_project_covariance(inv, np.zeros(5, dtype=bool), _matlab_nearest_spd), expected)


def test_singular_projects_only_the_identifiable_block_and_returns():
    # Parameters 1 and 2 (zero-based) enter only through their sum: the
    # pseudo-inverse is singular there.  Projecting it whole never returns
    # (the pre-fix behaviour), so run in a subprocess with a timeout.
    code = textwrap.dedent(
        """
        import numpy as np
        from nstat.decoding_algorithms import _em_project_covariance, _matlab_nearest_spd
        P = np.linalg.pinv(np.array([[4.0, 0, 0], [0, 1, 1], [0, 1, 1]]))
        out = _em_project_covariance(P, np.array([False, True, True]), _matlab_nearest_spd)
        assert np.all(np.isfinite(out))
        assert abs(out[0, 0] - _matlab_nearest_spd(P[:1, :1])[0, 0]) <= 1e-15
        np.testing.assert_array_equal(out[1:, 1:], P[1:, 1:])
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          cwd=REPO, env={"PYTHONPATH": str(REPO), "PATH": ""})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
