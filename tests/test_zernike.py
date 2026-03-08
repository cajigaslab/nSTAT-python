from __future__ import annotations

import numpy as np

from nstat.zernike import zernfun, zernike_basis_from_cartesian


def test_normalized_zernfun_matches_closed_form_first_ten_modes() -> None:
    x = np.array([0.3, -0.5], dtype=float)
    y = np.array([0.4, 0.25], dtype=float)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    basis = zernike_basis_from_cartesian(x, y)
    expected = np.column_stack(
        [
            np.ones_like(r),
            2.0 * r * np.cos(theta),
            2.0 * r * np.sin(theta),
            np.sqrt(6.0) * (r**2) * np.cos(2.0 * theta),
            np.sqrt(3.0) * (2.0 * r**2 - 1.0),
            np.sqrt(6.0) * (r**2) * np.sin(2.0 * theta),
            np.sqrt(8.0) * (r**3) * np.cos(3.0 * theta),
            np.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.cos(theta),
            np.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.sin(theta),
            np.sqrt(8.0) * (r**3) * np.sin(3.0 * theta),
        ]
    )

    np.testing.assert_allclose(basis, expected, rtol=1e-12, atol=1e-12)


def test_zernfun_matches_matlab_sign_convention_for_positive_and_negative_m() -> None:
    r = np.array([0.2, 0.7], dtype=float)
    theta = np.array([0.1, 1.2], dtype=float)

    values = zernfun(np.array([1, 1], dtype=int), np.array([-1, 1], dtype=int), r, theta, normalized=True)

    np.testing.assert_allclose(values[:, 0], 2.0 * r * np.cos(theta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(values[:, 1], 2.0 * r * np.sin(theta), rtol=1e-12, atol=1e-12)


def test_zernike_basis_from_cartesian_can_mask_outside_unit_disk() -> None:
    x = np.array([0.0, 1.1], dtype=float)
    y = np.array([0.0, 0.0], dtype=float)

    basis = zernike_basis_from_cartesian(x, y, fill_value=np.nan)

    np.testing.assert_allclose(basis[0, 0], 1.0, rtol=1e-12, atol=1e-12)
    assert np.isnan(basis[1]).all()
