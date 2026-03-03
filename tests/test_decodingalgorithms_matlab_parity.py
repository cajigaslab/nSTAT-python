from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import DecodingAlgorithms as MatlabDecodingAlgorithms


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "DecodingAlgorithms" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def test_compute_spike_rate_cis_full_signature_matches_matlab_fixture() -> None:
    m = _mat()

    xK = np.asarray(m["xK"], dtype=float)
    Wku = np.asarray(m["Wku"], dtype=float)
    dN = np.asarray(m["dN"], dtype=float)
    t0 = _scalar(m, "t0")
    tf = _scalar(m, "tf")
    fit_type = str(np.asarray(m["fitType"], dtype=object).reshape(-1)[0])
    delta = _scalar(m, "delta")
    Mc = int(np.asarray(m["Mc"], dtype=float).reshape(-1)[0])
    alpha = _scalar(m, "alphaVal")

    spike_rate_sig, prob_mat, sig_mat = MatlabDecodingAlgorithms.computeSpikeRateCIs(
        xK,
        Wku,
        dN,
        t0,
        tf,
        fit_type,
        delta,
        np.array([], dtype=float),
        np.array([], dtype=float),
        Mc,
        alpha,
    )

    spike_rate_data = np.asarray(spike_rate_sig.dataToMatrix(), dtype=float).reshape(-1)

    np.testing.assert_allclose(spike_rate_data, np.asarray(m["spike_rate_data"], dtype=float).reshape(-1), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(prob_mat, dtype=float), np.asarray(m["ProbMat"], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sig_mat, dtype=float), np.asarray(m["sigMat"], dtype=float), rtol=0.0, atol=1e-12)
