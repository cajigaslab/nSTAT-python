from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import nspikeTrain as MatlabSpikeTrain


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nspikeTrain" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _arr(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def test_nspiketrain_compat_core_matches_matlab_fixture() -> None:
    m = _mat()
    st = MatlabSpikeTrain(
        spike_times=_vec(m, "spikeTimes"),
        t_start=0.0,
        t_end=1.0,
        name="u1",
    )

    np.testing.assert_allclose(
        st.getSigRep(binSize_s=0.1, mode="count", minTime_s=0.0, maxTime_s=1.0),
        _vec(m, "sig_count"),
        rtol=0.0,
        atol=1e-12,
    )
    assert st.isSigRepBinary(0.1) == bool(_scalar(m, "is_binary"))

    np.testing.assert_allclose(st.getISIs(), _vec(m, "isis"), rtol=0.0, atol=1e-12)
    assert np.isclose(st.getMinISI(), _scalar(m, "min_isi"), atol=1e-12)
    assert np.isclose(st.getMaxBinSizeBinary(), _scalar(m, "max_bin_size"), atol=1e-12)
    assert np.isclose(st.computeRate(), _scalar(m, "firing_rate"), atol=1e-12)
    assert np.isclose(st.getLStatistic(), _scalar(m, "l_stat"), atol=1e-12)

    cp = st.nstCopy()
    np.testing.assert_allclose(cp.spike_times, _vec(m, "copy_spike_times"), rtol=0.0, atol=1e-12)

    bounds = st.nstCopy()
    bounds.setMinTime(0.05)
    bounds.setMaxTime(0.95)
    assert np.isclose(bounds.t_start, _scalar(m, "set_min_time"), atol=1e-12)
    assert np.isclose(float(bounds.t_end), _scalar(m, "set_max_time"), atol=1e-12)
    np.testing.assert_allclose(bounds.spike_times, _vec(m, "set_spike_times"), rtol=0.0, atol=1e-12)

    rs = st.nstCopy()
    rs.resample(_scalar(m, "resample_rate"))
    np.testing.assert_allclose(
        rs.getSigRep(binSize_s=0.1, mode="count", minTime_s=0.0, maxTime_s=1.0),
        _vec(m, "resample_sig"),
        rtol=0.0,
        atol=1e-12,
    )

    parts = st.partitionNST(np.array([0.0, 0.5, 1.0]))
    assert len(parts) == int(_scalar(m, "parts_num"))
    np.testing.assert_allclose(parts[0].spike_times, _vec(m, "part1_spikes"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(parts[1].spike_times, _vec(m, "part2_spikes"), rtol=0.0, atol=1e-12)


def test_nspiketrain_compat_roundtrip_matches_matlab_fixture() -> None:
    m = _mat()
    mat_struct = np.asarray(m["nst_struct"], dtype=object).reshape(-1)[0]

    restored = MatlabSpikeTrain.fromStructure(mat_struct)
    np.testing.assert_allclose(restored.spike_times, _vec(m, "roundtrip_spike_times"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        restored.getSigRep(binSize_s=0.1, mode="count", minTime_s=0.0, maxTime_s=1.0),
        _vec(m, "roundtrip_sig"),
        rtol=0.0,
        atol=1e-12,
    )

    payload = restored.toStructure()
    reloaded = MatlabSpikeTrain.fromStructure(payload)
    np.testing.assert_allclose(reloaded.spike_times, restored.spike_times, rtol=0.0, atol=1e-12)
