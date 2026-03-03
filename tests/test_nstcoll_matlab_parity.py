from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import nspikeTrain as MatlabSpikeTrain
from nstat.compat.matlab import nstColl as MatlabSpikeCollection


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nstColl" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _arr(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float)


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _cellstr(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        arr = np.asarray(value, dtype=object).reshape(-1)
        if arr.size == 1:
            out.append(str(arr[0]))
        else:
            out.append("".join(str(v) for v in arr))
    return out


def _build_coll() -> MatlabSpikeCollection:
    st1 = MatlabSpikeTrain(spike_times=np.array([0.10, 0.20, 0.25, 0.90]), t_start=0.0, t_end=1.0, name="u1")
    st2 = MatlabSpikeTrain(spike_times=np.array([0.15, 0.40, 0.80]), t_start=0.0, t_end=1.0, name="u2")
    st1.resample(10.0)
    st2.resample(10.0)
    return MatlabSpikeCollection([st1, st2])


def test_nstcoll_compat_core_matches_matlab_fixture() -> None:
    m = _mat()
    coll = _build_coll()

    assert np.isclose(coll.getFirstSpikeTime(), _scalar(m, "first_spike"), atol=1e-12)
    assert np.isclose(coll.getLastSpikeTime(), _scalar(m, "last_spike"), atol=1e-12)

    assert coll.getNSTnames() == _cellstr(np.asarray(m["names"], dtype=object))
    # MATLAB indices are 1-based.
    assert [idx + 1 for idx in coll.getNSTIndicesFromName("u2")] == _vec(m, "indices_u2").astype(int).tolist()
    assert coll.getNSTnameFromInd(1) == str(np.asarray(m["name_ind2"], dtype=object).reshape(-1)[0][0])

    np.testing.assert_allclose(coll.dataToMatrix(0.1, "count"), _arr(m, "data_mat"), rtol=0.0, atol=1e-12)
    assert coll.isSigRepBinary(0.1) == bool(_scalar(m, "is_binary"))
    assert coll.BinarySigRep(0.1) == bool(_scalar(m, "binary_sig"))

    np.testing.assert_allclose(coll.getMinISIs(), _vec(m, "min_isis"), rtol=0.0, atol=1e-12)
    assert np.isclose(coll.getMaxBinSizeBinary(), _scalar(m, "max_bin_size"), atol=1e-12)
    assert np.isclose(coll.findMaxSampleRate(), _scalar(m, "max_sample_rate"), atol=1e-12)

    t_psth, y_psth = coll.psth(0.1)
    np.testing.assert_allclose(t_psth, _vec(m, "psth_time"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(y_psth, _vec(m, "psth_data"), rtol=0.0, atol=1e-12)

    merged = coll.toSpikeTrain()
    np.testing.assert_allclose(merged.spike_times, _vec(m, "merged_spike_times"), rtol=0.0, atol=1e-12)

    basis = MatlabSpikeCollection.generateUnitImpulseBasis(0.2, 0.0, 1.0, 10.0)
    np.testing.assert_allclose(basis.time, _vec(m, "basis_time"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(basis.data_to_matrix(), _arr(m, "basis_data"), rtol=0.0, atol=1e-12)

    coll_mask = _build_coll()
    coll_mask.setNeuronMaskFromInd([1])
    assert [idx + 1 for idx in coll_mask.getIndFromMask()] == _vec(m, "mask_indices").astype(int).tolist()
    assert coll_mask.isNeuronMaskSet() == bool(_scalar(m, "mask_is_set"))

    coll_neigh = _build_coll()
    coll_neigh.setNeighbors(np.array([[1], [0]], dtype=int))
    assert coll_neigh.areNeighborsSet() == bool(_scalar(m, "are_neighbors_set"))


def test_nstcoll_compat_roundtrip_matches_matlab_fixture() -> None:
    m = _mat()
    mat_struct = np.asarray(m["coll_struct"], dtype=object).reshape(-1)[0]

    restored = MatlabSpikeCollection.fromStructure(mat_struct)
    np.testing.assert_allclose(restored.dataToMatrix(0.1, "count"), _arr(m, "roundtrip_data"), rtol=0.0, atol=1e-12)

    payload = restored.toStructure()
    reloaded = MatlabSpikeCollection.fromStructure(payload)
    np.testing.assert_allclose(reloaded.dataToMatrix(0.1, "count"), restored.dataToMatrix(0.1, "count"), rtol=0.0, atol=1e-12)


def test_generate_unit_impulse_basis_honors_sample_rate_and_defaults() -> None:
    explicit = MatlabSpikeCollection.generateUnitImpulseBasis(0.2, 0.0, 1.0, 10.0)
    defaulted = MatlabSpikeCollection.generateUnitImpulseBasis(0.2, 0.0, 1.0)

    assert explicit.time.shape == (11,)
    assert defaulted.time.shape == (1001,)
    np.testing.assert_allclose(np.diff(explicit.time), np.full(10, 0.1), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.diff(defaulted.time), np.full(1000, 0.001), rtol=0.0, atol=1e-12)
