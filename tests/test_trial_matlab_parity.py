from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import CovColl as MatlabCovColl
from nstat.compat.matlab import Covariate as MatlabCovariate
from nstat.compat.matlab import Trial as MatlabTrial
from nstat.compat.matlab import nspikeTrain as MatlabSpikeTrain
from nstat.compat.matlab import nstColl as MatlabSpikeTrainCollection


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "Trial" / "basic.mat"


def _mat() -> dict[str, object]:
    return loadmat(str(FIXTURE), squeeze_me=False, struct_as_record=False)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_to_python(v) for v in value.reshape(-1)]
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            scalar = value.reshape(-1)[0]
            return scalar.item() if hasattr(scalar, "item") else scalar
        return value.tolist()
    if hasattr(value, "_fieldnames"):
        return {name: _to_python(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _scalar(m: dict[str, object], key: str) -> float:
    return float(np.asarray(m[key], dtype=float).reshape(-1)[0])


def _vec(m: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _cellstr(values: Any) -> list[str]:
    arr = np.asarray(values, dtype=object).reshape(-1)
    out: list[str] = []
    for value in arr:
        parsed = _to_python(value)
        if isinstance(parsed, list):
            out.append("" if not parsed else str(parsed[0]))
        else:
            out.append(str(parsed))
    return out


def _build_trial(m: dict[str, object]) -> MatlabTrial:
    time = _vec(m, "time_cov")
    cov1 = MatlabCovariate(time=time, data=_vec(m, "cov_stim"), name="sine", labels=["sine"])
    cov2 = MatlabCovariate(time=time, data=_vec(m, "cov_ctx"), name="ctx", labels=["ctx"])
    covs = MatlabCovColl([cov1, cov2])

    st1 = MatlabSpikeTrain(spike_times=_vec(m, "spike_times_u1"), t_start=0.0, t_end=1.0, name="u1")
    st2 = MatlabSpikeTrain(spike_times=_vec(m, "spike_times_u2"), t_start=0.0, t_end=1.0, name="u2")
    spikes = MatlabSpikeTrainCollection([st1, st2])
    return MatlabTrial(spikes=spikes, covariates=covs)


def test_trial_core_matches_matlab_fixture() -> None:
    m = _mat()
    trial = _build_trial(m)

    assert np.isclose(float(trial.findMinTime()), _scalar(m, "initial_minTime"), atol=1e-12)
    assert np.isclose(float(trial.findMaxTime()), _scalar(m, "initial_maxTime"), atol=1e-12)
    assert np.isclose(float(trial.findMinSampleRate()), _scalar(m, "initial_sampleRate"), atol=1e-12)
    assert trial.getAllCovLabels() == _cellstr(m["initial_cov_labels"])
    assert trial.getNeuronNames() == _cellstr(m["initial_neuron_names"])

    X_design, labels = trial.getDesignMatrix()
    np.testing.assert_allclose(X_design, np.asarray(m["initial_design_matrix"], dtype=float), rtol=0.0, atol=1e-12)
    assert labels == trial.getAllCovLabels()

    bin_size = _scalar(m, "bin_size")
    t_bins, y_u1, X = trial.getAlignedBinnedObservation(bin_size, unitIndex=0, mode="count")
    np.testing.assert_allclose(t_bins, _vec(m, "expected_t_bins"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(y_u1, _vec(m, "expected_y_u1"), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(X, np.asarray(m["expected_X"], dtype=float), rtol=0.0, atol=1e-12)

    _, y_u2, _ = trial.getAlignedBinnedObservation(bin_size, unitIndex=1, mode="count")
    np.testing.assert_allclose(y_u2, _vec(m, "expected_y_u2"), rtol=0.0, atol=1e-12)


def test_trial_masks_roundtrip_and_restore_match_matlab_fixture() -> None:
    m = _mat()

    cov_mask_trial = _build_trial(m)
    cov_mask_trial.setCovMask(["sine"])
    assert cov_mask_trial.getCovLabelsFromMask() == _cellstr(m["cov_mask_labels"])
    cov_mask_trial.resetCovMask()
    assert cov_mask_trial.getCovLabelsFromMask() == _cellstr(m["cov_mask_reset_labels"])

    neuron_mask_trial = _build_trial(m)
    neuron_mask_trial.setNeuronMask([1])  # MATLAB fixture stores 1-based selection.
    assert [idx + 1 for idx in neuron_mask_trial.getNeuronIndFromMask()] == _vec(m, "neuron_mask_indices").astype(int).tolist()
    neuron_mask_trial.resetNeuronMask()
    assert [idx + 1 for idx in neuron_mask_trial.getNeuronIndFromMask()] == _vec(m, "neuron_mask_reset_indices").astype(int).tolist()

    payload = _to_python(np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0])
    restored = MatlabTrial.fromStructure(payload)
    assert np.isclose(float(restored.findMinTime()), _scalar(m, "roundtrip_minTime"), atol=1e-12)
    assert np.isclose(float(restored.findMaxTime()), _scalar(m, "roundtrip_maxTime"), atol=1e-12)
    assert np.isclose(float(restored.findMinSampleRate()), _scalar(m, "roundtrip_sampleRate"), atol=1e-12)
    assert restored.getAllCovLabels() == _cellstr(m["roundtrip_cov_labels"])
    assert restored.getNeuronNames() == _cellstr(m["roundtrip_neuron_names"])
    X_rt, _ = restored.getDesignMatrix()
    np.testing.assert_allclose(X_rt, np.asarray(m["roundtrip_design_matrix"], dtype=float), rtol=0.0, atol=1e-12)

    shifted = _build_trial(m)
    shifted.shiftCovariates(0.2)
    assert np.isclose(float(shifted.findMinTime()), _scalar(m, "shift_minTime"), atol=1e-12)
    assert np.isclose(float(shifted.findMaxTime()), _scalar(m, "shift_maxTime"), atol=1e-12)

    shifted.restoreToOriginal()
    assert np.isclose(float(shifted.findMinTime()), _scalar(m, "restore_minTime"), atol=1e-12)
    assert np.isclose(float(shifted.findMaxTime()), _scalar(m, "restore_maxTime"), atol=1e-12)
