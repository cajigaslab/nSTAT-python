from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import TrialConfig as MatlabTrialConfig


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "TrialConfig" / "basic.mat"


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
        return value.astype(float).tolist() if np.issubdtype(value.dtype, np.number) else value.tolist()
    if hasattr(value, "_fieldnames"):
        return {name: _to_python(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _from_mat_key(m: dict[str, object], key: str) -> Any:
    arr = np.asarray(m[key], dtype=object)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return _to_python(arr.reshape(-1)[0])
    return _to_python(arr)


def _as_name(value: Any) -> str:
    if value == []:
        return ""
    return str(value)


def test_trialconfig_constructor_and_structure_match_matlab_fixture() -> None:
    m = _mat()

    cov_mask = _from_mat_key(m, "covMask")
    sample_rate = float(np.asarray(m["sampleRate"], dtype=float).reshape(-1)[0])
    history = np.asarray(m["history"], dtype=float).reshape(-1)
    ens_cov_hist = np.asarray(m["ensCovHist"], dtype=float).reshape(-1)
    ens_cov_mask = np.asarray(m["ensCovMask"], dtype=float)
    cov_lag = float(np.asarray(m["covLag"], dtype=float).reshape(-1)[0])
    name = str(_from_mat_key(m, "name"))

    default_cfg = MatlabTrialConfig()
    assert default_cfg.covMask == _from_mat_key(m, "default_covMask")
    assert default_cfg.sampleRate == _from_mat_key(m, "default_sampleRate")
    assert default_cfg.history == _from_mat_key(m, "default_history")
    assert default_cfg.ensCovHist == _from_mat_key(m, "default_ensCovHist")
    assert default_cfg.ensCovMask == _from_mat_key(m, "default_ensCovMask")
    assert default_cfg.covLag == _from_mat_key(m, "default_covLag")
    assert default_cfg.name == _as_name(_from_mat_key(m, "default_name"))

    cfg = MatlabTrialConfig(cov_mask, sample_rate, history, ens_cov_hist, ens_cov_mask, cov_lag, name)
    assert cfg.covMask == _from_mat_key(m, "custom_covMask")
    assert float(cfg.sampleRate) == float(_from_mat_key(m, "custom_sampleRate"))
    np.testing.assert_allclose(np.asarray(cfg.history, dtype=float).reshape(-1), np.asarray(_from_mat_key(m, "custom_history"), dtype=float).reshape(-1), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(cfg.ensCovHist, dtype=float).reshape(-1),
        np.asarray(_from_mat_key(m, "custom_ensCovHist"), dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(cfg.ensCovMask, dtype=float).reshape(-1),
        np.asarray(_from_mat_key(m, "custom_ensCovMask"), dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(np.asarray(cfg.covLag, dtype=float).reshape(-1)[0]) == float(np.asarray(_from_mat_key(m, "custom_covLag"), dtype=float).reshape(-1)[0])
    assert cfg.getName() == str(_from_mat_key(m, "custom_getName"))

    cfg.setName("cfgRenamed")
    assert cfg.getName() == str(_from_mat_key(m, "custom_name_after_set"))

    payload = cfg.toStructure()
    expected_payload = _from_mat_key(m, "custom_struct")
    assert _to_python(payload["covMask"]) == _to_python(expected_payload["covMask"])
    assert float(payload["sampleRate"]) == float(np.asarray(expected_payload["sampleRate"], dtype=float).reshape(-1)[0])
    np.testing.assert_allclose(np.asarray(payload["history"], dtype=float).reshape(-1), np.asarray(expected_payload["history"], dtype=float).reshape(-1), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(payload["ensCovHist"], dtype=float).reshape(-1), np.asarray(expected_payload["ensCovHist"], dtype=float).reshape(-1), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(payload["ensCovMask"], dtype=float), np.asarray(expected_payload["ensCovMask"], dtype=float), rtol=0.0, atol=1e-12)
    assert float(np.asarray(payload["covLag"], dtype=float).reshape(-1)[0]) == float(np.asarray(expected_payload["covLag"], dtype=float).reshape(-1)[0])
    assert str(payload["name"]) == str(_to_python(expected_payload["name"]))


def test_trialconfig_from_structure_matches_matlab_fixture_roundtrip() -> None:
    m = _mat()
    payload = _from_mat_key(m, "custom_struct")
    restored = MatlabTrialConfig.fromStructure(payload)

    assert restored.covMask == _from_mat_key(m, "roundtrip_covMask")
    assert float(restored.sampleRate) == float(np.asarray(_from_mat_key(m, "roundtrip_sampleRate"), dtype=float).reshape(-1)[0])
    np.testing.assert_allclose(
        np.asarray(restored.history, dtype=float).reshape(-1),
        np.asarray(_from_mat_key(m, "roundtrip_history"), dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(restored.ensCovHist, dtype=float).reshape(-1),
        np.asarray(_from_mat_key(m, "roundtrip_ensCovHist"), dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(np.asarray(restored.ensCovMask, dtype=float).reshape(-1)[0]) == float(
        np.asarray(_from_mat_key(m, "roundtrip_ensCovMask"), dtype=float).reshape(-1)[0]
    )
    assert str(restored.covLag) == str(_from_mat_key(m, "roundtrip_covLag"))
    assert restored.name == _as_name(_from_mat_key(m, "roundtrip_name"))

    roundtrip_payload = restored.toStructure()
    expected_roundtrip_payload = _from_mat_key(m, "roundtrip_struct")
    assert _to_python(roundtrip_payload["covMask"]) == _to_python(expected_roundtrip_payload["covMask"])
    assert float(roundtrip_payload["sampleRate"]) == float(
        np.asarray(expected_roundtrip_payload["sampleRate"], dtype=float).reshape(-1)[0]
    )
    np.testing.assert_allclose(
        np.asarray(roundtrip_payload["history"], dtype=float).reshape(-1),
        np.asarray(expected_roundtrip_payload["history"], dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(roundtrip_payload["ensCovHist"], dtype=float).reshape(-1),
        np.asarray(expected_roundtrip_payload["ensCovHist"], dtype=float).reshape(-1),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(np.asarray(roundtrip_payload["ensCovMask"], dtype=float).reshape(-1)[0]) == float(
        np.asarray(expected_roundtrip_payload["ensCovMask"], dtype=float).reshape(-1)[0]
    )
    assert str(roundtrip_payload["covLag"]) == str(_to_python(expected_roundtrip_payload["covLag"]))
    assert str(roundtrip_payload["name"]) == _as_name(_to_python(expected_roundtrip_payload["name"]))
