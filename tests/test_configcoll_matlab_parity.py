from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from nstat.compat.matlab import ConfigColl as MatlabConfigColl
from nstat.compat.matlab import TrialConfig as MatlabTrialConfig


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ConfigColl" / "basic.mat"


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
    return value


def _cellstr(value: Any) -> list[str]:
    arr = np.asarray(value, dtype=object).reshape(-1)
    out: list[str] = []
    for item in arr:
        parsed = _to_python(item)
        if isinstance(parsed, list):
            if not parsed:
                out.append("")
            else:
                out.append(str(parsed[0]))
        else:
            out.append(str(parsed))
    return out


def _scalar(value: Any) -> int:
    return int(np.asarray(value, dtype=float).reshape(-1)[0])


def _as_name(value: Any) -> str:
    names = _cellstr(value)
    if not names:
        return ""
    return names[0]


def _build_coll() -> MatlabConfigColl:
    tc1 = MatlabTrialConfig(["Force", "f_x"], 2000.0, [0.1, 0.2], -1.0, 2.0)
    tc2 = MatlabTrialConfig(["Position", "x"], 2000.0, [0.1, 0.2], -1.0, 2.0)
    return MatlabConfigColl([tc1, tc2])


def test_configcoll_core_behavior_matches_matlab_fixture() -> None:
    m = _mat()
    coll = _build_coll()

    assert coll.numConfigs == _scalar(m["initial_numConfigs"])
    assert coll.getConfigNames() == _cellstr(m["initial_getConfigNames"])
    assert coll.getConfig(2).name == _as_name(m["initial_config2_name"])

    coll.setConfigNames(["cfgA", "cfgB"])
    assert coll.getConfigNames() == _cellstr(m["names_after_set"])

    tc3 = MatlabTrialConfig(["Velocity", "v_x"], 1000.0, [0.05, 0.1], -1.0, 2.0, [], "cfgC")
    coll.addConfig(tc3)
    assert coll.getConfigNames() == _cellstr(m["names_after_add"])
    assert coll.numConfigs == _scalar(m["numConfigs_after_add"])

    subset = coll.getSubsetConfigs([1, 3])
    assert subset.getConfigNames() == _cellstr(m["subset_names"])


def test_configcoll_structure_roundtrip_matches_matlab_fixture() -> None:
    m = _mat()
    coll = _build_coll()
    coll.setConfigNames(["cfgA", "cfgB"])
    coll.addConfig(MatlabTrialConfig(["Velocity", "v_x"], 1000.0, [0.05, 0.1], -1.0, 2.0, [], "cfgC"))

    payload = coll.toStructure()
    assert int(payload["numConfigs"]) == _scalar(np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0].numConfigs)
    assert [str(v) for v in payload["configNames"]] == _cellstr(
        np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0].configNames
    )
    assert len(payload["configArray"]) == len(_to_python(np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0].configArray))

    struct_payload = _to_python(np.asarray(m["struct_payload"], dtype=object).reshape(-1)[0])
    restored = MatlabConfigColl.fromStructure(struct_payload)
    assert restored.numConfigs == _scalar(m["roundtrip_numConfigs"])
    assert restored.getConfigNames() == _cellstr(m["roundtrip_getConfigNames"])

    roundtrip_payload = restored.toStructure()
    expected_roundtrip = _to_python(np.asarray(m["roundtrip_struct"], dtype=object).reshape(-1)[0])
    assert int(roundtrip_payload["numConfigs"]) == int(expected_roundtrip["numConfigs"])
    assert [str(v) for v in roundtrip_payload["configNames"]] == [str(v) for v in expected_roundtrip["configNames"]]
    assert len(roundtrip_payload["configArray"]) == len(expected_roundtrip["configArray"])
