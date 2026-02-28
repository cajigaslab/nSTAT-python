from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nstat.analysis import Analysis
from nstat.cif import CIFModel
from nstat.confidence import ConfidenceInterval
from nstat.decoding import DecodingAlgorithms
from nstat.events import Events
from nstat.fit import FitResult, FitSummary
from nstat.history import HistoryBasis
from nstat.signal import Covariate, Signal
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import ConfigCollection, CovariateCollection, Trial, TrialConfig


REQUIRED_MATLAB_CLASSES = {
    "SignalObj",
    "Covariate",
    "ConfidenceInterval",
    "Events",
    "History",
    "nspikeTrain",
    "nstColl",
    "CovColl",
    "TrialConfig",
    "ConfigColl",
    "Trial",
    "CIF",
    "Analysis",
    "FitResult",
    "FitResSummary",
    "DecodingAlgorithms",
}


def _load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_obj(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _extract_value(obj: Any, path: str) -> Any:
    cursor = obj
    for token in path.split("."):
        if token.isdigit():
            cursor = cursor[int(token)]
        else:
            cursor = getattr(cursor, token)
    return cursor


def _tol_value(key: str) -> float:
    payload = _load_yaml("tests/parity/tolerances.yml")
    value: Any = payload
    for token in key.split("."):
        value = value[token]
    return float(value)


def _build_signal_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 2.0, 5)
    obj = Signal(time=time, data=np.array([1.0, 2.0, 3.0, 2.0, 1.0]), name="sig")
    return obj, {}


def _build_covariate_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    data = np.column_stack([np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)])
    obj = Covariate(time=time, data=data, name="stim", labels=["stim1", "stim2"])
    return obj, {}


def _build_confidence_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    lower = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
    upper = lower + 0.5
    obj = ConfidenceInterval(time=time, lower=lower, upper=upper, level=0.95)
    return obj, {"contains_args": [np.array([0.1, 0.5, 0.9, 2.1, -0.1])]}


def _build_events_basic() -> tuple[Any, dict[str, Any]]:
    obj = Events(times=np.array([0.1, 0.4, 0.9]), labels=["a", "b", "c"])
    return obj, {}


def _build_history_basic() -> tuple[Any, dict[str, Any]]:
    obj = HistoryBasis(bin_edges_s=np.array([0.0, 0.05, 0.1, 0.2]))
    args = [np.array([0.12, 0.28]), np.array([0.15, 0.25, 0.30, 0.40])]
    return obj, {"design_args": args}


def _build_spike_train_basic() -> tuple[Any, dict[str, Any]]:
    obj = SpikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0)
    return obj, {}


def _build_spike_coll_basic() -> tuple[Any, dict[str, Any]]:
    st1 = SpikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0)
    st2 = SpikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0)
    obj = SpikeTrainCollection([st1, st2])
    return obj, {}


def _build_covcoll_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    cov1 = Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = Covariate(
        time=time,
        data=np.column_stack([time, time**2]),
        name="poly",
        labels=["t", "t2"],
    )
    obj = CovariateCollection([cov1, cov2])
    return obj, {}


def _build_trial_config_basic() -> tuple[Any, dict[str, Any]]:
    obj = TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg")
    return obj, {}


def _build_config_coll_basic() -> tuple[Any, dict[str, Any]]:
    cfg1 = TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="a")
    cfg2 = TrialConfig(covariate_labels=["stim"], sample_rate_hz=500.0, fit_type="binomial", name="b")
    obj = ConfigCollection([cfg1, cfg2])
    return obj, {}


def _build_trial_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 11)
    cov1 = Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    st = SpikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0)
    obj = Trial(spikes=SpikeTrainCollection([st]), covariates=CovariateCollection([cov1, cov2]))
    return obj, {}


def _build_cif_basic() -> tuple[Any, dict[str, Any]]:
    obj = CIFModel(coefficients=np.array([0.8]), intercept=-1.0, link="binomial")
    X = np.linspace(-1.0, 1.0, 5)[:, None]
    y = np.array([0, 0, 0, 1, 1], dtype=float)
    return obj, {"eval_args": [X], "ll_args": [y, X, 1.0]}


def _build_analysis_basic() -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=2000)
    X = x[:, None]
    true_beta = 1.15
    p = 1.0 / (1.0 + np.exp(-(-1.2 + true_beta * x)))
    y = rng.binomial(1, p).astype(float)
    kwargs = {"X": X, "y": y, "fit_type": "binomial", "dt": 1.0}
    return Analysis, {"fit_kwargs": kwargs, "true_beta": true_beta}


def _build_fit_result_basic() -> tuple[Any, dict[str, Any]]:
    obj = FitResult(
        coefficients=np.array([0.5]),
        intercept=-1.0,
        fit_type="binomial",
        log_likelihood=-10.0,
        n_samples=100,
        n_parameters=2,
    )
    X = np.array([[-1.0], [0.0], [1.0]])
    return obj, {"predict_args": [X]}


def _build_fit_summary_basic() -> tuple[Any, dict[str, Any]]:
    r1 = FitResult(
        coefficients=np.array([0.5]),
        intercept=-1.0,
        fit_type="binomial",
        log_likelihood=-8.0,
        n_samples=100,
        n_parameters=2,
    )
    r2 = FitResult(
        coefficients=np.array([0.1]),
        intercept=-0.5,
        fit_type="poisson",
        log_likelihood=-20.0,
        n_samples=100,
        n_parameters=2,
    )
    obj = FitSummary([r1, r2])
    return obj, {}


def _build_decoding_basic() -> tuple[Any, dict[str, Any]]:
    ci_matrix = np.array(
        [
            [0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1],
        ],
        dtype=float,
    )
    wc_counts = np.array(
        [
            [0, 0, 1, 2, 1, 0],
            [1, 2, 2, 1, 0, 0],
            [0, 0, 0, 1, 2, 2],
        ],
        dtype=float,
    )
    tuning = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.2, 1.1, 0.4],
            [0.1, 0.4, 1.2],
        ],
        dtype=float,
    )
    return DecodingAlgorithms, {"ci_args": [ci_matrix], "wc_args": [wc_counts, tuning]}


SCENARIO_BUILDERS = {
    "signal_basic": _build_signal_basic,
    "covariate_basic": _build_covariate_basic,
    "confidence_basic": _build_confidence_basic,
    "events_basic": _build_events_basic,
    "history_basic": _build_history_basic,
    "spike_train_basic": _build_spike_train_basic,
    "spike_coll_basic": _build_spike_coll_basic,
    "covcoll_basic": _build_covcoll_basic,
    "trial_config_basic": _build_trial_config_basic,
    "config_coll_basic": _build_config_coll_basic,
    "trial_basic": _build_trial_basic,
    "cif_basic": _build_cif_basic,
    "analysis_basic": _build_analysis_basic,
    "fit_result_basic": _build_fit_result_basic,
    "fit_summary_basic": _build_fit_summary_basic,
    "decoding_basic": _build_decoding_basic,
}


def _assert_expectation(result: Any, expect: dict[str, Any], context: dict[str, Any]) -> None:
    if "instance_of" in expect:
        klass = _resolve_obj(str(expect["instance_of"]))
        assert isinstance(result, klass)
    if "shape" in expect:
        expected_shape = tuple(int(x) for x in expect["shape"])
        assert np.asarray(result).shape == expected_shape
    if "length" in expect:
        assert len(result) == int(expect["length"])
    if "equals" in expect:
        expected = expect["equals"]
        if isinstance(expected, list):
            assert np.array_equal(np.asarray(result), np.asarray(expected))
        else:
            assert result == expected
    if "approx" in expect or "approx_key" in expect:
        if "approx" in expect:
            target = float(expect["approx"])
        else:
            target = float(context[str(expect["approx_key"])])
        if "abs_tol_key" in expect:
            abs_tol = _tol_value(str(expect["abs_tol_key"]))
        else:
            abs_tol = float(expect.get("abs_tol", 1.0e-8))
        assert np.isclose(float(result), target, atol=abs_tol)
    if "sum_approx" in expect:
        arr = np.asarray(result, dtype=float)
        abs_tol = float(expect.get("abs_tol", 1.0e-8))
        assert np.isclose(float(np.sum(arr)), float(expect["sum_approx"]), atol=abs_tol)
    if "min" in expect:
        assert float(np.min(np.asarray(result, dtype=float))) >= float(expect["min"])
    if "max" in expect:
        assert float(np.max(np.asarray(result, dtype=float))) <= float(expect["max"])
    if bool(expect.get("finite", False)):
        assert np.all(np.isfinite(np.asarray(result, dtype=float)))


def _execute_contract(obj: Any, context: dict[str, Any], contract: dict[str, Any]) -> Any:
    member = getattr(obj, str(contract["member"]))
    if contract.get("access", "method") == "property":
        result = member
    else:
        args = list(contract.get("args", []))
        kwargs = dict(contract.get("kwargs", {}))
        if "args_key" in contract:
            args = list(context[str(contract["args_key"])])
        if "kwargs_key" in contract:
            kwargs = dict(context[str(contract["kwargs_key"])])
        result = member(*args, **kwargs)

    if "select" in contract:
        result = result[int(contract["select"])]
    if "extract" in contract:
        result = _extract_value(result, str(contract["extract"]))
    return result


def test_class_behavior_specs_cover_all_mapped_classes() -> None:
    payload = _load_yaml("tests/parity/class_behavior_specs.yml")
    mapped = {row["matlab_class"] for row in payload["classes"]}
    assert mapped == REQUIRED_MATLAB_CLASSES


def test_class_behavior_contracts_execute() -> None:
    payload = _load_yaml("tests/parity/class_behavior_specs.yml")
    for row in payload["classes"]:
        scenario_name = str(row["scenario"])
        obj, context = SCENARIO_BUILDERS[scenario_name]()
        expected_class = _resolve_obj(str(row["python_class"]))
        if isinstance(obj, type):
            assert obj is expected_class
        else:
            assert isinstance(obj, expected_class)

        for contract in row["contracts"]:
            result = _execute_contract(obj, context, contract)
            _assert_expectation(result, dict(contract["expect"]), context)
