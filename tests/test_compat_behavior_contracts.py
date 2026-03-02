from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nstat.compat import matlab as M


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


def _assert_expectation(result: Any, expect: dict[str, Any]) -> None:
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
    if "approx" in expect:
        abs_tol = float(expect.get("abs_tol", 1.0e-8))
        assert np.isclose(float(result), float(expect["approx"]), atol=abs_tol)
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


def _build_compat_signal_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 2.0, 5)
    obj = M.SignalObj(time=time, data=np.array([1.0, 2.0, 3.0, 2.0, 1.0]), name="sig")
    other = M.SignalObj(time=time, data=np.array([0.5, 1.0, 1.5, 1.0, 0.5]), name="sig2")
    other_shifted = M.SignalObj(
        time=time + 0.5, data=np.array([0.5, 1.0, 1.5, 1.0, 0.5]), name="sig2_shifted"
    )
    return obj, {
        "signal_labels_args": [["ch1"]],
        "signal_label_lookup_args": [["ch1"]],
        "signal_index_arg": ["ch1"],
        "signal_times_args": [np.array([0.0, 1.0, 2.0])],
        "signal_time_window_args": [0.5, 1.5],
        "signal_other_args": [other],
        "signal_other_shifted_args": [other_shifted],
        "signal_mask_args": [[1]],
        "signal_names_args": [["ch1"]],
        "signal_cell2str_args": [["a", "b", "c"]],
    }


def _build_compat_covariate_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    obj = M.Covariate(
        time=time,
        data=np.column_stack([time, time**2]),
        name="stim",
        labels=["stim1", "stim2"],
    )
    return obj, {"cov_from_structure_args": [obj.toStructure()]}


def _build_compat_confidence_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    obj = M.ConfidenceInterval(
        time=time,
        lower=np.array([0.0, 0.4, 0.8, 1.2, 1.6]),
        upper=np.array([0.5, 0.9, 1.3, 1.7, 2.1]),
        level=0.95,
    )
    return obj, {
        "ci_set_color_args": ["r"],
        "ci_set_value_args": [1.0],
        "ci_from_structure_args": [obj.toStructure()],
    }


def _build_compat_events_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.Events(times=np.array([0.1, 0.4, 0.9]), labels=["a", "b", "c"])
    return obj, {"events_from_structure_args": [obj.toStructure()]}


def _build_compat_history_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.History(bin_edges_s=np.array([0.0, 0.05, 0.1, 0.2]))
    return obj, {
        "history_compute_args": [
            np.array([0.12, 0.28]),
            np.array([0.15, 0.25, 0.30, 0.40]),
        ],
        "history_set_window_args": [0.0, 0.3, 3],
        "history_from_structure_args": [obj.toStructure()],
    }


def _build_compat_spike_train_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0)
    return obj, {
        "spike_sigrep_args": [0.1, "count"],
        "spike_isbinary_args": [0.01],
        "spike_from_structure_args": [obj.toStructure()],
    }


def _build_compat_spike_coll_basic() -> tuple[Any, dict[str, Any]]:
    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    obj = M.nstColl([st1, st2])
    return obj, {
        "coll_name_args": ["u1"],
        "coll_ind_args": [1],
        "coll_binned_args": [0.1, "count"],
        "coll_from_structure_args": [obj.toStructure()],
    }


def _build_compat_covcoll_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    cov1 = M.Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    obj = M.CovColl([cov1, cov2])
    return obj, {
        "covcoll_names_args": [["sine", "poly"]],
        "covcoll_sel_args": [[1]],
        "covcoll_name_arg": ["poly"],
        "covcoll_names_lookup_args": [["sine", "poly"]],
        "covcoll_present_args": ["sine"],
        "covcoll_from_structure_args": [obj.toStructure()],
    }


def _build_compat_trial_config_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg")
    return obj, {
        "trial_cfg_set_name_args": ["cfg2"],
        "trial_cfg_from_structure_args": [obj.toStructure()],
    }


def _build_compat_config_coll_basic() -> tuple[Any, dict[str, Any]]:
    cfg = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg")
    obj = M.ConfigColl([cfg])
    return obj, {
        "config_get_args": [1],
        "config_subset_args": [[1]],
        "config_from_structure_args": [obj.toStructure()],
    }


def _build_compat_trial_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 11)
    cov1 = M.Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    cc = M.CovColl([cov1, cov2])
    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    obj = M.Trial(spikes=M.nstColl([st1, st2]), covariates=cc)
    return obj, {
        "trial_spike_vector_args": [0.1, 0, "count"],
        "trial_cov_args": [0],
        "trial_neuron_args": [0],
        "trial_all_labels_args": [0.1],
        "trial_aligned_args": [0.1, 0, "count"],
        "trial_from_structure_args": [obj.toStructure()],
    }


def _build_compat_cif_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.CIF(coefficients=np.array([0.8, -0.2]), intercept=-1.0, link="binomial")
    X = np.column_stack([np.linspace(-1.0, 1.0, 5), np.linspace(1.0, -1.0, 5)])
    spike = M.nspikeTrain(spike_times=np.array([0.1, 0.2]), t_start=0.0, t_end=1.0)
    history = M.History(bin_edges_s=np.array([0.0, 0.05, 0.1]))
    lam = M.Covariate(
        time=np.linspace(0.0, 1.0, 101),
        data=np.abs(np.sin(np.linspace(0.0, 1.0, 101) * 2.0 * np.pi)) + 0.1,
        name="lam",
        labels=["lam"],
    )
    return obj, {
        "cif_from_structure_args": [obj.toStructure()],
        "cif_eval_args": [X, 1.0],
        "cif_grad_args": [X],
        "cif_fn_args": [X],
        "cif_set_spike_args": [spike],
        "cif_set_history_args": [history],
        "cif_lambda_sim_args": [lam, 2],
    }


def _build_compat_analysis_basic() -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=400)
    X = x[:, None]
    p = 1.0 / (1.0 + np.exp(-(-1.2 + 1.15 * x)))
    y = rng.binomial(1, p).astype(float)
    fit = M.Analysis.GLMFit(X, y, "binomial", 1.0, 0.0)
    return M.Analysis, {
        "glm_args": [X, y, "binomial", 1.0, 0.0],
        "residual_args": [y, X, fit, 1.0],
        "inv_args": [y, X, fit, 1.0],
        "ks_args": [np.sort(np.random.default_rng(1).uniform(size=50))],
        "fdr_args": [np.array([0.001, 0.02, 0.04, 0.2]), 0.05],
        "hist_lag_args": [np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]), 2],
        "neighbors_args": [np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 1],
        "sta_args": [
            np.linspace(0.0, 1.0, 100),
            np.array([0.2, 0.6]),
            np.linspace(0.0, 1.0, 100),
            (-0.05, 0.05),
        ],
    }


def _build_compat_fit_result_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.FitResult(
        coefficients=np.array([0.5, -0.2]),
        intercept=-1.0,
        fit_type="binomial",
        log_likelihood=-10.0,
        n_samples=100,
        n_parameters=3,
        parameter_labels=["stim", "hist"],
    )
    X = np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]])
    return obj, {
        "fit_coeff_index_args": ["stim"],
        "fit_param_args": ["fit_type"],
        "fit_eval_args": [X],
        "fit_from_structure_args": [obj.toStructure()],
        "fit_cell_array_args": [[obj]],
        "fit_subset_args": [[1]],
    }


def _build_compat_fit_summary_basic() -> tuple[Any, dict[str, Any]]:
    f1 = M.FitResult(
        coefficients=np.array([0.5, -0.2]),
        intercept=-1.0,
        fit_type="binomial",
        log_likelihood=-10.0,
        n_samples=100,
        n_parameters=3,
        parameter_labels=["stim", "hist"],
    )
    f2 = M.FitResult(
        coefficients=np.array([0.1, 0.3]),
        intercept=-0.5,
        fit_type="poisson",
        log_likelihood=-20.0,
        n_samples=100,
        n_parameters=3,
        parameter_labels=["stim", "hist"],
    )
    obj = M.FitResSummary([f1, f2])
    return obj, {
        "summary_bin_coeffs_args": [-1.0, 1.0, 0.2],
        "summary_diff_metric_args": ["aic"],
        "summary_from_structure_args": [obj.toStructure()],
    }


def _build_compat_decoding_basic() -> tuple[Any, dict[str, Any]]:
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
    posterior = M.DecodingAlgorithms.decodeStatePosterior(wc_counts, tuning)[1]
    A = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=float)
    Q = np.eye(2, dtype=float) * 0.01
    H = np.array([[1.0, 0.0]], dtype=float)
    R = np.eye(1, dtype=float) * 0.05
    x_prev = np.array([0.0, 1.0], dtype=float)
    p_prev = np.eye(2, dtype=float)
    y = np.array([[0.1], [0.2], [0.4], [0.3]], dtype=float)
    xf, pf, xp, pp = M.DecodingAlgorithms.kalman_filter(y, A, H, Q, R, x_prev, p_prev)
    sig = M.DecodingAlgorithms.ukf_sigmas(np.array([0.0, 1.0]), np.eye(2), 0.0)
    wm = np.ones(sig.shape[1], dtype=float) / sig.shape[1]
    wc = wm.copy()
    return M.DecodingAlgorithms, {
        "diff_args": [ci_matrix, ci_matrix],
        "posterior_args": [wc_counts, tuning],
        "stim_ci_args": [posterior, np.arange(tuning.shape[1])],
        "ukf_sigmas_args": [np.array([0.0, 1.0]), np.eye(2), 0.0],
        "ukf_ut_args": [sig, wm, wc, np.eye(2, dtype=float) * 0.01],
        "kalman_predict_args": [x_prev, p_prev, A, Q],
        "kalman_update_args": [x_prev, p_prev, np.array([0.1], dtype=float), H, R],
        "kalman_filter_args": [y, A, H, Q, R, x_prev, p_prev],
        "kalman_smoother_args": [xf, pf, xp, pp, A],
        "kalman_from_filtered_args": [y, A, H, Q, R, x_prev, p_prev],
    }


SCENARIO_BUILDERS = {
    "compat_signal_basic": _build_compat_signal_basic,
    "compat_covariate_basic": _build_compat_covariate_basic,
    "compat_confidence_basic": _build_compat_confidence_basic,
    "compat_events_basic": _build_compat_events_basic,
    "compat_history_basic": _build_compat_history_basic,
    "compat_spike_train_basic": _build_compat_spike_train_basic,
    "compat_spike_coll_basic": _build_compat_spike_coll_basic,
    "compat_covcoll_basic": _build_compat_covcoll_basic,
    "compat_trial_config_basic": _build_compat_trial_config_basic,
    "compat_config_coll_basic": _build_compat_config_coll_basic,
    "compat_trial_basic": _build_compat_trial_basic,
    "compat_cif_basic": _build_compat_cif_basic,
    "compat_analysis_basic": _build_compat_analysis_basic,
    "compat_fit_result_basic": _build_compat_fit_result_basic,
    "compat_fit_summary_basic": _build_compat_fit_summary_basic,
    "compat_decoding_basic": _build_compat_decoding_basic,
}


def test_compat_behavior_specs_cover_all_mapped_classes() -> None:
    payload = _load_yaml("tests/parity/compat_behavior_specs.yml")
    mapped = {row["matlab_class"] for row in payload["classes"]}
    assert mapped == REQUIRED_MATLAB_CLASSES


def test_compat_behavior_contracts_execute() -> None:
    payload = _load_yaml("tests/parity/compat_behavior_specs.yml")
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
            _assert_expectation(result, dict(contract["expect"]))
