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
        "signal_plot_props_args": [{"LineWidth": 2.0}],
        "signal_filter_args": [np.array([0.2, 0.2]), np.array([1.0, -0.3])],
        "signal_crosscorr_args": [other, 1],
    }


def _build_compat_covariate_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    obj = M.Covariate(
        time=time,
        data=np.column_stack([time, time**2]),
        name="stim",
        labels=["stim1", "stim2"],
    )
    ci = M.ConfidenceInterval(
        time=time,
        lower=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        upper=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
        level=0.95,
    )
    return obj, {
        "cov_from_structure_args": [obj.toStructure()],
        "cov_filter_args": [np.array([0.2, 0.2]), np.array([1.0, -0.3])],
        "cov_set_ci_args": [ci],
    }


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
        "ci_ctor_args": [obj.toStructure()],
    }


def _build_compat_events_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.Events(times=np.array([0.1, 0.4, 0.9]), labels=["a", "b", "c"])
    return obj, {
        "events_from_structure_args": [obj.toStructure()],
        "events_ctor_args": [obj.toStructure()],
        "events_dsxy_args": [0.2, 0.3],
    }


def _build_compat_history_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.History(bin_edges_s=np.array([0.0, 0.05, 0.1, 0.2]))
    spike = M.nspikeTrain(spike_times=np.array([0.12, 0.28]), t_start=0.0, t_end=1.0, name="u1")
    return obj, {
        "history_compute_args": [
            np.array([0.12, 0.28]),
            np.array([0.15, 0.25, 0.30, 0.40]),
        ],
        "history_set_window_args": [0.0, 0.3, 3],
        "history_from_structure_args": [obj.toStructure()],
        "history_ctor_args": [obj.toStructure()],
        "history_nst_window_args": [spike, np.array([0.15, 0.25, 0.30, 0.40])],
    }


def _build_compat_spike_train_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0)
    return obj, {
        "spike_sigrep_args": [0.1, "count"],
        "spike_isbinary_args": [0.01],
        "spike_from_structure_args": [obj.toStructure()],
        "spike_ctor_args": [obj.toStructure()],
        "spike_set_sigrep_args": [np.array([0.0, 1.0, 0.0, 1.0])],
        "spike_get_field_name_args": ["name"],
    }


def _build_compat_spike_coll_basic() -> tuple[Any, dict[str, Any]]:
    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    obj = M.nstColl([st1, st2])
    st_merge = M.nspikeTrain(spike_times=np.array([0.05, 0.35]), t_start=0.0, t_end=1.0, name="u3m")
    st_add = M.nspikeTrain(spike_times=np.array([0.12, 0.22]), t_start=0.0, t_end=1.0, name="u3")
    merge_coll = M.nstColl([st_merge])
    ens_time = np.linspace(0.0, 1.0, 11)
    ens_cov = M.CovColl(
        [
            M.Covariate(time=ens_time, data=np.zeros_like(ens_time), name="c1", labels=["c1"]),
            M.Covariate(time=ens_time, data=np.ones_like(ens_time), name="c2", labels=["c2"]),
        ]
    )
    return obj, {
        "coll_name_args": ["u1"],
        "coll_ind0_args": [0],
        "coll_ind_args": [1],
        "coll_binned_args": [0.1, "count"],
        "coll_merge_args": [merge_coll],
        "coll_field_arg": ["name"],
        "coll_shift_args": [0.1],
        "coll_setmask_args": [["u1"]],
        "coll_setneuronmask_ind_args": [[1]],
        "coll_setneuronmask_args": [["u2"]],
        "coll_neighbors_args": [np.array([[1], [0]])],
        "coll_add_args": [st_add],
        "coll_addspike_args": [0, 0.95],
        "coll_addnames_args": [ens_cov],
        "coll_basis_args": [0.2, 10.0, 1.0, "basis"],
        "coll_from_structure_args": [obj.toStructure()],
    }


def _build_compat_covcoll_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 5)
    cov1 = M.Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    obj = M.CovColl([cov1, cov2])
    cov3 = M.Covariate(time=time, data=np.cos(2 * np.pi * time), name="cosine", labels=["cosine"])
    cov4 = M.Covariate(time=time, data=time**3, name="cube", labels=["cube"])
    cov5 = M.Covariate(time=time, data=time**4, name="quartic", labels=["quartic"])
    cov6 = M.Covariate(time=time, data=time**5, name="quintic", labels=["quintic"])
    cov_extra_coll = M.CovColl([cov6])
    return obj, {
        "covcoll_names_args": [["sine", "poly"]],
        "covcoll_sel_args": [[1]],
        "covcoll_name_arg": ["poly"],
        "covcoll_names_lookup_args": [["sine", "poly"]],
        "covcoll_present_args": ["sine"],
        "covcoll_contains_args": ["abc", "b"],
        "covcoll_parse_args": [[1]],
        "covcoll_from_selector_args": [["sine"]],
        "covcoll_remaining_args": [["sine"]],
        "covcoll_mask_args": [["sine"]],
        "covcoll_masks_selector_args": [["poly"]],
        "covcoll_flat_mask_args": [[[0], [1]]],
        "covcoll_selector_mask_args": [[0, 1]],
        "covcoll_add_single_args": [cov3],
        "covcoll_add_args": [cov4],
        "covcoll_add_cell_args": [[cov5]],
        "covcoll_add_collection_args": [cov_extra_coll],
        "covcoll_remove_indices_args": [[5]],
        "covcoll_from_structure_args": [obj.toStructure()],
    }


def _build_compat_trial_config_basic() -> tuple[Any, dict[str, Any]]:
    obj = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg")
    time = np.linspace(0.0, 1.0, 5)
    cov = M.Covariate(time=time, data=time, name="stim", labels=["stim"])
    spike = M.nspikeTrain(spike_times=np.array([0.2, 0.4]), t_start=0.0, t_end=1.0, name="u1")
    trial = M.Trial(spikes=M.nstColl([spike]), covariates=M.CovColl([cov]))
    return obj, {
        "trial_cfg_set_name_args": ["cfg2"],
        "trial_cfg_from_structure_args": [obj.toStructure()],
        "trial_cfg_set_config_args": [trial],
    }


def _build_compat_config_coll_basic() -> tuple[Any, dict[str, Any]]:
    cfg = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg")
    obj = M.ConfigColl([cfg])
    cfg2 = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg2")
    cfg3 = M.TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="poisson", name="cfg3")
    return obj, {
        "config_get_args": [1],
        "config_subset_args": [[1]],
        "config_from_structure_args": [obj.toStructure()],
        "config_ctor_args": [obj.toStructure()],
        "config_add_args": [cfg2],
        "config_set_args": [2, cfg3],
        "config_set_names_args": [["cfgA", "cfgB"]],
    }


def _build_compat_trial_basic() -> tuple[Any, dict[str, Any]]:
    time = np.linspace(0.0, 1.0, 11)
    cov1 = M.Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time, data=np.column_stack([time, time**2]), name="poly", labels=["t", "t2"])
    cc = M.CovColl([cov1, cov2])
    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.25, 0.9]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    obj = M.Trial(spikes=M.nstColl([st1, st2]), covariates=cc)
    events = M.Events(times=np.array([0.2, 0.6]), labels=["e1", "e2"])
    history = M.History(bin_edges_s=np.array([0.0, 0.05, 0.1]))
    extra_time = time[(time >= 0.2) & (time <= 0.8)]
    cov_extra = M.Covariate(
        time=extra_time,
        data=np.cos(2 * np.pi * extra_time),
        name="extra",
        labels=["extra"],
    )
    return obj, {
        "trial_spike_vector_args": [0.1, 0, "count"],
        "trial_cov_args": [0],
        "trial_neuron_args": [0],
        "trial_all_labels_args": [0.1],
        "trial_aligned_args": [0.1, 0, "count"],
        "trial_events_args": [events],
        "trial_partition_args": [{"task": (0.2, 0.8)}],
        "trial_times_for_args": [0.2, 0.8],
        "trial_cov_mask_args": [["sine"]],
        "trial_ens_cov_mask_args": [[1]],
        "trial_neuron_mask_args": [["u1"]],
        "trial_neighbors_args": [np.array([[1], [0]])],
        "trial_history_args": [history],
        "trial_ens_hist_args": [np.array([1.0, 2.0])],
        "trial_add_cov_args": [cov_extra],
        "trial_remove_cov_args": ["extra"],
        "trial_hist_for_neurons_args": [[0], 0.1],
        "trial_flat_mask_args": [[[0], [1]]],
        "trial_shift_cov_args": [0.1],
        "trial_consistent_sr_args": [10.0],
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
    time = np.linspace(0.0, 1.0, 101)
    cov1 = M.Covariate(time=time, data=np.sin(2 * np.pi * time), name="sine", labels=["sine"])
    cov2 = M.Covariate(time=time, data=np.cos(2 * np.pi * time), name="cos", labels=["cos"])
    cc = M.CovColl([cov1, cov2])
    st1 = M.nspikeTrain(spike_times=np.array([0.1, 0.2, 0.45, 0.7]), t_start=0.0, t_end=1.0, name="u1")
    st2 = M.nspikeTrain(spike_times=np.array([0.15, 0.4, 0.8]), t_start=0.0, t_end=1.0, name="u2")
    trial = M.Trial(spikes=M.nstColl([st1, st2]), covariates=cc)
    config = M.TrialConfig(covariate_labels=["sine", "cos"], sample_rate_hz=100.0, fit_type="poisson", name="cfgA")
    y1 = (rng.random(200) < 0.25).astype(float)
    y2 = (rng.random(200) < 0.20).astype(float)
    Xh = rng.normal(size=(200, 2))
    spike_mat = rng.poisson(0.2, size=(3, 60)).astype(float)
    return M.Analysis, {
        "glm_args": [X, y, "binomial", 1.0, 0.0],
        "residual_args": [y, X, fit, 1.0],
        "inv_args": [y, X, fit, 1.0],
        "ks_args": [np.sort(np.random.default_rng(1).uniform(size=50))],
        "analysis_run_neuron_args": [trial, config, 0],
        "analysis_run_all_args": [trial, config],
        "analysis_plot_seq_args": [np.array([0.2, -0.1, 0.05, 0.0])],
        "analysis_comp_hist_all_args": [[y1, y2], [Xh, Xh], 1.0],
        "analysis_granger_args": [spike_mat, 1],
        "analysis_flat_mask_args": [[np.array([1, 0]), np.array([0, 1, 1])]],
        "analysis_ksdisc_args": [np.array([1, 2, 2, 3]), np.array([1, 1, 2, 2, 3, 4])],
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
        "fit_set_ks_args": [np.array([0.1, 0.2]), np.array([0.9, 0.8]), np.array([1.0, 1.0])],
        "fit_set_resid_args": [np.array([0.1, -0.2, 0.0])],
        "fit_set_inv_args": [{"z": np.array([0.1, 0.2])}],
        "fit_xtick_rotate_args": [np.array([0.0, 1.0]), 15.0],
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
        "summary_ctor_args": [[f1, f2]],
        "summary_set_coeff_range_args": [-1.0, 1.0],
        "summary_xtick_rotate_args": [np.array([0.0, 1.0]), 15.0],
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
