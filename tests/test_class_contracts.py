from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import yaml

from nstat.compat import matlab as M
from tests.parity_utils import assert_allclose_scaled, assert_same_shape


CLASS_CONTRACTS = Path("parity/class_contracts.yml")
FIXTURE_SPEC = Path("parity/class_fixture_export_spec.yml")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_mat(path: Path) -> dict[str, Any]:
    return scipy.io.loadmat(path)


def _vec(m: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _str_list(mat_cell: Any) -> list[str]:
    arr = np.asarray(mat_cell, dtype=object).reshape(-1)
    out: list[str] = []
    for item in arr:
        if isinstance(item, np.ndarray):
            if item.size == 1:
                out.append(str(item.reshape(-1)[0]))
            else:
                out.append(str(item.squeeze().tolist()))
        else:
            out.append(str(item))
    return out


def test_class_contract_manifest_covers_all_classes_and_fixtures_exist() -> None:
    contracts = _load_yaml(CLASS_CONTRACTS)
    fixture_spec = _load_yaml(FIXTURE_SPEC)

    contract_classes = {str(row["matlab_class"]) for row in contracts.get("classes", [])}
    fixture_classes = {str(row["matlab_class"]) for row in fixture_spec.get("classes", [])}
    expected = {
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

    assert contract_classes == expected
    assert fixture_classes == expected

    for row in contracts.get("classes", []):
        fixture_path = Path(str(row["fixture_path"]))
        assert fixture_path.exists(), f"missing fixture for {row['matlab_class']}: {fixture_path}"
        assert row.get("key_methods"), f"missing key_methods for {row['matlab_class']}"


def test_signalobj_contract_against_matlab_fixture() -> None:
    m = _load_mat(Path("tests/parity/fixtures/matlab_gold/SignalObjExamples_gold.mat"))
    t = _vec(m, "time_sig")
    v1 = _vec(m, "v1_sig")
    v2 = _vec(m, "v2_sig")
    fs = float(_vec(m, "sampleRate_sig")[0])
    rs = float(_vec(m, "resample_hz_sig")[0])
    window_t0 = float(_vec(m, "window_t0_sig")[0])
    window_t1 = float(_vec(m, "window_t1_sig")[0])
    expected_peak = int(round(float(_vec(m, "periodogram_peak_idx_sig")[0])))

    s = M.SignalObj(time=t, data=np.column_stack([v1, v2]), name="sig")
    assert s.getNumSamples() == int(round(float(_vec(m, "n_samples_sig")[0])))
    assert np.isclose(s.getSampleRate(), fs, atol=1e-10)

    rs_obj = s.resample(rs)
    assert rs_obj.getNumSamples() == int(round(float(_vec(m, "resampled_n_samples_sig")[0])))
    win = s.getSigInTimeWindow(window_t0, window_t1)
    assert win.getNumSamples() == int(round(float(_vec(m, "window_n_samples_sig")[0])))

    _f, p = s.periodogram()
    assert int(np.argmax(np.asarray(p).reshape(-1))) == expected_peak


def test_covariate_contract_against_matlab_fixture() -> None:
    m = _load_mat(Path("tests/parity/fixtures/matlab_gold/CovCollExamples_gold.mat"))
    time = _vec(m, "time_cov")
    stim = _vec(m, "cov_stim")
    ctx = np.asarray(m["cov_ctx"], dtype=float)
    expected_design = np.asarray(m["expected_design_cov"], dtype=float)

    cov = M.Covariate(
        time=time,
        data=np.column_stack([stim, ctx]),
        name="cov",
        labels=["stim", "ctx1", "ctx2"],
    )
    cov_mat = np.asarray(cov.getSigRep(), dtype=float)
    assert_same_shape(cov_mat, expected_design)
    assert_allclose_scaled(cov_mat, expected_design, rtol=0.0, atol=1e-12)

    sub = cov.getSubSignal(["stim"])
    assert_same_shape(np.asarray(sub.getSigRep(), dtype=float), stim.reshape(-1, 1))


def test_confidence_interval_contract_against_matlab_fixture() -> None:
    m = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/ConfidenceInterval/basic.mat"))
    time = _vec(m, "time")
    ci_data = np.asarray(m["ci_data"], dtype=float)
    expected_width = _vec(m, "width")
    expected_value = float(_vec(m, "ci_value")[0])

    ci = M.ConfidenceInterval(time=time, lower=ci_data[:, 0], upper=ci_data[:, 1], level=expected_value)
    ci.setColor("r")
    ci.setValue(expected_value)

    got_width = np.asarray(ci.getWidth(), dtype=float).reshape(-1)
    assert_same_shape(got_width, expected_width)
    assert_allclose_scaled(got_width, expected_width, rtol=0.0, atol=1e-12)
    assert np.isclose(ci.level, expected_value)


def test_events_history_nspiketrain_nstcoll_contracts_against_matlab_fixtures() -> None:
    m_events = _load_mat(Path("tests/parity/fixtures/matlab_gold/EventsExamples_gold.mat"))
    event_times = _vec(m_events, "event_times")
    subset_start = float(_vec(m_events, "subset_start")[0])
    subset_end = float(_vec(m_events, "subset_end")[0])
    expected_subset = _vec(m_events, "expected_subset_times")

    ev = M.Events(times=event_times)
    ev_subset = ev.subset(subset_start, subset_end)
    assert_allclose_scaled(np.asarray(ev_subset.times, dtype=float), expected_subset, rtol=0.0, atol=1e-12)

    m_hist = _load_mat(Path("tests/parity/fixtures/matlab_gold/HistoryExamples_gold.mat"))
    hist = M.History(_vec(m_hist, "bin_edges_hist"))
    h = np.asarray(
        hist.computeHistory(_vec(m_hist, "spike_times_hist"), _vec(m_hist, "time_grid_hist")),
        dtype=float,
    )
    expected_h = np.asarray(m_hist["H_expected_hist"], dtype=float)
    assert_same_shape(h, expected_h)
    assert_allclose_scaled(h, expected_h, rtol=0.0, atol=1e-12)

    m_coll = _load_mat(Path("tests/parity/fixtures/matlab_gold/nstCollExamples_gold.mat"))
    t_start = float(_vec(m_coll, "t_start_coll")[0])
    t_end = float(_vec(m_coll, "t_end_coll")[0])
    bin_size = float(_vec(m_coll, "bin_size_coll")[0])
    st1 = M.nspikeTrain(spike_times=_vec(m_coll, "spike_times_1"), t_start=t_start, t_end=t_end, name="u1")
    st2 = M.nspikeTrain(spike_times=_vec(m_coll, "spike_times_2"), t_start=t_start, t_end=t_end, name="u2")
    coll = M.nstColl([st1, st2])
    _t, count_mat = coll.getBinnedMatrix(bin_size, "count")
    expected_count = np.asarray(m_coll["expected_count_matrix"], dtype=float)
    assert_same_shape(count_mat, expected_count)
    assert_allclose_scaled(count_mat, expected_count, rtol=0.0, atol=1e-12)
    assert np.isclose(coll.getFirstSpikeTime(), float(_vec(m_coll, "expected_first_spike")[0]))
    assert np.isclose(coll.getLastSpikeTime(), float(_vec(m_coll, "expected_last_spike")[0]))


def test_covcoll_trial_trialconfig_configcoll_contracts_against_matlab_fixtures() -> None:
    m_cov = _load_mat(Path("tests/parity/fixtures/matlab_gold/CovCollExamples_gold.mat"))
    time = _vec(m_cov, "time_cov")
    stim = _vec(m_cov, "cov_stim")
    ctx = np.asarray(m_cov["cov_ctx"], dtype=float)

    cov1 = M.Covariate(time=time, data=stim, name="stim", labels=["stim"])
    cov2 = M.Covariate(time=time, data=ctx, name="ctx", labels=["ctx1", "ctx2"])
    cov_coll = M.CovColl([cov1, cov2])
    design, labels = cov_coll.getDesignMatrix()
    assert_same_shape(design, np.asarray(m_cov["expected_design_cov"], dtype=float))
    assert labels == ["stim", "ctx1", "ctx2"]

    m_trial = _load_mat(Path("tests/parity/fixtures/matlab_gold/TrialExamples_gold.mat"))
    st = M.nspikeTrain(
        spike_times=_vec(m_trial, "spike_times_trial"),
        t_start=0.0,
        t_end=1.0,
        name="u1",
    )
    trial = M.Trial(M.nstColl([st]), cov_coll)
    _t_obs, y_obs, _x_obs = trial.getAlignedBinnedObservation(
        binSize_s=float(_vec(m_trial, "bin_size_trial")[0]),
        unitIndex=0,
        mode="count",
    )
    y_expected = _vec(m_trial, "expected_y_trial")
    y_obs_vec = np.asarray(y_obs, dtype=float).reshape(-1)
    assert_same_shape(y_obs_vec, y_expected)
    assert_allclose_scaled(y_obs_vec, y_expected, rtol=0.0, atol=1e-12)

    m_tc = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/TrialConfig/basic.mat"))
    tc = M.TrialConfig(covariateLabels=["stim"], Fs=float(_vec(m_tc, "tc_sample_rate")[0]), name=str(_str_list(m_tc["tc_name"])[0]))
    assert np.isclose(tc.getSampleRate(), float(_vec(m_tc, "tc_sample_rate")[0]))

    m_cc = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/ConfigColl/basic.mat"))
    cc = M.ConfigColl([tc, M.TrialConfig(covariateLabels=["ctx"], Fs=500.0, name="cfg2")])
    assert len(cc.getConfigs()) == int(round(float(_vec(m_cc, "num_configs")[0])))


def test_cif_analysis_decoding_fitresult_fitsummary_contracts_against_matlab_fixtures() -> None:
    m_cif = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/CIF/basic.mat"))
    time = _vec(m_cif, "time")
    lam = _vec(m_cif, "lambda_values")
    dt = float(_vec(m_cif, "dt")[0])
    expected_counts = _vec(m_cif, "spike_counts")

    lambda_cov = M.Covariate(time=time, data=lam, name="Lambda", labels=["lambda"])
    np.random.seed(0)
    coll = M.CIF.simulateCIFByThinningFromLambda(lambda_cov, int(expected_counts.size), dt)
    counts = np.asarray([len(coll.getNST(i).spike_times) for i in range(int(expected_counts.size))], dtype=float)
    # Stochastic realizations are deterministic under seed=0; allow a tiny tolerance.
    assert_allclose_scaled(counts, expected_counts, rtol=0.0, atol=3.0)

    m_analysis = _load_mat(Path("tests/parity/fixtures/matlab_gold/AnalysisExamples_gold.mat"))
    fit = M.Analysis.fitGLM(
        X=np.asarray(m_analysis["X_analysis"], dtype=float),
        y=_vec(m_analysis, "counts_analysis"),
        fitType="poisson",
        dt=float(_vec(m_analysis, "dt_analysis")[0]),
    )
    expected_b = _vec(m_analysis, "b_analysis")
    got_b = np.concatenate(([fit.intercept], np.asarray(fit.coefficients, dtype=float).reshape(-1)))
    assert_allclose_scaled(got_b, expected_b, rtol=0.0, atol=0.5)

    m_dec = _load_mat(Path("tests/parity/fixtures/matlab_gold/DecodingExample_gold.mat"))
    decoded, posterior = M.DecodingAlgorithms.decodeStatePosterior(
        spike_counts=np.asarray(m_dec["spike_counts_dec"], dtype=float),
        tuning_rates=np.asarray(m_dec["tuning_dec"], dtype=float),
        transition=np.asarray(m_dec["transition_dec"], dtype=float),
    )
    assert np.array_equal(decoded, np.asarray(m_dec["expected_decoded_dec"], dtype=int).reshape(-1))
    assert_same_shape(posterior, np.asarray(m_dec["expected_posterior_dec"], dtype=float))

    m_fr = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/FitResult/basic.mat"))
    fr = M.FitResult(
        coefficients=np.array([0.1], dtype=float),
        intercept=0.0,
        fit_type="poisson",
        log_likelihood=float(_vec(m_fr, "fit_logll")[0]),
        n_samples=3,
        n_parameters=1,
        parameter_labels=["Baseline"],
    )
    fr.setKSStats(np.asarray(_vec(m_fr, "fit_ks_stat"), dtype=float))
    fr.setNeuronName(str(int(round(float(_vec(m_fr, "fit_neuron_number")[0])))))
    assert np.isfinite(fr.getAIC())
    assert np.isfinite(fr.getBIC())

    m_fs = _load_mat(Path("tests/parity/fixtures/matlab_gold/classes/FitResSummary/basic.mat"))
    fs = M.FitResSummary([fr])
    assert len(fs.results) == int(round(float(_vec(m_fs, "summary_num_neurons")[0])))
    assert fs.bestByAIC() is fr
