from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms
from nstat.events import Events
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial


MANIFEST = Path("tests/parity/fixtures/matlab_gold/manifest.yml")
NOTEBOOK_MANIFEST = Path("tools/notebooks/notebook_manifest.yml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _mat(path: str) -> dict:
    return scipy.io.loadmat(path)


def _vec(m: dict, key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=float).reshape(-1)


def _scalar(m: dict, key: str) -> float:
    return float(_vec(m, key)[0])


def test_matlab_gold_manifest_and_checksums() -> None:
    payload = _load_manifest()
    assert payload["version"] == 1
    assert len(payload["fixtures"]) >= 30

    for row in payload["fixtures"]:
        path = Path(row["path"])
        assert path.exists(), f"missing fixture {path}"
        assert _sha256(path) == row["sha256"], f"checksum mismatch for {path}"


def test_matlab_gold_manifest_covers_all_notebook_topics() -> None:
    payload = _load_manifest()
    fixture_topics = {str(row["name"]) for row in payload["fixtures"]}
    notebook_payload = yaml.safe_load(NOTEBOOK_MANIFEST.read_text(encoding="utf-8")) or {}
    notebook_topics = {
        str(row.get("topic", "")).strip()
        for row in notebook_payload.get("notebooks", [])
        if str(row.get("topic", "")).strip()
    }
    assert notebook_topics.issubset(fixture_topics), (
        "Missing fixture coverage for topics: "
        + ", ".join(sorted(notebook_topics - fixture_topics))
    )


def test_ppsimexample_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/PPSimExample_gold.mat")
    X = np.asarray(m["X"], dtype=float)
    y = _vec(m, "y")
    dt = _scalar(m, "dt")
    b = _vec(m, "b")

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)

    assert np.isclose(fit.intercept, b[0], atol=0.25)
    assert np.isclose(fit.coefficients[0], b[1], atol=0.25)

    expected_rate = _vec(m, "expected_rate")
    pred_rate = np.asarray(fit.predict(X), dtype=float).reshape(-1)
    rel_err = np.mean(np.abs(pred_rate - expected_rate) / np.maximum(expected_rate, 1e-12))
    assert rel_err <= 0.25


def test_decoding_with_hist_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/DecodingExampleWithHist_gold.mat")
    spike_counts = np.asarray(m["spike_counts"], dtype=float)
    tuning = np.asarray(m["tuning"], dtype=float)
    transition = np.asarray(m["transition"], dtype=float)

    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=spike_counts,
        tuning_rates=tuning,
        transition=transition,
    )

    expected_decoded = np.asarray(m["expected_decoded"], dtype=int).reshape(-1)
    expected_posterior = np.asarray(m["expected_posterior"], dtype=float)

    assert np.array_equal(decoded, expected_decoded)
    assert np.allclose(posterior, expected_posterior, atol=1e-8)


def test_hippocampal_placecell_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/HippocampalPlaceCellExample_gold.mat")
    spike_counts = np.asarray(m["spike_counts_pc"], dtype=float)
    tuning_curves = np.asarray(m["tuning_curves"], dtype=float)

    decoded = DecodingAlgorithms.decode_weighted_center(
        spike_counts=spike_counts,
        tuning_curves=tuning_curves,
    )

    expected = _vec(m, "expected_decoded_weighted")
    assert np.allclose(decoded, expected, atol=1e-8)


def test_spikerate_diff_cis_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/SpikeRateDiffCIs_gold.mat")
    spike_a = np.asarray(m["spike_matrix_a"], dtype=float)
    spike_b = np.asarray(m["spike_matrix_b"], dtype=float)
    alpha = _scalar(m, "alpha_diff")

    diff, lo, hi = DecodingAlgorithms.compute_spike_rate_diff_cis(
        spike_matrix_a=spike_a, spike_matrix_b=spike_b, alpha=alpha
    )

    expected_diff = _vec(m, "expected_diff")
    expected_lo = _vec(m, "expected_lo")
    expected_hi = _vec(m, "expected_hi")

    assert np.allclose(diff, expected_diff, atol=1e-8)
    assert np.allclose(lo, expected_lo, atol=1e-8)
    assert np.allclose(hi, expected_hi, atol=1e-8)


def test_psthe_stimation_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/PSTHEstimation_gold.mat")
    spike_matrix = np.asarray(m["spike_matrix_psth"], dtype=float)
    alpha = _scalar(m, "alpha_psth")

    rate, prob_mat, sig_mat = DecodingAlgorithms.compute_spike_rate_cis(
        spike_matrix=spike_matrix,
        alpha=alpha,
    )

    expected_rate = _vec(m, "expected_rate_psth")
    expected_prob = np.asarray(m["expected_prob_psth"], dtype=float)
    expected_sig = np.asarray(m["expected_sig_psth"], dtype=int)

    assert np.allclose(rate, expected_rate, atol=1e-10)
    assert np.allclose(prob_mat, expected_prob, atol=1e-10)
    assert np.array_equal(sig_mat, expected_sig)


def test_nstcoll_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/nstCollExamples_gold.mat")
    st1_times = _vec(m, "spike_times_1")
    st2_times = _vec(m, "spike_times_2")
    t_start = _scalar(m, "t_start_coll")
    t_end = _scalar(m, "t_end_coll")
    bin_size = _scalar(m, "bin_size_coll")

    st1 = SpikeTrain(spike_times=st1_times, t_start=t_start, t_end=t_end, name="u1")
    st2 = SpikeTrain(spike_times=st2_times, t_start=t_start, t_end=t_end, name="u2")
    coll = SpikeTrainCollection([st1, st2])

    centers_count, count_mat = coll.to_binned_matrix(bin_size_s=bin_size, mode="count")
    centers_binary, binary_mat = coll.to_binned_matrix(bin_size_s=bin_size, mode="binary")

    assert np.allclose(centers_count, _vec(m, "expected_centers"), atol=1e-12)
    assert np.allclose(centers_binary, _vec(m, "expected_centers"), atol=1e-12)
    assert np.allclose(count_mat, np.asarray(m["expected_count_matrix"], dtype=float), atol=1e-12)
    assert np.array_equal(binary_mat, np.asarray(m["expected_binary_matrix"], dtype=float))

    assert np.isclose(coll.get_first_spike_time(), _scalar(m, "expected_first_spike"), atol=1e-12)
    assert np.isclose(coll.get_last_spike_time(), _scalar(m, "expected_last_spike"), atol=1e-12)

    merged = coll.to_spike_train(name="merged").spike_times
    assert np.allclose(merged, _vec(m, "expected_merged_spikes"), atol=1e-12)


def test_covcoll_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/CovCollExamples_gold.mat")
    time = _vec(m, "time_cov")
    cov_stim = _vec(m, "cov_stim")
    cov_ctx = np.asarray(m["cov_ctx"], dtype=float)

    stim = Covariate(time=time, data=cov_stim, name="stim", labels=["stim"])
    ctx = Covariate(time=time, data=cov_ctx, name="ctx", labels=["cosine", "ramp"])
    coll = CovariateCollection([stim, ctx])

    design, labels = coll.design_matrix()
    assert labels == ["stim", "cosine", "ramp"]
    assert np.allclose(design, np.asarray(m["expected_design_cov"], dtype=float), atol=1e-12)

    ctx_only, ctx_labels = coll.data_to_matrix_from_names(["ctx"])
    assert ctx_labels == ["cosine", "ramp"]
    assert np.allclose(ctx_only, np.asarray(m["expected_ctx_only"], dtype=float), atol=1e-12)

    stim_only, stim_labels = coll.data_to_matrix_from_sel([0])
    assert stim_labels == ["stim"]
    assert np.allclose(stim_only.reshape(-1), _vec(m, "expected_stim_only"), atol=1e-12)


def test_trial_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/TrialExamples_gold.mat")
    time = _vec(m, "time_cov")
    cov_stim = _vec(m, "cov_stim")
    cov_ctx = np.asarray(m["cov_ctx"], dtype=float)
    spike_times = _vec(m, "spike_times_trial")
    bin_size = _scalar(m, "bin_size_trial")

    stim = Covariate(time=time, data=cov_stim, name="stim", labels=["stim"])
    ctx = Covariate(time=time, data=cov_ctx, name="ctx", labels=["cosine", "ramp"])
    cov_coll = CovariateCollection([stim, ctx])

    st = SpikeTrain(spike_times=spike_times, t_start=0.0, t_end=1.0, name="u1")
    trial = Trial(spikes=SpikeTrainCollection([st]), covariates=cov_coll)

    t_bins, y, X = trial.aligned_binned_observation(
        bin_size_s=bin_size,
        unit_index=0,
        mode="count",
    )

    assert np.allclose(t_bins, _vec(m, "expected_t_bins_trial"), atol=1e-12)
    assert np.allclose(y.reshape(-1), _vec(m, "expected_y_trial"), atol=1e-12)
    assert np.allclose(X, np.asarray(m["expected_X_trial"], dtype=float), atol=1e-12)


def test_events_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/EventsExamples_gold.mat")
    times = _vec(m, "event_times")
    subset_start = _scalar(m, "subset_start")
    subset_end = _scalar(m, "subset_end")

    events = Events(times=times, labels=["E1", "E2", "E3"])
    subset = events.subset(subset_start, subset_end)

    assert np.allclose(events.times, times, atol=1e-12)
    assert np.allclose(subset.times, _vec(m, "expected_subset_times"), atol=1e-12)


def test_analysis_examples_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/AnalysisExamples_gold.mat")
    X = np.asarray(m["X_analysis"], dtype=float)
    y = _vec(m, "counts_analysis")
    dt = _scalar(m, "dt_analysis")
    b = _vec(m, "b_analysis")

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
    pred = np.asarray(fit.predict(X), dtype=float).reshape(-1)

    assert np.isclose(fit.intercept, b[0], atol=0.35)
    assert np.allclose(fit.coefficients, b[1:], atol=0.35)
    assert np.allclose(pred, _vec(m, "expected_rate_analysis"), atol=0.35)

    rmse = float(np.sqrt(np.mean((pred - _vec(m, "true_rate_analysis")) ** 2)))
    assert np.isclose(rmse, _scalar(m, "expected_rmse_analysis"), atol=0.25)


def test_nstatpaperexamples_plot_arrays_matlab_gold_comparison() -> None:
    combo = _mat("tests/parity/fixtures/matlab_gold/nSTATPaperExamples_plot_gold.mat")

    # Stimulus-rate plot arrays (PPSim section).
    ppsim = _mat("tests/parity/fixtures/matlab_gold/PPSimExample_gold.mat")
    X = np.asarray(ppsim["X"], dtype=float)
    y = _vec(ppsim, "y")
    dt = _scalar(ppsim, "dt")
    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
    pred_rate = np.asarray(fit.predict(X), dtype=float).reshape(-1)
    expected_rate = np.asarray(combo["expected_rate_pp"], dtype=float).reshape(-1)
    rel_err = np.mean(np.abs(pred_rate - expected_rate) / np.maximum(expected_rate, 1e-12))
    assert rel_err <= 0.25

    # Decode-with-history plot arrays.
    dec = _mat("tests/parity/fixtures/matlab_gold/DecodingExampleWithHist_gold.mat")
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=np.asarray(dec["spike_counts"], dtype=float),
        tuning_rates=np.asarray(dec["tuning"], dtype=float),
        transition=np.asarray(dec["transition"], dtype=float),
    )
    expected_decoded = np.asarray(combo["expected_decoded_hist"], dtype=int).reshape(-1)
    expected_post = np.asarray(combo["expected_posterior_hist"], dtype=float)
    assert np.array_equal(decoded, expected_decoded)
    assert np.allclose(posterior, expected_post, atol=1e-8)

    # Place-cell weighted decode arrays.
    place = _mat("tests/parity/fixtures/matlab_gold/HippocampalPlaceCellExample_gold.mat")
    decoded_weighted = DecodingAlgorithms.decode_weighted_center(
        spike_counts=np.asarray(place["spike_counts_pc"], dtype=float),
        tuning_curves=np.asarray(place["tuning_curves"], dtype=float),
    )
    expected_weighted = np.asarray(combo["expected_weighted_decode"], dtype=float).reshape(-1)
    assert np.allclose(decoded_weighted, expected_weighted, atol=1e-8)

    # PSTH significance-matrix arrays.
    psth = _mat("tests/parity/fixtures/matlab_gold/PSTHEstimation_gold.mat")
    rate, prob, sig = DecodingAlgorithms.compute_spike_rate_cis(
        spike_matrix=np.asarray(psth["spike_matrix_psth"], dtype=float),
        alpha=_scalar(psth, "alpha_psth"),
    )
    assert np.allclose(rate, np.asarray(combo["expected_psth_rate"], dtype=float).reshape(-1), atol=1e-10)
    assert np.allclose(prob, np.asarray(combo["expected_psth_prob"], dtype=float), atol=1e-10)
    assert np.array_equal(sig, np.asarray(combo["expected_psth_sig"], dtype=int))

    # mEPSC trace arrays used for data-plot panels.
    trace = np.asarray(combo["trace_mepsc"], dtype=float).reshape(-1)
    time = np.asarray(combo["time_mepsc"], dtype=float).reshape(-1)
    event_times = np.asarray(combo["event_times_mepsc"], dtype=float).reshape(-1)
    assert trace.size == time.size
    assert trace.size > 1000
    assert event_times.size >= 40
    assert np.all(np.diff(time) > 0.0)


def test_decoding_example_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/DecodingExample_gold.mat")
    spike_counts = np.asarray(m["spike_counts_dec"], dtype=float)
    tuning = np.asarray(m["tuning_dec"], dtype=float)
    transition = np.asarray(m["transition_dec"], dtype=float)

    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=spike_counts,
        tuning_rates=tuning,
        transition=transition,
    )

    expected_decoded = np.asarray(m["expected_decoded_dec"], dtype=int).reshape(-1)
    expected_posterior = np.asarray(m["expected_posterior_dec"], dtype=float)
    expected_rmse = _scalar(m, "expected_rmse_dec")
    latent = _vec(m, "latent_zero_dec").astype(int)

    rmse = float(np.sqrt(np.mean((decoded - latent) ** 2)) / max(tuning.shape[1] - 1, 1))
    assert np.array_equal(decoded, expected_decoded)
    assert np.allclose(posterior, expected_posterior, atol=1e-8)
    assert np.isclose(rmse, expected_rmse, atol=1e-8)


def test_explicit_stimulus_whisker_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/ExplicitStimulusWhiskerData_gold.mat")
    stimulus = _vec(m, "stimulus_ws")
    y = _vec(m, "spike_ws")
    b = _vec(m, "b_ws")

    fit = Analysis.fit_glm(X=stimulus[:, None], y=y, fit_type="binomial", dt=1.0)
    pred = np.asarray(fit.predict(stimulus[:, None]), dtype=float).reshape(-1)
    expected_pred = _vec(m, "expected_prob_ws")
    expected_rmse = _scalar(m, "expected_rmse_ws")

    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    assert np.isclose(fit.intercept, b[0], atol=0.2)
    assert np.isclose(fit.coefficients[0], b[1], atol=0.2)
    assert np.allclose(pred, expected_pred, atol=0.1)
    assert np.isclose(rmse, expected_rmse, atol=0.1)


def _detect_mepsc_events(trace: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = -0.12
    refractory = int(round(0.006 / dt))
    candidate = np.where(trace < threshold)[0]
    detected_idx: list[int] = []
    last = -refractory
    for idx in candidate:
        if idx - last >= refractory:
            window_end = min(idx + int(round(0.004 / dt)) + 1, trace.size)
            local = idx + int(np.argmin(trace[idx:window_end]))
            detected_idx.append(local)
            last = local
    det = np.asarray(detected_idx, dtype=int)
    return det * dt, -trace[det]


def test_mepsc_analysis_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/mEPSCAnalysis_gold.mat")
    dt = _scalar(m, "dt_mepsc")
    trace = _vec(m, "trace_mepsc")
    exp_times = _vec(m, "detected_times_mepsc")
    exp_amps = _vec(m, "detected_amps_mepsc")
    exp_count = int(round(_scalar(m, "expected_event_count_mepsc")))
    exp_mean_amp = _scalar(m, "expected_mean_amp_mepsc")

    det_times, det_amps = _detect_mepsc_events(trace, dt)
    events = Events(times=det_times, labels=[f"e{i}" for i in range(det_times.size)])

    assert det_times.size == exp_count
    assert np.allclose(det_times, exp_times, atol=dt)
    assert np.allclose(det_amps, exp_amps, atol=1e-9)
    assert np.isclose(float(np.mean(det_amps)), exp_mean_amp, atol=1e-9)
    assert events.times.size == exp_count
