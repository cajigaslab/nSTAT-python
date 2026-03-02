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
    assert len(payload["fixtures"]) == 9

    for row in payload["fixtures"]:
        path = Path(row["path"])
        assert path.exists(), f"missing fixture {path}"
        assert _sha256(path) == row["sha256"], f"checksum mismatch for {path}"


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
