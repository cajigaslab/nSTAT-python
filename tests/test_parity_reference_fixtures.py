from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from nstat.analysis import Analysis
from nstat.compat.matlab import FitResult as MatlabFitResult
from nstat.compat.matlab import FitResSummary as MatlabFitResSummary
from nstat.decoding import DecodingAlgorithms
from nstat.fit import FitResult
from nstat.fit import FitSummary
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial


FIXTURE_MANIFEST = Path("tests/parity/fixtures/manifest.yml")
REPO_ROOT = Path(__file__).resolve().parents[1]



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _load_manifest() -> dict:
    return yaml.safe_load(FIXTURE_MANIFEST.read_text(encoding="utf-8"))



def test_fixture_manifest_present() -> None:
    assert FIXTURE_MANIFEST.exists()
    payload = _load_manifest()
    assert payload["version"] == 1
    assert len(payload["fixtures"]) >= 3



def test_fixture_checksums_match_manifest() -> None:
    payload = _load_manifest()
    for row in payload["fixtures"]:
        path = (REPO_ROOT / Path(row["path"])).resolve()
        assert path.exists(), f"missing fixture file: {path}"
        assert _sha256(path) == row["sha256"], f"checksum mismatch for {path}"



def test_fixture_manifest_paths_are_repo_relative() -> None:
    payload = _load_manifest()
    for row in payload["fixtures"]:
        path_text = str(row["path"])
        path = Path(path_text)
        assert not path.is_absolute(), f"fixture path must be relative: {path_text}"
        assert not path_text.startswith(".."), f"fixture path cannot escape repo root: {path_text}"



def test_analysis_poisson_fixture_regression() -> None:
    fixture = np.load("tests/parity/fixtures/analysis_poisson_glm.npz")
    X = fixture["X"]
    y = fixture["y"]
    dt = float(fixture["dt"][0])

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)

    assert np.allclose(fit.coefficients, fixture["expected_coefficients"], atol=1e-10)
    assert np.isclose(fit.intercept, float(fixture["expected_intercept"][0]), atol=1e-10)
    assert np.isclose(fit.log_likelihood, float(fixture["expected_log_likelihood"][0]), atol=1e-8)
    assert np.allclose(fit.predict(X), fixture["expected_rate"], atol=1e-10)



def test_decoding_fixture_regression() -> None:
    fixture = np.load("tests/parity/fixtures/decoding_state_posterior.npz")
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(
        spike_counts=fixture["spike_counts"],
        tuning_rates=fixture["tuning_rates"],
        transition=fixture["transition"],
    )

    assert np.array_equal(decoded, fixture["expected_decoded"])
    assert np.allclose(posterior, fixture["expected_posterior"], atol=1e-10)



def test_trial_alignment_fixture_regression() -> None:
    fixture = np.load("tests/parity/fixtures/trial_alignment.npz")
    time = fixture["time"]
    cov_data = fixture["cov_data"]
    spike_times = fixture["spike_times"]
    bin_size = float(fixture["bin_size"][0])

    trial = Trial(
        spikes=SpikeTrainCollection([SpikeTrain(spike_times=spike_times, t_start=0.0, t_end=1.0)]),
        covariates=CovariateCollection([
            Covariate(time=time, data=cov_data, name="stim", labels=["stim"]),
        ]),
    )

    tb, y, X = trial.aligned_binned_observation(bin_size_s=bin_size, unit_index=0, mode="count")
    assert np.allclose(tb, fixture["expected_time_bins"], atol=1e-12)
    assert np.array_equal(y, fixture["expected_counts"])
    assert np.allclose(X, fixture["expected_design"], atol=1e-12)


def test_fit_summary_structure_fixture_regression() -> None:
    fixture = np.load("tests/parity/fixtures/fit_summary_structure.npz")

    structures = []
    for i in range(fixture["result_coefficients"].shape[0]):
        structures.append(
            {
                "coefficients": fixture["result_coefficients"][i],
                "intercept": float(fixture["result_intercepts"][i]),
                "fit_type": str(fixture["result_fit_types"][i]),
                "log_likelihood": float(fixture["result_log_likelihoods"][i]),
                "n_samples": int(fixture["result_n_samples"][i]),
                "n_parameters": int(fixture["result_n_parameters"][i]),
                "parameter_labels": [str(v) for v in fixture["result_labels"][i]],
            }
        )
    payload = {"results": structures}

    native = FitSummary.from_structure(payload)
    coeff_mat, labels, se_mat = native.get_coeffs()
    hist_counts, hist_edges, hist_percent_sig = native.bin_coeffs(min_val=-1.0, max_val=1.0, bin_size=0.2)

    assert np.array_equal(np.array(labels, dtype="<U16"), fixture["expected_unique_labels"])
    assert np.allclose(coeff_mat, fixture["expected_coeff_matrix"], atol=1e-12, equal_nan=True)
    assert np.allclose(se_mat, fixture["expected_se_matrix"], atol=1e-12, equal_nan=True)
    assert np.array_equal(hist_counts, fixture["expected_hist_counts"])
    assert np.allclose(hist_edges, fixture["expected_hist_edges"], atol=1e-12)
    assert np.allclose(hist_percent_sig, fixture["expected_hist_percent_sig"], atol=1e-12)

    compat = MatlabFitResSummary.fromStructure(payload)
    coeff_mat_compat, labels_compat, _se_compat = compat.getCoeffs(1)
    assert np.allclose(coeff_mat_compat, fixture["expected_coeff_matrix"], atol=1e-12, equal_nan=True)
    assert labels_compat == labels
    assert len(compat.toStructure()["results"]) == len(structures)


def test_fit_result_roundtrip_fixture_regression() -> None:
    fixture = np.load("tests/parity/fixtures/fit_result_roundtrip.npz")

    payload = {
        "coefficients": fixture["coefficients"],
        "intercept": float(fixture["intercept"][0]),
        "fit_type": str(fixture["fit_type"][0]),
        "log_likelihood": float(fixture["log_likelihood"][0]),
        "n_samples": int(fixture["n_samples"][0]),
        "n_parameters": int(fixture["n_parameters"][0]),
        "parameter_labels": [str(v) for v in fixture["parameter_labels"]],
        "ks_stats": {
            "ks_stat": fixture["ks_stat"],
            "pValue": fixture["p_value"],
            "withinConfInt": fixture["within_conf_int"],
        },
        "fit_residual": fixture["fit_residual"],
        "inv_gaus_stats": {"z": fixture["inv_gaus_z"]},
        "neuron_name": str(fixture["neuron_name"][0]),
    }

    native = FitResult.from_structure(payload)
    plot = native.get_plot_params()
    assert np.isclose(native.aic(), float(fixture["expected_aic"][0]), atol=1e-12)
    assert np.isclose(native.bic(), float(fixture["expected_bic"][0]), atol=1e-12)
    assert np.allclose(plot["bAct"], fixture["expected_plot_bact"], atol=1e-12)
    assert np.allclose(plot["seAct"], fixture["expected_plot_seact"], atol=1e-12)
    assert np.allclose(plot["sigIndex"], fixture["expected_plot_sigindex"], atol=1e-12)
    assert np.array_equal(np.array(plot["xLabels"], dtype="<U32"), fixture["expected_plot_xlabels"])

    compat = MatlabFitResult.fromStructure(payload)
    compat.setKSStats(
        fixture["ks_stat"],
        fixture["p_value"],
        fixture["within_conf_int"],
    ).setFitResidual(fixture["fit_residual"]).setNeuronName(str(fixture["neuron_name"][0]))
    compat_plot = compat.getPlotParams()
    roundtrip = MatlabFitResult.fromStructure(compat.toStructure())
    assert np.allclose(roundtrip.coefficients, fixture["expected_roundtrip_coefficients"], atol=1e-12)
    assert np.allclose(roundtrip.fit_residual, fixture["expected_roundtrip_fit_residual"], atol=1e-12)
    assert np.allclose(roundtrip.ks_stats["ks_stat"], fixture["expected_roundtrip_ks_stat"], atol=1e-12)
    assert roundtrip.neuron_name == str(fixture["expected_roundtrip_neuron_name"][0])
    assert np.allclose(compat_plot["bAct"], fixture["expected_roundtrip_plot_bact"], atol=1e-12)
