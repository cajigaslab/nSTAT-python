from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial


FIXTURE_MANIFEST = Path("tests/parity/fixtures/manifest.yml")



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
        path = Path(row["path"])
        assert path.exists(), f"missing fixture file: {path}"
        assert _sha256(path) == row["sha256"], f"checksum mismatch for {path}"



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
