from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from nstat.analysis import Analysis
from nstat.decoding import DecodingAlgorithms


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



def test_matlab_gold_manifest_and_checksums() -> None:
    payload = _load_manifest()
    assert payload["version"] == 1
    assert len(payload["fixtures"]) == 3

    for row in payload["fixtures"]:
        path = Path(row["path"])
        assert path.exists(), f"missing fixture {path}"
        assert _sha256(path) == row["sha256"], f"checksum mismatch for {path}"



def test_ppsimexample_matlab_gold_comparison() -> None:
    m = _mat("tests/parity/fixtures/matlab_gold/PPSimExample_gold.mat")
    X = np.asarray(m["X"], dtype=float)
    y = np.asarray(m["y"], dtype=float).reshape(-1)
    dt = float(np.asarray(m["dt"]).reshape(-1)[0])
    b = np.asarray(m["b"], dtype=float).reshape(-1)

    fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)

    assert np.isclose(fit.intercept, b[0], atol=0.25)
    assert np.isclose(fit.coefficients[0], b[1], atol=0.25)

    expected_rate = np.asarray(m["expected_rate"], dtype=float).reshape(-1)
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

    expected = np.asarray(m["expected_decoded_weighted"], dtype=float).reshape(-1)
    assert np.allclose(decoded, expected, atol=1e-8)
