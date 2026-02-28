from __future__ import annotations

import numpy as np

from nstat.cif import CIFModel
from nstat.signal import Covariate



def test_cif_tier2_methods() -> None:
    rng = np.random.default_rng(2026)
    t = np.linspace(0.0, 1.0, 200)
    X = np.sin(2.0 * np.pi * 3.0 * t)[:, None]

    model = CIFModel(coefficients=np.array([0.4]), intercept=np.log(8.0), link="poisson")
    lam = model.evaluate(X)
    ld = model.eval_lambda_delta(X, dt=t[1] - t[0])
    assert lam.shape == ld.shape
    assert np.all(ld >= 0.0)

    params = model.compute_plot_params(X)
    assert params["max"] >= params["min"]

    payload = model.to_structure()
    recovered = CIFModel.from_structure(payload)
    assert np.allclose(recovered.coefficients, model.coefficients)

    sim = CIFModel.simulate_cif_by_thinning_from_lambda(t, lam, num_realizations=2, rng=rng)
    assert len(sim) == 2



def test_cif_matlab_style_lambda_simulation() -> None:
    t = np.linspace(0.0, 1.0, 1000)
    lam = 5.0 + 2.0 * np.sin(2.0 * np.pi * 2.0 * t)
    lam_cov = Covariate(time=t, data=lam, name="lambda", labels=["lambda"])

    from nstat.compat.matlab import CIF

    coll = CIF.simulateCIFByThinningFromLambda(lam_cov, numRealizations=3)
    assert coll.n_units == 3
