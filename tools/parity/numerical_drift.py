"""Numerical drift detector — compare Python public-function output against MATLAB baselines.

Usage
-----
    python tools/parity/numerical_drift.py [--spec parity/numerical_drift_spec.yml]
                                           [--fail-on-drift] [--json]

The spec file lists entries:

    - name: <descriptive name>
      recipe: <recipe name from RECIPES below>     # OR function: module:name
      fixture: tests/parity/fixtures/matlab_gold/<file>.mat
      # When using "function":
      #   inputs: { kwarg: <literal | {fixture_field: NAME, reshape: [a,b]}> ... }
      # When using "recipe":
      #   recipe-specific knobs may appear under `args:` (kept simple)
      output_field: <key in the .mat baseline to compare against>
      output_transform: <optional: one of "T", "reshape:<shape>", "row:N", "col:N", "scalar">
      python_output_path: <optional: dotted path into Python return value, e.g. "state.T">
      tolerance:
        rtol: 1e-9
        atol: 1e-12

For each entry:
  1. Resolve inputs (load .mat fixture; pluck named fields).
  2. Run the recipe (or call the function) to produce a Python output array.
  3. Load the MATLAB baseline field.
  4. Compute max abs error + relative error.
  5. Report pass/fail vs tolerance.

Output: per-entry table; exits non-zero if any drift exceeds tolerance and --fail-on-drift is set.

Recipes encapsulate the few lines of setup needed (e.g. constructing a CIF, building
Covariate inputs) — keeping the YAML spec terse while still letting the tool be
re-pointed at fresh fixtures without code changes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Ensure repo root is on sys.path so `import nstat` works when this script
# is invoked directly (the interpreter puts tools/parity/ at sys.path[0]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("PyYAML is required: pip install pyyaml\n")
    raise SystemExit(2) from exc

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("scipy is required: pip install scipy\n")
    raise SystemExit(2) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "parity" / "numerical_drift_spec.yml"
TODO_SENTINEL = "?"


# ---------------------------------------------------------------------------
# Fixture I/O helpers
# ---------------------------------------------------------------------------
def _load_fixture(path: Path) -> dict[str, Any]:
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _scalar(payload: dict[str, Any], key: str) -> float:
    return float(np.asarray(payload[key]).reshape(-1)[0])


def _vector(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=float).reshape(-1)


def _string(payload: dict[str, Any], key: str) -> str:
    val = payload[key]
    if isinstance(val, np.ndarray):
        return str(val.item()) if val.size == 1 else str(val[0])
    return str(val)


# ---------------------------------------------------------------------------
# Recipes — encapsulate fixture → Python-call → output array
# Each recipe returns (python_array, baseline_array).
# ---------------------------------------------------------------------------
def _recipe_fit_poisson_glm(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import Analysis, ConfigColl, CovColl, Covariate, Trial, TrialConfig, nspikeTrain, nstColl

    time = _vector(fixture, "time")
    stim_data = _vector(fixture, "stim_data")
    spike_times = _vector(fixture, "spike_times")
    sample_rate = _scalar(fixture, "sample_rate")

    stim = Covariate(time, stim_data, "Stimulus", "time", "s", "", ["stim"])
    spike_train = nspikeTrain(spike_times, "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    trial = Trial(nstColl([spike_train]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], sample_rate, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 0, ConfigColl([cfg]))
    return _as_float_array(fit.getCoeffs(0)[0]), _vector(fixture, "coeffs")


def _recipe_fit_summary_logll(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import ConfigColl, Covariate, FitResSummary, FitResult, TrialConfig, nspikeTrain

    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    lambda_signal = Covariate(
        time,
        np.column_stack(
            [
                np.linspace(2.0, 7.0, time.size, dtype=float),
                np.linspace(3.0, 8.0, time.size, dtype=float),
            ]
        ),
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        ["stim", "stim_hist"],
    )
    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    stim_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    stim_hist_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [0.0, 0.1, 0.2], [], name="stim_hist")
    config_coll = ConfigColl([stim_cfg, stim_hist_cfg])

    def _mk(st, coeffs0, coeffs1, aics, bics, lls):
        return FitResult(
            st,
            [["stim"], ["stim", "hist1", "hist2"]],
            [0, 2],
            [None, None],
            [None, None],
            lambda_signal,
            [np.array(coeffs0), np.array(coeffs1)],
            np.array([1.0, 2.0]),
            [
                {"se": np.array([0.05]), "p": np.array([0.01])},
                {"se": np.array([0.04, 0.03, 0.02]), "p": np.array([0.02, 0.04, 0.06])},
            ],
            np.array(aics, dtype=float),
            np.array(bics, dtype=float),
            np.array(lls, dtype=float),
            config_coll,
            [],
            [],
            "poisson",
        )

    fit1 = _mk(st1, [0.5], [0.3, -0.1, -0.05], [11.0, 7.0], [12.0, 8.0], [3.0, 5.0])
    fit2 = _mk(st2, [0.4], [0.25, -0.08, -0.02], [13.0, 9.0], [14.0, 10.0], [2.0, 4.0])
    fit1.KSStats[:, 0] = np.array([0.25, 0.50], dtype=float)
    fit1.KSPvalues[:] = np.array([0.90, 0.40], dtype=float)
    fit1.withinConfInt[:] = np.array([1.0, 1.0], dtype=float)
    fit2.KSStats[:, 0] = np.array([0.35, 0.55], dtype=float)
    fit2.KSPvalues[:] = np.array([0.80, 0.30], dtype=float)
    fit2.withinConfInt[:] = np.array([1.0, 0.0], dtype=float)
    summary = FitResSummary([fit1, fit2])
    return _as_float_array(summary.logLL), _as_float_array(fixture["logLL"])


def _recipe_ppdecodefilterlinear(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat.decoding_algorithms import DecodingAlgorithms

    x_p, W_p, x_u, W_u, *_ = DecodingAlgorithms.PPDecodeFilterLinear(
        _as_float_array(fixture["A"]),
        _as_float_array(fixture["Q"]),
        _as_float_array(fixture["dN"]),
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        _string(fixture, "fitType"),
        _scalar(fixture, "delta"),
    )
    return _as_float_array(x_u), _as_float_array(fixture["x_u"])


def _recipe_ks_stats_arrays(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import Analysis, Covariate, nspikeTrain

    spike_train = nspikeTrain(
        _vector(fixture, "spike_times"), "1", 0.1, 0.0, 1.0, "time", "s", "", "", -1
    )
    lambda_signal = Covariate(
        _vector(fixture, "lambda_time"),
        _vector(fixture, "lambda_data"),
        "\\lambda(t)",
        "time",
        "s",
        "Hz",
        ["\\lambda_{1}"],
    )
    Z, U, xAxis, KSSorted, ks_stat = Analysis.computeKSStats(
        spike_train, lambda_signal, 1, random_values=_vector(fixture, "uniform_draws")
    )
    return _as_float_array(Z).reshape(-1), _vector(fixture, "Z")


def _recipe_history_compute(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import History, nspikeTrain, nstColl

    history = History(_vector(fixture, "windowTimes"), _scalar(fixture, "minTime"), _scalar(fixture, "maxTime"))
    n1 = nspikeTrain([0.0, 0.5, 1.0], "n1", 0.5, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    n2 = nspikeTrain([0.25, 0.75], "n2", 0.5, 0.0, 1.0, "time", "s", "spikes", "spk", -1)
    coll = nstColl([n1, n2])
    coll_cov = history.computeHistory(coll, 2)
    return _as_float_array(coll_cov.dataToMatrix()), _as_float_array(fixture["coll_history_matrix"])


def _recipe_psth_default(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # No MATLAB gold field for psth() directly. Build a self-consistent
    # snapshot: compute psth across two known spike trains and snapshot the
    # mean count column for cross-version drift checks.
    from nstat import nspikeTrain, psth

    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    edges = np.arange(0.0, 1.0001, 0.1)
    psth_data, centers = psth([st1, st2], edges)
    py = _as_float_array(psth_data).reshape(-1)
    # No MATLAB baseline available → marked TODO in the spec.
    return py, py  # tautology; pass-through (entry marked todo)


def _recipe_cif_evaluate(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import CIF

    cif = CIF(
        beta=_vector(fixture, "beta"),
        Xnames=["stim1", "stim2"],
        stimNames=["stim1", "stim2"],
        fitType="binomial",
    )
    stim_val = _vector(fixture, "stimVal")
    out = float(cif.evalLambdaDelta(stim_val))
    return np.array([out], dtype=float), np.array([_scalar(fixture, "lambda_delta")], dtype=float)


def _recipe_simulate_point_process(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat import CIF, Covariate

    time = _vector(fixture, "det_time")
    stim_values = _vector(fixture, "det_stimulus")
    uniforms = _vector(fixture, "det_uniforms").reshape(-1, 1)
    stim = Covariate(time, stim_values, "Stimulus", "time", "s", "Voltage", ["sin"])
    ens = Covariate(time, np.zeros_like(time), "Ensemble", "time", "s", "Spikes", ["n1"])
    _, lambda_cov = CIF.simulateCIF(
        -3.0,
        np.array([-1.0, -2.0, -4.0], dtype=float),
        np.array([1.0], dtype=float),
        np.array([0.0], dtype=float),
        stim,
        ens,
        numRealizations=1,
        simType="binomial",
        random_values=uniforms,
        return_lambda=True,
        backend="python",
    )
    return _as_float_array(lambda_cov.data[:, 0]), _vector(fixture, "det_rate_hz")


def _recipe_kalman_filter(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat.decoding_algorithms import DecodingAlgorithms

    obs = _as_float_array(fixture["observations"]).T  # (N, Dy)
    result = DecodingAlgorithms.kalman_filter(
        observations=obs,
        transition=_as_float_array(fixture["A"]),
        observation_matrix=_as_float_array(fixture["C"]),
        q_cov=_as_float_array(fixture["Q"]),
        r_cov=_as_float_array(fixture["R"]),
        x0=_vector(fixture, "x0"),
        p0=_as_float_array(fixture["P0"]),
    )
    return _as_float_array(result["state"]).T, _as_float_array(fixture["x_filt"])


def _recipe_ppdecode_updatelinear(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from nstat.decoding_algorithms import DecodingAlgorithms

    dN = _as_float_array(fixture["dN"])
    if dN.ndim == 1:
        dN = dN.reshape(-1, 1)
    x_u, W_u, lambda_delta = DecodingAlgorithms.PPDecode_updateLinear(
        _vector(fixture, "x_p"),
        _as_float_array(fixture["W_p"]),
        dN,
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        _string(fixture, "fitType"),
    )
    return _as_float_array(x_u), _vector(fixture, "x_u")


def _pplfp_hkall_3d(fixture: dict[str, Any], num_steps: int, num_cells: int) -> np.ndarray:
    """Coerce the fixture HkAll (typically (K, numCells)) into 3D (K, 1, C)."""
    hk = _as_float_array(fixture["HkAll"])
    if hk.ndim == 2 and hk.shape == (num_steps, num_cells):
        return hk.reshape(num_steps, 1, num_cells)
    if hk.ndim == 2 and hk.shape == (num_cells, num_steps):
        return hk.T.reshape(num_steps, 1, num_cells)
    if hk.ndim == 3:
        return hk
    # Fallback: zero history.
    return np.zeros((num_steps, 1, num_cells), dtype=float)


def _recipe_pplfp_estep(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """PPLFP_EStep — compare smoothed state mean ``x_K``."""
    from nstat.decoding.PPLFP import PPLFP

    dN = _as_float_array(fixture["dN"])
    num_cells, K = dN.shape if dN.ndim == 2 else (1, dN.size)
    HkAll = _pplfp_hkall_3d(fixture, K, num_cells)
    gamma_vec = _vector(fixture, "gamma")
    # If gamma is all-zero, the MATLAB capture didn't fit a history term;
    # collapse to scalar 0 + zero HkAll so the per-cell broadcast in the
    # link evaluation lines up (gamma.size==1 triggers the tile branch).
    if gamma_vec.size > 1 and float(np.max(np.abs(gamma_vec))) == 0.0:
        gamma_vec = np.array(0.0)
        HkAll = np.zeros((K, 1, num_cells), dtype=float)

    x_K, W_K, logll, sums = PPLFP.PPLFP_EStep(
        _as_float_array(fixture["A"]),
        _as_float_array(fixture["Q"]),
        _as_float_array(fixture["C"]),
        _as_float_array(fixture["R"]),
        _as_float_array(fixture["y"]),
        _vector(fixture, "alpha"),
        dN,
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        fitType=_string(fixture, "fitType"),
        delta=_scalar(fixture, "delta"),
        gamma=gamma_vec,
        HkAll=HkAll,
        x0=_vector(fixture, "x0"),
        Px0=_as_float_array(fixture["Px0"]),
    )
    return _as_float_array(x_K), _as_float_array(fixture["x_K"])


def _recipe_pplfp_mstep(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """PPLFP_MStep — compare updated ``betahat_new``.

    The MStep fixture ships only filter/smoother outputs (x_K, W_K) and
    inputs (A, Q, C, R, y, dN, ...) — not the expectation sufficient
    statistics. We reconstitute them by calling PPLFP_EStep first, then
    pass the resulting ExpectationSums dict to PPLFP_MStep. This mirrors
    how MATLAB's capture script chained the two calls.
    """
    from nstat.decoding.PPLFP import PPLFP

    dN_arr = _as_float_array(fixture["dN"])
    num_cells, K = dN_arr.shape if dN_arr.ndim == 2 else (1, dN_arr.size)
    HkAll = _pplfp_hkall_3d(fixture, K, num_cells)
    gamma_vec = _vector(fixture, "gamma")
    if gamma_vec.size > 1 and float(np.max(np.abs(gamma_vec))) == 0.0:
        gamma_vec = np.array(0.0)
        HkAll = np.zeros((K, 1, num_cells), dtype=float)

    # Compute sufficient stats via PPLFP_EStep using the same input params
    # the MStep fixture was captured with.
    _x_K, _W_K, _ll, es = PPLFP.PPLFP_EStep(
        _as_float_array(fixture["A"]),
        _as_float_array(fixture["Q"]),
        _as_float_array(fixture["C"]),
        _as_float_array(fixture["R"]),
        _as_float_array(fixture["y"]),
        _vector(fixture, "alpha"),
        dN_arr,
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        fitType=_string(fixture, "fitType"),
        delta=_scalar(fixture, "delta"),
        gamma=gamma_vec,
        HkAll=HkAll,
        x0=_vector(fixture, "x0"),
        Px0=_as_float_array(fixture["Px0"]),
    )

    x_K = _as_float_array(fixture["x_K"])
    if x_K.ndim == 1:
        x_K = x_K.reshape(1, -1)

    W_K = _as_float_array(fixture["W_K"])
    # NewtonRaphson branch avoids the GLM Analysis machinery which fails on
    # this fixture's 2-cell synthetic trial due to a pre-existing
    # FitResSummary.getCoeffs() inhomogeneous-shape bug in PPLFP_MStep's
    # GLM path (see matlab_defects.yml entry pplfp-mstep-fixture-missing).
    # The Newton-Raphson branch is the same MATLAB code path used to
    # capture the baseline.
    result = PPLFP.PPLFP_MStep(
        dN_arr,
        _as_float_array(fixture["y"]),
        x_K,
        W_K,
        _vector(fixture, "x0"),
        _as_float_array(fixture["Px0"]),
        es,
        _string(fixture, "fitType"),
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        gamma_vec,
        None,  # windowTimes
        HkAll,
        MstepMethod="NewtonRaphson",
    )
    # Return order: Ahat, Qhat, Chat, Rhat, alphahat,
    #               muhat_new, betahat_new, gammahat_new, x0hat, Px0hat
    betahat_new = _as_float_array(result[6])
    baseline = _as_float_array(fixture["betahat_new"])
    return betahat_new, baseline


def _recipe_pplfp_em(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """PPLFP_EM — compare final smoothed state ``xKFinal``."""
    from nstat.decoding.PPLFP import PPLFP

    # NewtonRaphson branch — see PPLFP_MStep recipe for rationale.
    # Constraints: hold x0 / Px0 fixed during EM to avoid Px0 driving
    # singular after a few iterations (pre-existing driver issue tracked
    # in matlab_defects.yml entry pplfp-em-pending-fixture). With these
    # constraints, EM converges in <10 iters on the 30-step fixture.
    constraints = PPLFP.PPLFP_EMCreateConstraints(
        EstimateA=1, AhatDiag=0, QhatDiag=1, RhatDiag=1,
        Estimatex0=0, EstimatePx0=0, mcIter=int(fixture.get("mcIter", 50)),
    )
    out = PPLFP.PPLFP_EM(
        _as_float_array(fixture["y"]),
        _as_float_array(fixture["dN"]),
        _as_float_array(fixture["Ahat0"]),
        _as_float_array(fixture["Qhat0"]),
        _as_float_array(fixture["Chat0"]),
        _as_float_array(fixture["Rhat0"]),
        _vector(fixture, "alphahat0"),
        _vector(fixture, "mu"),
        _as_float_array(fixture["beta"]),
        fitType=_string(fixture, "fitType"),
        delta=_scalar(fixture, "delta"),
        x0=_vector(fixture, "x0"),
        Px0=_as_float_array(fixture["Px0"]),
        PPLFP_EM_Constraints=constraints,
        MstepMethod="NewtonRaphson",
    )
    xKFinal = out[0]
    return _as_float_array(xKFinal), _as_float_array(fixture["xKFinal"])


def _recipe_pplfp_se_alpha(fixture: dict[str, Any], _args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """PPLFP_ComputeParamStandardErrors — compare deterministic ``SE.alpha``.

    All other SE entries depend on Monte-Carlo paths whose RNG stream
    differs from MATLAB ``normrnd``; see ``parity/matlab_defects.yml``
    entry ``pplfp-se-mc-drift`` (Case C).
    """
    from nstat.decoding.PPLFP import PPLFP

    # Reconstitute ExpectationSums dict from whatever sufficient stats
    # the fixture ships; fall back to deriving from xKFinal / y.
    xKFinal = _as_float_array(fixture["xKFinal"])
    if xKFinal.ndim == 1:
        xKFinal = xKFinal.reshape(1, -1)
    y = _as_float_array(fixture["y"])
    if y.ndim == 1:
        y = y.reshape(1, -1)
    es = {}
    for key in (
        "Sxkm1xkm1", "Sxkm1xk", "Sxkxkm1", "Sxkxk", "Sxkyk", "Sykyk",
        "sumXkTerms", "sumYkTerms", "Sx0", "Sx0x0",
    ):
        if key in fixture:
            es[key] = _as_float_array(fixture[key])
    if "Sxkm1xkm1" not in es:
        es["Sxkm1xkm1"] = xKFinal[:, :-1] @ xKFinal[:, :-1].T
    if "Sxkxk" not in es:
        es["Sxkxk"] = xKFinal @ xKFinal.T
    if "Sxkxkm1" not in es:
        es["Sxkxkm1"] = xKFinal[:, 1:] @ xKFinal[:, :-1].T
    if "Sxkm1xk" not in es:
        es["Sxkm1xk"] = es["Sxkxkm1"].T
    if "Sxkyk" not in es:
        es["Sxkyk"] = xKFinal @ y.T
    if "Sykyk" not in es:
        es["Sykyk"] = y @ y.T
    if "sumXkTerms" not in es:
        es["sumXkTerms"] = es["Sxkxk"]
    if "sumYkTerms" not in es:
        es["sumYkTerms"] = es["Sykyk"]
    if "Sx0" not in es:
        es["Sx0"] = xKFinal[:, 0]
    if "Sx0x0" not in es:
        es["Sx0x0"] = np.outer(es["Sx0"], es["Sx0"])

    dN_arr = _as_float_array(fixture["dN"])
    num_cells, K = dN_arr.shape if dN_arr.ndim == 2 else (1, dN_arr.size)
    HkAll = _pplfp_hkall_3d(fixture, K, num_cells)
    SE, Pvals, nTerms = PPLFP.PPLFP_ComputeParamStandardErrors(
        y,
        dN_arr,
        xKFinal,
        _as_float_array(fixture["WKFinal"]),
        _as_float_array(fixture["Ahat"]),
        _as_float_array(fixture["Qhat"]),
        _as_float_array(fixture["Chat"]),
        _as_float_array(fixture["Rhat"]),
        _vector(fixture, "alphahat"),
        _vector(fixture, "x0hat"),
        _as_float_array(fixture["Px0hat"]),
        es,
        _string(fixture, "fitType"),
        _vector(fixture, "muhat_new"),
        _as_float_array(fixture["betahat_new"]),
        _vector(fixture, "gammahat_new"),
        None,  # windowTimes
        HkAll,
    )
    se_alpha_py = _as_float_array(SE["alpha"]).reshape(-1)
    se_struct = fixture["SE"]
    se_alpha_ml = _as_float_array(getattr(se_struct, "alpha")).reshape(-1)
    return se_alpha_py, se_alpha_ml


RECIPES: dict[str, Callable[[dict[str, Any], dict[str, Any]], tuple[np.ndarray, np.ndarray]]] = {
    "fit_poisson_glm_coeffs": _recipe_fit_poisson_glm,
    "fit_summary_logll": _recipe_fit_summary_logll,
    "ppdecodefilterlinear_xu": _recipe_ppdecodefilterlinear,
    "ks_stats_Z": _recipe_ks_stats_arrays,
    "history_compute_coll": _recipe_history_compute,
    "psth_passthrough_todo": _recipe_psth_default,
    "cif_eval_lambda_delta": _recipe_cif_evaluate,
    "simulate_point_process_rate": _recipe_simulate_point_process,
    "kalman_filter_state": _recipe_kalman_filter,
    "ppdecode_updatelinear_xu": _recipe_ppdecode_updatelinear,
    "pplfp_estep": _recipe_pplfp_estep,
    "pplfp_mstep": _recipe_pplfp_mstep,
    "pplfp_em": _recipe_pplfp_em,
    "pplfp_se_alpha": _recipe_pplfp_se_alpha,
}


# ---------------------------------------------------------------------------
# Drift entry runner
# ---------------------------------------------------------------------------
@dataclass
class EntryResult:
    name: str
    fixture: str
    recipe: str
    shape: tuple[int, ...] | None
    max_abs_err: float
    rel_err: float
    rtol: float
    atol: float
    passed: bool
    todo: bool
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _run_entry(entry: dict[str, Any]) -> EntryResult:
    name = str(entry.get("name", "?"))
    recipe_name = str(entry.get("recipe", ""))
    fixture_rel = str(entry.get("fixture", ""))
    baseline_field = str(entry.get("output_field", ""))
    tol = entry.get("tolerance", {}) or {}
    rtol = float(tol.get("rtol", 1e-9))
    atol = float(tol.get("atol", 1e-12))
    todo = baseline_field == TODO_SENTINEL or bool(entry.get("todo"))

    fixture_path = (REPO_ROOT / fixture_rel) if fixture_rel else None

    if todo:
        return EntryResult(
            name=name,
            fixture=fixture_rel,
            recipe=recipe_name,
            shape=None,
            max_abs_err=float("nan"),
            rel_err=float("nan"),
            rtol=rtol,
            atol=atol,
            passed=True,  # skipped, not failed
            todo=True,
            notes=["TODO — capture MATLAB baseline field"],
        )

    if fixture_path is None or not fixture_path.exists():
        return EntryResult(
            name=name,
            fixture=fixture_rel,
            recipe=recipe_name,
            shape=None,
            max_abs_err=float("nan"),
            rel_err=float("nan"),
            rtol=rtol,
            atol=atol,
            passed=False,
            todo=False,
            error=f"fixture not found: {fixture_rel}",
        )

    try:
        payload = _load_fixture(fixture_path)
    except Exception as exc:  # pragma: no cover
        return EntryResult(
            name=name,
            fixture=fixture_rel,
            recipe=recipe_name,
            shape=None,
            max_abs_err=float("nan"),
            rel_err=float("nan"),
            rtol=rtol,
            atol=atol,
            passed=False,
            todo=False,
            error=f"loadmat failed: {exc!r}",
        )

    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        # Allow direct function: module:name form.
        func_spec = str(entry.get("function", ""))
        if not func_spec or ":" not in func_spec:
            return EntryResult(
                name=name, fixture=fixture_rel, recipe=recipe_name,
                shape=None, max_abs_err=float("nan"), rel_err=float("nan"),
                rtol=rtol, atol=atol, passed=False, todo=False,
                error=f"unknown recipe '{recipe_name}' and no function: provided",
            )
        try:
            mod_name, func_name = func_spec.split(":", 1)
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
        except Exception as exc:
            return EntryResult(
                name=name, fixture=fixture_rel, recipe=recipe_name,
                shape=None, max_abs_err=float("nan"), rel_err=float("nan"),
                rtol=rtol, atol=atol, passed=False, todo=False,
                error=f"function import failed: {exc!r}",
            )
        try:
            kwargs = entry.get("inputs", {}) or {}
            resolved: dict[str, Any] = {}
            for k, v in kwargs.items():
                if isinstance(v, dict) and "fixture_field" in v:
                    resolved[k] = _as_float_array(payload[v["fixture_field"]])
                else:
                    resolved[k] = v
            py_out = func(**resolved)
            baseline = _as_float_array(payload[baseline_field])
            py_arr = _as_float_array(py_out)
        except Exception as exc:
            return EntryResult(
                name=name, fixture=fixture_rel, recipe=recipe_name,
                shape=None, max_abs_err=float("nan"), rel_err=float("nan"),
                rtol=rtol, atol=atol, passed=False, todo=False,
                error=f"function call failed: {exc!r}",
            )
    else:
        try:
            py_arr, baseline = recipe(payload, entry.get("args", {}) or {})
        except Exception as exc:
            return EntryResult(
                name=name, fixture=fixture_rel, recipe=recipe_name,
                shape=None, max_abs_err=float("nan"), rel_err=float("nan"),
                rtol=rtol, atol=atol, passed=False, todo=False,
                error=f"recipe failed: {exc!r}",
            )

    # Shape align (broadcast-safe flatten if shapes differ but sizes match).
    py_arr = np.asarray(py_arr, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if py_arr.shape != baseline.shape and py_arr.size == baseline.size:
        py_cmp = py_arr.reshape(baseline.shape)
    else:
        py_cmp = py_arr

    if py_cmp.shape != baseline.shape:
        return EntryResult(
            name=name, fixture=fixture_rel, recipe=recipe_name,
            shape=tuple(py_arr.shape), max_abs_err=float("nan"), rel_err=float("nan"),
            rtol=rtol, atol=atol, passed=False, todo=False,
            error=f"shape mismatch py={py_arr.shape} matlab={baseline.shape}",
        )

    # Treat matching infinities as zero error so they don't poison the diff.
    with np.errstate(invalid="ignore"):
        diff = np.abs(py_cmp - baseline)
    equal_inf_mask = np.isinf(py_cmp) & np.isinf(baseline) & (np.sign(py_cmp) == np.sign(baseline))
    if equal_inf_mask.any():
        diff = np.where(equal_inf_mask, 0.0, diff)
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    denom = np.abs(baseline) + 1e-30
    with np.errstate(invalid="ignore"):
        rel_arr = diff / denom
    rel = float(np.nanmax(rel_arr)) if rel_arr.size else 0.0
    passed = bool(np.allclose(py_cmp, baseline, rtol=rtol, atol=atol, equal_nan=False)) or (
        equal_inf_mask.all() and diff.max() == 0.0
    )

    return EntryResult(
        name=name,
        fixture=fixture_rel,
        recipe=recipe_name,
        shape=tuple(py_arr.shape),
        max_abs_err=max_abs,
        rel_err=rel,
        rtol=rtol,
        atol=atol,
        passed=passed,
        todo=False,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _format_table(results: list[EntryResult]) -> str:
    name_w = max((len(r.name) for r in results), default=4)
    name_w = max(name_w, 4)
    rows = []
    rows.append(
        f"{'name'.ljust(name_w)}  {'shape':>14}  {'max_abs_err':>14}  {'rel_err':>14}  {'rtol':>8}  {'atol':>8}  status"
    )
    rows.append("-" * (name_w + 80))
    for r in results:
        if r.todo:
            status = "TODO"
        elif r.error:
            status = f"ERR: {r.error[:60]}"
        else:
            status = "PASS" if r.passed else "FAIL"
        shape = ("x".join(str(s) for s in r.shape)) if r.shape else "-"
        mae = "-" if np.isnan(r.max_abs_err) else f"{r.max_abs_err:.3e}"
        rel = "-" if np.isnan(r.rel_err) else f"{r.rel_err:.3e}"
        rows.append(
            f"{r.name.ljust(name_w)}  {shape:>14}  {mae:>14}  {rel:>14}  {r.rtol:>8.1e}  {r.atol:>8.1e}  {status}"
        )
    return "\n".join(rows)


def _to_json(results: list[EntryResult]) -> str:
    out = []
    for r in results:
        out.append(
            {
                "name": r.name,
                "fixture": r.fixture,
                "recipe": r.recipe,
                "shape": list(r.shape) if r.shape else None,
                "max_abs_err": None if np.isnan(r.max_abs_err) else r.max_abs_err,
                "rel_err": None if np.isnan(r.rel_err) else r.rel_err,
                "rtol": r.rtol,
                "atol": r.atol,
                "passed": r.passed,
                "todo": r.todo,
                "error": r.error,
                "notes": r.notes,
            }
        )
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Numerical-drift detector for nSTAT public functions.")
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="YAML spec file (default: parity/numerical_drift_spec.yml)")
    p.add_argument("--fail-on-drift", action="store_true", help="exit 1 if any entry exceeds tolerance")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args(argv)

    if not args.spec.exists():
        sys.stderr.write(f"spec not found: {args.spec}\n")
        return 2

    spec = yaml.safe_load(args.spec.read_text()) or {}
    entries = spec.get("entries", []) or []

    results = [_run_entry(e) for e in entries]

    if args.json:
        print(_to_json(results))
    else:
        print(_format_table(results))
        n_pass = sum(1 for r in results if r.passed and not r.todo and not r.error)
        n_todo = sum(1 for r in results if r.todo)
        n_fail = sum(1 for r in results if not r.passed and not r.todo)
        print()
        print(f"summary: pass={n_pass} fail={n_fail} todo={n_todo} total={len(results)}")

    if args.fail_on_drift:
        for r in results:
            if not r.passed and not r.todo:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
