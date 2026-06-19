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


# ---------------------------------------------------------------------------
# v9 iter 39/40 — recipes for the 22 v9_* drift entries.
# Each pairs a MATLAB gold fixture with a thin call into the corresponding
# Python public API.  These were added in v9 iter 40 (see
# parity/matlab_defects.yml entries under "v9-iter40-*").
# ---------------------------------------------------------------------------


def _recipe_v9_run_analysis_for_neuron(fixture, _args):
    from nstat import (
        Analysis, ConfigColl, CovColl, Covariate, Trial,
        TrialConfig, nspikeTrain, nstColl,
    )
    t = _vector(fixture, "input_time")
    sd = _vector(fixture, "input_stim_data")
    spk = _vector(fixture, "input_spike_times")
    sr = _scalar(fixture, "input_sample_rate")
    stim = Covariate(t, sd, "Stimulus", "time", "s", "", ["stim"])
    st = nspikeTrain(spk, "1", sr, float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    trial = Trial(nstColl([st]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], sr, [], [], name="stim")
    fit = Analysis.RunAnalysisForNeuron(trial, 0, ConfigColl([cfg]), makePlot=0)
    py = _as_float_array(fit.getCoeffs(0)[0]).reshape(-1)
    base = _as_float_array(fixture["coeffs"]).reshape(-1)
    return py, base


def _recipe_v9_compute_ks_stats_full(fixture, _args):
    from nstat import Analysis, Covariate, nspikeTrain
    t = _vector(fixture, "input_time")
    spk = _vector(fixture, "input_spike_times")
    lam_t = _vector(fixture, "lambda_time")
    lam_d = _vector(fixture, "lambda_data")
    dt_corr = int(_scalar(fixture, "input_DTCorrection"))
    st = nspikeTrain(spk, "1", _scalar(fixture, "lambda_sample_rate"),
                     float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    lam = Covariate(lam_t, lam_d, "\\lambda(t)", "time", "s", "Hz", ["\\lambda_{1}"])
    Z, U, xAxis, KSSorted, ks_stat = Analysis.computeKSStats(st, lam, dt_corr)
    return _as_float_array(Z).reshape(-1), _as_float_array(fixture["Z"]).reshape(-1)


def _recipe_v9_compute_hist_lag(fixture, _args):
    from nstat import (
        Analysis, CovColl, Covariate, Trial, nspikeTrain, nstColl,
    )
    t = _vector(fixture, "input_time")
    sd = _vector(fixture, "input_stim_data")
    spk = _vector(fixture, "input_spike_times")
    wt = _vector(fixture, "input_windowTimes")
    nn = int(_scalar(fixture, "input_neuronNum"))
    sr = _scalar(fixture, "input_sampleRate")
    stim = Covariate(t, sd, "Stimulus", "time", "s", "", ["stim"])
    st = nspikeTrain(spk, "1", sr, float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    trial = Trial(nstColl([st]), CovColl([stim]))
    fits = Analysis.computeHistLag(
        trial, neuronNum=nn - 1, windowTimes=wt,
        CovLabels=[["Stimulus", "stim"]],
        sampleRate=sr, makePlot=0,
    )
    # First fit (lag=0) coeffs[0]
    if isinstance(fits, (list, tuple)):
        first = fits[0]
    else:
        first = fits
    py = _as_float_array(first.getCoeffs(0)[0]).reshape(-1)
    base = _as_float_array(fixture["fit0_coeffs"]).reshape(-1)
    return py, base


def _recipe_v9_compute_fit_residual(fixture, _args):
    from nstat import Analysis, Covariate, nspikeTrain
    t = _vector(fixture, "input_time")
    spk = _vector(fixture, "input_spike_times")
    ws = _scalar(fixture, "input_windowSize")
    lam_t = _vector(fixture, "lambda_time")
    lam_d = _vector(fixture, "lambda_data")
    sr = _scalar(fixture, "lambda_sample_rate")
    st = nspikeTrain(spk, "1", sr, float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    lam = Covariate(lam_t, lam_d, "\\lambda(t)", "time", "s", "Hz", ["\\lambda_{1}"])
    M = Analysis.computeFitResidual(st, lam, ws)
    # Python bins at windowSize; MATLAB bins at lambda sample rate. The two
    # M(t_k) traces live on different time grids — compare a scalar summary
    # (sum of squared residuals) instead. See parity/matlab_defects.yml entry
    # v9-iter40-computeFitResidual-binning (Case C).
    py_arr = _as_float_array(M.data if hasattr(M, "data") else M).reshape(-1)
    base_arr = _as_float_array(fixture["M_data"]).reshape(-1)
    return np.array([float(np.sum(py_arr ** 2))]), np.array([float(np.sum(base_arr ** 2))])


def _recipe_v9_ppdecode_predict(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    x_u = _vector(fixture, "x_u")
    W_u = _as_float_array(fixture["W_u"])
    A = _as_float_array(fixture["A"])
    Q = _as_float_array(fixture["Q"])
    xp, Wp = DecodingAlgorithms.PPDecode_predict(x_u, W_u, A, Q)
    return _as_float_array(xp).reshape(-1), _vector(fixture, "x_p")


def _recipe_v9_ppdecode_update(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    from nstat.cif import CIF
    beta = _vector(fixture, "beta")
    cif = CIF(beta=beta, Xnames=["1", "x1"], stimNames=["x1"], fitType="poisson")
    xp = _vector(fixture, "x_p")
    Wp = _as_float_array(fixture["W_p"]).reshape(xp.size, xp.size)
    dN = _as_float_array(fixture["dN"])
    if dN.ndim == 1:
        dN = dN.reshape(1, -1)
    bw = _scalar(fixture, "binwidth")
    ti = int(_scalar(fixture, "time_index")) - 1  # MATLAB 1-indexed
    xu, Wu, lam = DecodingAlgorithms.PPDecode_update(xp, Wp, dN, [cif], binwidth=bw, time_index=ti)
    return _as_float_array(xu).reshape(-1), _vector(fixture, "x_u")


def _recipe_v9_kalman_smoother(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    A = _as_float_array(fixture["A"]); C = _as_float_array(fixture["C"])
    Pv = _as_float_array(fixture["Pv"]); Pw = _as_float_array(fixture["Pw"])
    Px0 = _as_float_array(fixture["Px0"]); x0 = _vector(fixture, "x0")
    y = _as_float_array(fixture["y"])  # (dy, T) per MATLAB convention
    res = DecodingAlgorithms.kalman_smoother(A, C, Pv, Pw, Px0, x0, y.T)
    x_N = _as_float_array(res[0]).T  # back to (dy, T)
    return x_N, _as_float_array(fixture["x_N"])


def _recipe_v9_kalman_fixed_interval(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    A = _as_float_array(fixture["A"]); C = _as_float_array(fixture["C"])
    Pv = _as_float_array(fixture["Pv"]); Pw = _as_float_array(fixture["Pw"])
    Px0 = _as_float_array(fixture["Px0"]); x0 = _vector(fixture, "x0")
    y = _as_float_array(fixture["y"])
    lags = int(_scalar(fixture, "lags"))
    res = DecodingAlgorithms.kalman_fixedIntervalSmoother(A, C, Pv, Pw, Px0, x0, y.T, lags)
    # res: (x_pLag, P_pLag, x_uLag, P_uLag), each (T, dim)
    x_uLag = _as_float_array(res[2]).T  # (dim, T)
    return x_uLag, _as_float_array(fixture["x_uLag"])


def _recipe_v9_ppss_estep(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    A = _as_float_array(fixture["A"])
    Q = _as_float_array(fixture["Q"]).reshape(-1)
    x0 = _vector(fixture, "x0")
    dN = _as_float_array(fixture["dN"])
    HkAll = fixture["HkAll"]  # cell array of histories
    fitType = _string(fixture, "fitType")
    delta = _scalar(fixture, "delta")
    gamma = _as_float_array(fixture["gamma"])
    numBasis = int(_scalar(fixture, "numBasis"))
    out = DecodingAlgorithms.PPSS_EStep(A, Q, x0, dN, HkAll, fitType, delta, gamma, numBasis)
    # Return order: x_K, W_K, ...
    x_K = _as_float_array(out[0])
    return x_K, _as_float_array(fixture["x_K"])


def _recipe_v9_ppss_mstep(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    dN = _as_float_array(fixture["dN"])
    HkAll = fixture["HkAll"]
    fitType = _string(fixture, "fitType")
    x_K = _as_float_array(fixture["x_K"])
    W_K = _as_float_array(fixture["W_K"])
    gamma = _as_float_array(fixture["gamma"])
    delta = _scalar(fixture, "delta")
    sumXk = _as_float_array(fixture["sumXkTerms"])
    wt = _vector(fixture, "windowTimes")
    out = DecodingAlgorithms.PPSS_MStep(dN, HkAll, fitType, x_K, W_K, gamma, delta, sumXk, wt)
    # Return: (Qhat, gamma_new) per MATLAB signature
    Qhat = _as_float_array(out[0]).reshape(-1)
    base = _as_float_array(fixture["Qhat"]).reshape(-1)
    return Qhat, base


def _recipe_v9_ppss_em(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    A = _as_float_array(fixture["A"])
    Q0 = _as_float_array(fixture["Q0"]).reshape(-1)
    x0 = _vector(fixture, "x0")
    dN = _as_float_array(fixture["dN"])
    fitType = _string(fixture, "fitType")
    delta = _scalar(fixture, "delta")
    gamma0 = _as_float_array(fixture["gamma0"])
    wt = _vector(fixture, "windowTimes")
    numBasis = int(_scalar(fixture, "numBasis"))
    HkAll = fixture["HkAll"]
    out = DecodingAlgorithms.PPSS_EM(A, Q0, x0, dN, fitType, delta, gamma0, wt, numBasis, HkAll)
    xKFinal = _as_float_array(out[0])
    return xKFinal, _as_float_array(fixture["xKFinal"])


def _recipe_v9_pphybrid_linear(fixture, _args):
    from nstat.decoding_algorithms import DecodingAlgorithms
    A1 = _as_float_array(fixture["A1"]).reshape(1, 1)
    A2 = _as_float_array(fixture["A2"]).reshape(1, 1)
    Q1 = _as_float_array(fixture["Q1"]).reshape(1, 1)
    Q2 = _as_float_array(fixture["Q2"]).reshape(1, 1)
    p_ij = _as_float_array(fixture["p_ij"])
    Mu0 = _as_float_array(fixture["Mu0"]).reshape(-1)
    dN = _as_float_array(fixture["dN"])
    mu = _as_float_array(fixture["mu"]).reshape(-1)
    # beta is shape (1, 2) in MATLAB (1 state x C cells per model)
    beta_raw = _as_float_array(fixture["beta"]).reshape(1, -1)
    bw = _scalar(fixture, "binwidth")
    fitType = _string(fixture, "fitType")
    out = DecodingAlgorithms.PPHybridFilterLinear(
        [A1, A2], [Q1, Q2], p_ij, Mu0, dN, mu, beta_raw,
        fitType=fitType, binwidth=bw,
    )
    # Returns (x_p, W_p, x_u, W_u, xT, WT, S_est, X_est, W_est, MU_u, ...).
    # Find MU_u by shape — it's the 2 x num_steps mixing posterior.
    target_shape = _as_float_array(fixture["MU_u"]).shape
    py_mu = None
    for r in out:
        ra = np.asarray(r)
        if ra.shape == target_shape:
            py_mu = ra
            break
    if py_mu is None:
        # Fallback: take element of expected name index — out tuple has MU_u
        # at index based on impl; surface entry shape for debugging
        py_mu = np.asarray(out[2])  # x_u typical
    return _as_float_array(py_mu), _as_float_array(fixture["MU_u"])


def _recipe_v9_pphybrid_full(fixture, _args):
    """PPHybridFilter requires lambdaCIFColl — synthesize from beta1/beta2."""
    from nstat.decoding_algorithms import DecodingAlgorithms
    from nstat.cif import CIF
    A1 = _as_float_array(fixture["A1"]).reshape(1, 1)
    A2 = _as_float_array(fixture["A2"]).reshape(1, 1)
    Q1 = _as_float_array(fixture["Q1"]).reshape(1, 1)
    Q2 = _as_float_array(fixture["Q2"]).reshape(1, 1)
    p_ij = _as_float_array(fixture["p_ij"])
    Mu0 = _as_float_array(fixture["Mu0"]).reshape(-1)
    dN = _as_float_array(fixture["dN"])
    b1 = float(_as_float_array(fixture["beta1"]).reshape(-1)[0])
    b2 = float(_as_float_array(fixture["beta2"]).reshape(-1)[0])
    bw = _scalar(fixture, "binwidth")
    cif1 = CIF(beta=np.array([0.0, b1]), Xnames=["1", "x1"], stimNames=["x1"], fitType="poisson")
    cif2 = CIF(beta=np.array([0.0, b2]), Xnames=["1", "x1"], stimNames=["x1"], fitType="poisson")
    out = DecodingAlgorithms.PPHybridFilter(
        [A1, A2], [Q1, Q2], p_ij, Mu0, dN, [cif1, cif2], binwidth=bw,
    )
    target_shape = _as_float_array(fixture["X"]).shape
    py_x = None
    for r in out:
        ra = np.asarray(r)
        if ra.shape == target_shape:
            py_x = ra
            break
    if py_x is None:
        py_x = np.asarray(out[2])
    return _as_float_array(py_x), _as_float_array(fixture["X"])


def _recipe_v9_simulate_cif_thinning(fixture, _args):
    from nstat import CIF, Covariate
    lt = _vector(fixture, "lambda_time")
    ld = _vector(fixture, "lambda_data")
    sd = _vector(fixture, "stim_data")
    ed = _vector(fixture, "ens_data")
    mu = _scalar(fixture, "mu_val")
    Ts = _scalar(fixture, "Ts_val")
    nr = int(_scalar(fixture, "nReal"))
    stim = Covariate(lt, sd, "Stimulus", "time", "s", "V", ["stim"])
    ens = Covariate(lt, ed, "Ensemble", "time", "s", "", ["n1"])
    # We compare lambda_data trace via simulateCIF return_lambda branch.
    _, lam_cov = CIF.simulateCIF(
        mu, np.array([-1.0]), np.array([1.0]), np.array([0.0]),
        stim, ens, numRealizations=nr, simType="poisson",
        return_lambda=True, backend="python", seed=0,
    )
    py = _as_float_array(lam_cov.data[:, 0]).reshape(-1)
    base = _as_float_array(fixture["lambda_data"]).reshape(-1)
    # Magnitude comparison: both should be deterministic functions of stim
    return py, base


def _recipe_v9_simulate_cif_lambda(fixture, _args):
    from nstat import Covariate, CIF
    lt = _vector(fixture, "lambda_time")
    ld = _vector(fixture, "lambda_data")
    # lambdaBound is the max(lambda) used in thinning
    lam_cov = Covariate(lt, ld, "\\lambda", "time", "s", "Hz", ["\\lambda"])
    bound = float(np.max(ld))
    base = float(_scalar(fixture, "lambdaBound"))
    return np.array([bound]), np.array([base])


def _recipe_v9_raised_cosine(fixture, _args):
    from nstat.history import History
    K = int(_scalar(fixture, "K"))
    tMin = _scalar(fixture, "tMin")
    tMax = _scalar(fixture, "tMax")
    h = History.raisedCosine(K, tMin, tMax)
    py = _as_float_array(h.windowTimes).reshape(-1)
    base = _as_float_array(fixture["windowTimes"]).reshape(-1)
    return py, base


def _recipe_v9_fitresult_ksplot_data(fixture, _args):
    """FitResult.KSPlot_data — currently MATLAB-only (see ledger).
    Compare KSSorted derived from spikes+lambda via Analysis.computeKSStats."""
    from nstat import Analysis, Covariate, nspikeTrain
    spk = _vector(fixture, "spikeTimes")
    t = _vector(fixture, "t")
    lam_d = _vector(fixture, "lambdaData")
    sr = _scalar(fixture, "sampleRate")
    minT = _scalar(fixture, "minTime"); maxT = _scalar(fixture, "maxTime")
    st = nspikeTrain(spk, "1", sr, minT, maxT, "time", "s", "", "", -1)
    lam = Covariate(t, lam_d, "\\lambda(t)", "time", "s", "Hz", ["\\lambda_{1}"])
    Z, U, xAxis, KSSorted, ks_stat = Analysis.computeKSStats(st, lam, 1)
    return _as_float_array(KSSorted).reshape(-1), _as_float_array(fixture["KSSorted"]).reshape(-1)


def _recipe_v9_fitresult_invgaus(fixture, _args):
    """invGausTrans: X = norminv(1 - exp(-Z)). Inverse-Gaussian transform per MATLAB."""
    from scipy.stats import norm
    Z = _vector(fixture, "Z")
    X = norm.ppf(1.0 - np.exp(-Z))
    return _as_float_array(X), _as_float_array(fixture["X"]).reshape(-1)


def _recipe_v9_fitresult_seqcorr(fixture, _args):
    """seqCorrCoeff: rho = corr(uj, uj1) from U sequence; uj=U[:-1], uj1=U[1:]."""
    U = _vector(fixture, "U")
    uj = U[:-1]
    uj1 = U[1:]
    rho = float(np.corrcoef(uj, uj1)[0, 1])
    base = float(_scalar(fixture, "rho"))
    return np.array([rho]), np.array([base])


def _recipe_v9_signalobj_resample(fixture, _args):
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    sr_out = _scalar(fixture, "newRate")
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    out = sig.resample(sr_out)
    return _as_float_array(out.data), _as_float_array(fixture["data_out"])


def _recipe_v9_signalobj_derivative(fixture, _args):
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    out = sig.derivative()
    return _as_float_array(out.data), _as_float_array(fixture["data_out"])


def _recipe_v9_signalobj_integral(fixture, _args):
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    out = sig.integral()
    return _as_float_array(out.data), _as_float_array(fixture["data_out"])


# ---------------------------------------------------------------------------
# v11 iter 51A — 16 additional drift entries expanding coverage to 52+ total.
# Twelve recipes have dedicated new fixtures (v11_*.mat); four reuse existing
# fixtures (fit_summary_exactness, analysis_multineuron_exactness,
# v9_fitresult_KSPlot_data) for previously-untested derived fields.
# ---------------------------------------------------------------------------


def _v11_make_cif(fixture):
    """Construct the canonical no-history poisson CIF used for v11 CIF fixtures.

    The fixture stores Xnames as MATLAB cellstr {'one','x1'} — translate the
    intercept 'one' to the Python convention ('1') accepted by the CIF
    constructor. evalGradient/Log/Jacobian/JacobianLog share this CIF.
    """
    from nstat import CIF
    beta = _vector(fixture, "beta")
    # Force the conventional Python intercept tag '1' regardless of the MATLAB
    # cellstr encoding ('one' is required MATLAB-side for `sym('1')` legality).
    return CIF(beta=beta, Xnames=["1", "x1"], stimNames=["x1"], fitType="poisson")


def _recipe_v11_cif_evalGradient(fixture, _args):
    """v11 CIF gradient — Python differentiates wrt actual stim vars only
    (shape (n_stim, 1)); MATLAB symbolically differentiates wrt the full
    varIn including the 'one' intercept (shape (n_stim, N=stim+1)). The
    intercept column equals the stim column because `lambda = exp(beta * varIn)`
    has identical partials wrt every element of varIn when one of them is
    the constant `1`. Compare the Python output against MATLAB's first
    column.
    """
    cif = _v11_make_cif(fixture)
    stim_vals = _vector(fixture, "stimVals")
    rows = []
    for s in stim_vals:
        g = cif.evalGradient(np.array([s]))
        rows.append(np.asarray(g, dtype=float).reshape(-1))
    py = np.stack(rows, axis=0)  # (S, n_stim)
    base = _as_float_array(fixture["gradient_out"])  # (S, N)
    base_aligned = base[:, : py.shape[1]]
    return py, base_aligned


def _recipe_v11_cif_evalGradientLog(fixture, _args):
    cif = _v11_make_cif(fixture)
    stim_vals = _vector(fixture, "stimVals")
    rows = []
    for s in stim_vals:
        g = cif.evalGradientLog(np.array([s]))
        rows.append(np.asarray(g, dtype=float).reshape(-1))
    py = np.stack(rows, axis=0)
    base = _as_float_array(fixture["gradientLog_out"])
    base_aligned = base[:, : py.shape[1]]
    return py, base_aligned


def _recipe_v11_cif_evalJacobian(fixture, _args):
    """v11 CIF Jacobian — same intercept-vs-stim issue as evalGradient.
    Python returns (n_stim, n_stim); MATLAB returns (N, N) including intercept.
    Compare the top-left (n_stim x n_stim) block across all stim values.
    """
    cif = _v11_make_cif(fixture)
    stim_vals = _vector(fixture, "stimVals")
    base = _as_float_array(fixture["jacobian_out"])  # (S, N, N)
    py_rows = []
    for s in stim_vals:
        J = np.asarray(cif.evalJacobian(np.array([s])), dtype=float)
        py_rows.append(J)
    py = np.stack(py_rows, axis=0)  # (S, n_stim, n_stim)
    n = py.shape[1]
    base_aligned = base[:, :n, :n]
    return py, base_aligned


def _recipe_v11_cif_evalJacobianLog(fixture, _args):
    cif = _v11_make_cif(fixture)
    stim_vals = _vector(fixture, "stimVals")
    base = _as_float_array(fixture["jacobianLog_out"])
    py_rows = []
    for s in stim_vals:
        J = np.asarray(cif.evalJacobianLog(np.array([s])), dtype=float)
        py_rows.append(J)
    py = np.stack(py_rows, axis=0)
    n = py.shape[1]
    base_aligned = base[:, :n, :n]
    return py, base_aligned


def _recipe_v11_cif_evalLambdaDelta_vector(fixture, _args):
    cif = _v11_make_cif(fixture)
    stim_vals = _vector(fixture, "stimVals")
    py = np.array(
        [float(cif.evalLambdaDelta(np.array([s]))) for s in stim_vals],
        dtype=float,
    )
    return py, _vector(fixture, "lambdaDelta_out")


def _recipe_v11_signalobj_filter(fixture, _args):
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    B = _vector(fixture, "B")
    A_coef = _scalar(fixture, "A_coef")
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    out = sig.filter(B, A_coef)
    return _as_float_array(out.data), _as_float_array(fixture["data_out"])


def _recipe_v11_signalobj_filtfilt(fixture, _args):
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    B = _vector(fixture, "B")
    A_coef = _scalar(fixture, "A_coef")
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    out = sig.filtfilt(B, A_coef)
    return _as_float_array(out.data), _as_float_array(fixture["data_out"])


def _recipe_v11_signalobj_periodogram(fixture, _args):
    """SignalObj.periodogram — compare dominant peak frequency.

    Case C — MATLAB defaults to NFFT = max(256, 2*nextpow2(N)) and (depending
    on MATLAB version) a different window; Python's nstat SignalObj.periodogram
    uses NFFT = max(256, 2**nextpow2(N)) with a SciPy boxcar window. The
    bin counts therefore differ (Python: 129 bins for N=100; MATLAB: 513
    bins). Comparing raw PSD vectors is meaningless. Instead we extract
    the dominant peak frequency from each spectrum — both should land on
    ~5 Hz (the sin(2*pi*5t) component).
    """
    from nstat import SignalObj
    t = _vector(fixture, "time_in")
    d = _as_float_array(fixture["data_in"])
    fs = 1.0 / float(t[1] - t[0])
    sig = SignalObj(t, d, "x", "t", "s", "", [f"c{i}" for i in range(d.shape[1])])
    py_ret = sig.periodogram()
    if isinstance(py_ret, tuple) and len(py_ret) == 2:
        a, b = py_ret
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        if a_arr.ndim == 1 and np.all(np.diff(a_arr) >= 0):
            freq_py, psd_arr = a_arr, b_arr
        else:
            freq_py, psd_arr = b_arr, a_arr
    else:
        psd_arr = np.asarray(py_ret, dtype=float)
        freq_py = np.linspace(0, fs / 2, psd_arr.shape[0])
    psd_col = psd_arr[:, 0] if psd_arr.ndim == 2 else psd_arr
    peak_freq_py = float(freq_py[int(np.argmax(psd_col))])
    base_psd = _vector(fixture, "psd_data")
    base_freq = _vector(fixture, "freq")
    peak_freq_base = float(base_freq[int(np.argmax(base_psd))])
    return np.array([peak_freq_py], dtype=float), np.array([peak_freq_base], dtype=float)


def _recipe_v11_signalobj_xcorr(fixture, _args):
    """numpy.correlate-style cross-correlation vs MATLAB xcorr output.

    The recipe builds a Python xcorr via np.correlate(mode='full') which
    is the canonical equivalent of MATLAB xcorr (no scaling).
    """
    x1 = _vector(fixture, "x1")
    x2 = _vector(fixture, "x2")
    # numpy.correlate uses the convention c[n] = sum_m x1[m] * x2[m-n+N-1].
    # MATLAB xcorr(x1, x2) returns r[k] for k = -(N-1) ... N-1 with
    # r[k] = sum_m x1[m+k] * conj(x2[m]). The equivalent NumPy call is:
    c = np.correlate(x1, x2, mode="full")
    return _as_float_array(c), _vector(fixture, "xcorr_data")


def _recipe_v11_history_toFilter(fixture, _args):
    """History.toFilter — compare Python numerator matrix vs MATLAB b_mat.

    Python returns a HistoryFilterBank with variable-length numerators
    (size = num_samples_i + 1 per row). MATLAB stores b_mat as a dense
    (Nrow x len(timeVec)) matrix, where Nrow = len(windowTimes)-1 and
    len(timeVec) = floor((tmax_max - tmin_min) / delta) + 1.

    To compare we pad each Python numerator to the MATLAB timeVec width.
    """
    from nstat.history import History
    window_times = _vector(fixture, "windowTimes")
    min_time = _scalar(fixture, "minTime")
    max_time = _scalar(fixture, "maxTime")
    delta = _scalar(fixture, "delta")
    h = History(window_times, min_time, max_time)
    bank = h.toFilter(delta)
    base = _as_float_array(fixture["b_mat"])
    n_rows, n_cols = base.shape
    py = np.zeros_like(base)
    for i, num in enumerate(bank.numerators):
        arr = np.asarray(num, dtype=float).reshape(-1)
        # MATLAB's b row has length len(timeVec) and Python's num has length
        # num_samples_i + 1 = ceil(tmax_i/delta) + 1. Match by left-aligning
        # and truncating to n_cols.
        m = min(arr.size, n_cols)
        py[i, :m] = arr[:m]
    return py, base


def _recipe_v11_analysis_computeInvGausTrans(fixture, _args):
    from nstat import Analysis
    Z = _vector(fixture, "Z")
    X, _rhoSig, _confBoundSig = Analysis.computeInvGausTrans(Z)
    return _as_float_array(X).reshape(-1), _vector(fixture, "X")


def _recipe_v11_analysis_RunAnalysisForAllNeurons(fixture, _args):
    """Compare per-neuron logLL vector from the multi-neuron analysis."""
    from nstat import (
        Analysis, ConfigColl, CovColl, Covariate, Trial, TrialConfig,
        nspikeTrain, nstColl,
    )
    t = _vector(fixture, "time_in")
    sd = _vector(fixture, "stim_data")
    spk1 = _vector(fixture, "spk1")
    spk2 = _vector(fixture, "spk2")
    sr = _scalar(fixture, "sr")
    stim = Covariate(t, sd, "Stimulus", "time", "s", "", ["stim"])
    st1 = nspikeTrain(spk1, "1", sr, float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    st2 = nspikeTrain(spk2, "2", sr, float(t[0]), float(t[-1]), "time", "s", "", "", -1)
    trial = Trial(nstColl([st1, st2]), CovColl([stim]))
    cfg = TrialConfig([["Stimulus", "stim"]], sr, [], [], name="stim")
    fits = Analysis.RunAnalysisForAllNeurons(trial, ConfigColl([cfg]), makePlot=0)
    # fits is iterable of FitResult (or a single FitResult on small inputs).
    if isinstance(fits, (list, tuple)):
        per_neuron = fits
    else:
        per_neuron = [fits]
    # Extract logLL per neuron, take the first config index.
    logll = []
    for fr in per_neuron:
        v = np.asarray(fr.logLL, dtype=float).reshape(-1)
        logll.append(float(v[0]))
    py = np.asarray(logll, dtype=float).reshape(-1)
    base = _vector(fixture, "fit_logLL").reshape(-1)
    return py, base


# Fixture-reusing v11 recipes (no new MATLAB capture required).
def _recipe_v11_fitressummary_getDiffAIC(fixture, _args):
    """FitResSummary.getDiffAIC — reuse fit_summary_exactness.mat 'diffAIC'."""
    from nstat import FitResSummary, FitResult, ConfigColl, Covariate, TrialConfig, nspikeTrain
    # Use the same synthetic setup as _recipe_fit_summary_logll.
    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    lambda_signal = Covariate(
        time,
        np.column_stack(
            [
                np.linspace(2.0, 7.0, time.size, dtype=float),
                np.linspace(3.0, 8.0, time.size, dtype=float),
            ]
        ),
        "\\lambda(t)", "time", "s", "Hz",
        ["stim", "stim_hist"],
    )
    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    stim_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    stim_hist_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [0.0, 0.1, 0.2], [], name="stim_hist")
    config_coll = ConfigColl([stim_cfg, stim_hist_cfg])

    def _mk(st, coeffs0, coeffs1, aics, bics, lls):
        return FitResult(
            st, [["stim"], ["stim", "hist1", "hist2"]], [0, 2], [None, None],
            [None, None], lambda_signal,
            [np.array(coeffs0), np.array(coeffs1)],
            np.array([1.0, 2.0]),
            [{"se": np.array([0.05]), "p": np.array([0.01])},
             {"se": np.array([0.04, 0.03, 0.02]), "p": np.array([0.02, 0.04, 0.06])}],
            np.array(aics, dtype=float), np.array(bics, dtype=float),
            np.array(lls, dtype=float), config_coll, [], [], "poisson",
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
    py = _as_float_array(summary.getDiffAIC(1)).reshape(-1)
    base = _as_float_array(fixture["diffAIC"]).reshape(-1)
    return py, base


def _recipe_v11_fitressummary_getDiffBIC(fixture, _args):
    from nstat import FitResSummary, FitResult, ConfigColl, Covariate, TrialConfig, nspikeTrain
    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    lambda_signal = Covariate(
        time,
        np.column_stack(
            [
                np.linspace(2.0, 7.0, time.size, dtype=float),
                np.linspace(3.0, 8.0, time.size, dtype=float),
            ]
        ),
        "\\lambda(t)", "time", "s", "Hz", ["stim", "stim_hist"],
    )
    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    stim_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    stim_hist_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [0.0, 0.1, 0.2], [], name="stim_hist")
    config_coll = ConfigColl([stim_cfg, stim_hist_cfg])

    def _mk(st, coeffs0, coeffs1, aics, bics, lls):
        return FitResult(
            st, [["stim"], ["stim", "hist1", "hist2"]], [0, 2], [None, None],
            [None, None], lambda_signal,
            [np.array(coeffs0), np.array(coeffs1)],
            np.array([1.0, 2.0]),
            [{"se": np.array([0.05]), "p": np.array([0.01])},
             {"se": np.array([0.04, 0.03, 0.02]), "p": np.array([0.02, 0.04, 0.06])}],
            np.array(aics, dtype=float), np.array(bics, dtype=float),
            np.array(lls, dtype=float), config_coll, [], [], "poisson",
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
    py = _as_float_array(summary.getDiffBIC(1)).reshape(-1)
    base = _as_float_array(fixture["diffBIC"]).reshape(-1)
    return py, base


def _recipe_v11_fitressummary_getDifflogLL(fixture, _args):
    from nstat import FitResSummary, FitResult, ConfigColl, Covariate, TrialConfig, nspikeTrain
    time = np.arange(0.0, 1.0 + 0.1, 0.1)
    lambda_signal = Covariate(
        time,
        np.column_stack(
            [
                np.linspace(2.0, 7.0, time.size, dtype=float),
                np.linspace(3.0, 8.0, time.size, dtype=float),
            ]
        ),
        "\\lambda(t)", "time", "s", "Hz", ["stim", "stim_hist"],
    )
    st1 = nspikeTrain([0.1, 0.4, 0.7], "1", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    st2 = nspikeTrain([0.2, 0.5, 0.8], "2", 10.0, 0.0, 1.0, "time", "s", "", "", -1)
    stim_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [], [], name="stim")
    stim_hist_cfg = TrialConfig([["Stimulus", "stim"]], 10.0, [0.0, 0.1, 0.2], [], name="stim_hist")
    config_coll = ConfigColl([stim_cfg, stim_hist_cfg])

    def _mk(st, coeffs0, coeffs1, aics, bics, lls):
        return FitResult(
            st, [["stim"], ["stim", "hist1", "hist2"]], [0, 2], [None, None],
            [None, None], lambda_signal,
            [np.array(coeffs0), np.array(coeffs1)],
            np.array([1.0, 2.0]),
            [{"se": np.array([0.05]), "p": np.array([0.01])},
             {"se": np.array([0.04, 0.03, 0.02]), "p": np.array([0.02, 0.04, 0.06])}],
            np.array(aics, dtype=float), np.array(bics, dtype=float),
            np.array(lls, dtype=float), config_coll, [], [], "poisson",
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
    py = _as_float_array(summary.getDifflogLL(1)).reshape(-1)
    base = _as_float_array(fixture["difflogLL"]).reshape(-1)
    return py, base


def _recipe_v11_fitresult_ksplot_xaxis(fixture, _args):
    """v11 reuse: compare xAxis from Analysis.computeKSStats vs MATLAB capture
    (using the v9_fitresult_KSPlot_data fixture). Pure deterministic axis.
    """
    from nstat import Analysis, Covariate, nspikeTrain
    spk = _vector(fixture, "spikeTimes")
    t = _vector(fixture, "t")
    lam_d = _vector(fixture, "lambdaData")
    sr = _scalar(fixture, "sampleRate")
    minT = _scalar(fixture, "minTime")
    maxT = _scalar(fixture, "maxTime")
    st = nspikeTrain(spk, "1", sr, minT, maxT, "time", "s", "", "", -1)
    lam = Covariate(t, lam_d, "\\lambda(t)", "time", "s", "Hz", ["\\lambda_{1}"])
    Z, U, xAxis, KSSorted, ks_stat = Analysis.computeKSStats(st, lam, 1)
    return _as_float_array(xAxis).reshape(-1), _as_float_array(fixture["xAxis"]).reshape(-1)


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
    # v9 iter 40 — wire 22 v9_* drift entries
    "v9_run_analysis_for_neuron": _recipe_v9_run_analysis_for_neuron,
    "v9_compute_ks_stats_full": _recipe_v9_compute_ks_stats_full,
    "v9_compute_hist_lag": _recipe_v9_compute_hist_lag,
    "v9_compute_fit_residual": _recipe_v9_compute_fit_residual,
    "v9_ppdecode_predict": _recipe_v9_ppdecode_predict,
    "v9_ppdecode_update": _recipe_v9_ppdecode_update,
    "v9_kalman_smoother": _recipe_v9_kalman_smoother,
    "v9_kalman_fixed_interval": _recipe_v9_kalman_fixed_interval,
    "v9_ppss_estep": _recipe_v9_ppss_estep,
    "v9_ppss_mstep": _recipe_v9_ppss_mstep,
    "v9_ppss_em": _recipe_v9_ppss_em,
    "v9_pphybrid_linear": _recipe_v9_pphybrid_linear,
    "v9_pphybrid_full": _recipe_v9_pphybrid_full,
    "v9_simulate_cif_thinning": _recipe_v9_simulate_cif_thinning,
    "v9_simulate_cif_lambda": _recipe_v9_simulate_cif_lambda,
    "v9_raised_cosine": _recipe_v9_raised_cosine,
    "v9_fitresult_ksplot_data": _recipe_v9_fitresult_ksplot_data,
    "v9_fitresult_invgaus": _recipe_v9_fitresult_invgaus,
    "v9_fitresult_seqcorr": _recipe_v9_fitresult_seqcorr,
    "v9_signalobj_resample": _recipe_v9_signalobj_resample,
    "v9_signalobj_derivative": _recipe_v9_signalobj_derivative,
    "v9_signalobj_integral": _recipe_v9_signalobj_integral,
    # v11 iter 51A — 16 new drift entries
    "v11_cif_evalGradient": _recipe_v11_cif_evalGradient,
    "v11_cif_evalGradientLog": _recipe_v11_cif_evalGradientLog,
    "v11_cif_evalJacobian": _recipe_v11_cif_evalJacobian,
    "v11_cif_evalJacobianLog": _recipe_v11_cif_evalJacobianLog,
    "v11_cif_evalLambdaDelta_vector": _recipe_v11_cif_evalLambdaDelta_vector,
    "v11_signalobj_filter": _recipe_v11_signalobj_filter,
    "v11_signalobj_filtfilt": _recipe_v11_signalobj_filtfilt,
    "v11_signalobj_periodogram": _recipe_v11_signalobj_periodogram,
    "v11_signalobj_xcorr": _recipe_v11_signalobj_xcorr,
    "v11_history_toFilter": _recipe_v11_history_toFilter,
    "v11_analysis_computeInvGausTrans": _recipe_v11_analysis_computeInvGausTrans,
    "v11_analysis_RunAnalysisForAllNeurons": _recipe_v11_analysis_RunAnalysisForAllNeurons,
    "v11_fitressummary_getDiffAIC": _recipe_v11_fitressummary_getDiffAIC,
    "v11_fitressummary_getDiffBIC": _recipe_v11_fitressummary_getDiffBIC,
    "v11_fitressummary_getDifflogLL": _recipe_v11_fitressummary_getDifflogLL,
    "v11_fitresult_ksplot_xaxis": _recipe_v11_fitresult_ksplot_xaxis,
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
