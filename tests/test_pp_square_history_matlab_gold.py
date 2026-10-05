"""PPAF filters with a square history (nW == C) and the MATLAB history-window
rule, against MATLAB gold.

Gold: ``tests/parity/fixtures/matlab_gold/pp_square_history.mat``, captured
from MATLAB master by ``tools/parity/matlab/capture_pp_square_history.m``
(rng(42) synthetic inputs; dx = 2 states, delta = 1 ms unless noted,
non-symmetric gamma).

MATLAB reorients the history coefficients only with
``if(size(gamma,2)~=C) gamma=gamma'; end`` (PPAF.m PPDecodeFilterLinear and
PP_fixedIntervalSmoother) and ``PPDecode_updateLinear`` / ``PP_EStep`` use
gamma as given, so a square ``nW x C`` gamma is never transposed.  The Python
``_normalize_gamma`` used to transpose any gamma shaped ``(C, nW)`` -- which a
square gamma always is -- so ``PPDecodeFilterLinear`` (one transpose) and
``PP_EStep`` (one transpose, inside ``PPDecode_updateLinear``) used the wrong
window/cell pairing whenever nW == C.  ``PP_fixedIntervalSmoother`` transposed
twice (once itself, once in ``PPDecode_updateLinear``) and so already matched.

Cases:

* ``pdfl_pois_sq`` / ``pdfl_binom_sq`` -- ``PPDecodeFilterLinear``, nW == C
  (3 and 4); every step of x_p, W_p, x_u, W_u, on both the Numba and the
  pure-Python path.
* ``pfis_pois_sq`` -- ``PP_fixedIntervalSmoother`` (lags = 1), nW == C = 3.
  Every column, including x_pLag(:,2) = x_u(:,1), which MATLAB fills on its
  first step and the Python port used to leave at zero.
* ``pdfl_pois_ctrl`` -- ``PPDecodeFilterLinear`` control, nW = 2 != C = 3.
* ``estep_pois_sq`` -- ``PP_EStep`` x_K / W_K, nW == C = 3.
* ``estep_pois_NeqC`` -- ``PP_EStep`` x_K / W_K, N == C = 6 time bins and
  cells.  ``_normalize_history_tensor`` used to look the layout up in a dict
  keyed by candidate shapes; with N == C the canonical ``(N, nW, C)`` key
  collided with ``(C, nW, N)`` and the tensor was silently transposed
  (x_K off by 2.8e-1).
* ``estep_binom_C1`` -- ``PP_EStep`` x_K / W_K, C = 1, fed MATLAB's ``N x nW``
  history exactly as MATLAB stores it (it drops the trailing singleton cell
  axis); this used to raise ``ValueError``.
* ``pdfl_pois_offgrid`` / ``pdfl_pois_colon`` / ``pdfl_binom_delta2`` --
  ``PPDecodeFilterLinear`` with window edges off the 1 ms grid
  ([0 1.5 4 6.5] ms), edges ``0:delta:(9+1)*delta`` and delta = 2 ms.
  ``pdfl_pois_colon`` pins MATLAB's rounding of those edges only (its 0.009 s
  edge is 9.000000000000002 samples, so MATLAB leaves the last window empty); a
  real default call with numel(gamma) = 9 would pass a 9-row gamma, this case a
  10-row one.

Outside ``case_names``:

* ``emdef_*`` -- ``PPLFP_EM``'s default history (windowTimes = [] and a non-zero
  gamma): MATLAB PPLFP.m:1613-1636 with ``length(gamma)`` = 8 (an 8 x 2 gamma)
  at delta = 1 ms, where ``0:delta:9*delta`` differs bitwise from
  ``np.arange(10) * delta`` and moves spikes between windows 4-7.
* ``colon_*`` -- 487 MATLAB ``a:d:b`` outputs pinning
  ``nstat.core._matlab_colon_exact``.

For the ``PP_EStep`` cases the log-likelihood is not captured: for nW == C
MATLAB's logll transposes the square history slice, a suspected MATLAB defect
pending a fix (the Python port mirrors it).

History from ``windowTimes``.  ``PPDecodeFilterLinear`` and
``PP_fixedIntervalSmoother`` build the history tensor internally with the
private ``_compute_history_terms``.  It used to count lags in
``[t_start, t_stop)`` -- one bin earlier than MATLAB's
``History.computeHistory``, whose window ``[t_i, t_(i+1)]`` counts the spikes
``ceil(t_i*sampleRate)+1 .. ceil(t_(i+1)*sampleRate)`` samples back
(History.m:283-285, sampleRate = 1/delta; the product, not ``t/delta``, is what
MATLAB rounds) -- so a one-bin first window was always empty and no history
case could match MATLAB.  It now follows MATLAB's
rule; each case saves the tensor the MATLAB function consumed (``HkAll``) and
``test_compute_history_terms_matches_matlab_history`` checks it exactly, so the
filters below run end to end from ``windowTimes``.

Tolerance: measured on macOS arm64 / Accelerate vs MATLAB R2025b, the largest
absolute errors are 1.1e-15 (x_p / x_u, Numba path; 6.7e-16 pure Python),
1.9e-16 (W_p / W_u), 3.3e-16 (smoother) and 1.6e-15 / 1.9e-16 (PP_EStep
x_K / W_K; 2.2e-16 and 3.9e-16 for the N == C and C == 1 cases); the history
tensors match exactly.  ``rtol=1e-10, atol=1e-12`` (as in
``test_pp_estep_matlab_gold.py``) is >= ~600x above the worst error.  Before
the fixes the square ``PPDecodeFilterLinear`` / ``PP_EStep`` cases were off by
1.6e-1 to 3.4e-1 even on MATLAB's history tensor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import nstat.decoding_algorithms as da
from nstat.decoding_algorithms import DecodingAlgorithms
from tests._optional import probe_optional

FIXTURE = Path(__file__).resolve().parent / "parity" / "fixtures" / "matlab_gold" / "pp_square_history.mat"
RTOL = 1e-10
ATOL = 1e-12

# case -> (MATLAB function, fitType, N, nW, C)
CASES = {
    "pdfl_pois_sq": ("PPDecodeFilterLinear", "poisson", 150, 3, 3),
    "pdfl_binom_sq": ("PPDecodeFilterLinear", "binomial", 150, 4, 4),
    "pfis_pois_sq": ("PP_fixedIntervalSmoother", "poisson", 150, 3, 3),
    "pdfl_pois_ctrl": ("PPDecodeFilterLinear", "poisson", 150, 2, 3),
    "estep_pois_sq": ("PP_EStep", "poisson", 150, 3, 3),
    "estep_pois_NeqC": ("PP_EStep", "poisson", 6, 2, 6),
    "estep_binom_C1": ("PP_EStep", "binomial", 150, 2, 1),
    "pdfl_pois_offgrid": ("PPDecodeFilterLinear", "poisson", 150, 3, 3),
    "pdfl_pois_colon": ("PPDecodeFilterLinear", "poisson", 150, 10, 3),
    "pdfl_binom_delta2": ("PPDecodeFilterLinear", "binomial", 150, 3, 3),
}
FILTER_CASES = [case for case, spec in CASES.items() if spec[0] == "PPDecodeFilterLinear"]

_NUMBA_PROBE = probe_optional("numba")
_NUMBA_SKIP_REASON = "numba unavailable: " + _NUMBA_PROBE.reason.removeprefix("numba ")
PATHS = [
    pytest.param(True, marks=pytest.mark.skipif(not _NUMBA_PROBE.available, reason=_NUMBA_SKIP_REASON), id="numba"),
    pytest.param(False, id="pure-python"),
]


@pytest.fixture(scope="module")
def gold() -> dict:
    return loadmat(FIXTURE, squeeze_me=False, struct_as_record=False)


def _case(gold: dict, case: str) -> dict:
    """Inputs / outputs of one case, keyed by the MATLAB field names."""
    prefix = f"{case}_"
    out = {}
    for key, value in gold.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name in ("fitType", "func"):
            out[name] = str(np.asarray(value).reshape(-1)[0])
        else:
            out[name] = np.asarray(value, dtype=float)
    out["delta"] = float(out["delta"].reshape(-1)[0])
    out["windowTimes"] = out["windowTimes"].reshape(-1)
    return out


def _run_filter(cs: dict):
    return DecodingAlgorithms.PPDecodeFilterLinear(
        cs["A"], cs["Q"], cs["dN"], cs["mu"], cs["beta"], cs["fitType"], cs["delta"],
        cs["gamma"], cs["windowTimes"], cs["x0"], cs["Pi0"],
    )[:4]


def _run_smoother(cs: dict):
    return DecodingAlgorithms.PP_fixedIntervalSmoother(
        cs["A"], cs["Q"], cs["dN"], int(cs["lags"].reshape(-1)[0]), cs["mu"], cs["beta"],
        cs["fitType"], cs["delta"], cs["gamma"], cs["windowTimes"], cs["x0"], cs["Pi0"],
    )


def _assert_matches(actual, expected, label: str) -> None:
    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape, label
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=label)


def test_fixture_cases_are_the_documented_ones(gold) -> None:
    assert [str(np.asarray(c).reshape(-1)[0]) for c in gold["case_names"].reshape(-1)] == list(CASES)
    for case, (func, fit_type, N, nW, C) in CASES.items():
        cs = _case(gold, case)
        assert cs["func"] == func and cs["fitType"] == fit_type
        assert [int(v) for v in cs["sizes"].reshape(-1)] == [N, nW, C, 2]
        assert cs["dN"].shape == (C, N) and cs["windowTimes"].size == nW + 1
        assert cs["gamma"].shape == (nW, C) and np.all(cs["gamma"] < 0)
        if func == "PPDecodeFilterLinear":
            # MATLAB PPDecodeFilterLinear transposes a square (dx == C) beta.
            assert cs["beta"].shape == (2, C) and C != 2
        if nW == C:
            # Non-symmetric, so a window/cell transpose cannot pass.
            assert np.max(np.abs(cs["gamma"] - cs["gamma"].T)) > 0.1
        # MATLAB stores N x nW x C, dropping the trailing singleton when C == 1.
        assert cs["HkAll"].shape == ((N, nW, C) if C > 1 else (N, nW))
        assert np.any(cs["HkAll"] != 0)


@pytest.mark.parametrize("case", list(CASES))
def test_compute_history_terms_matches_matlab_history(gold, case) -> None:
    # The filters' history tensor equals MATLAB's History.computeHistory one
    # (also the PP_EM-style tensors of the PP_EStep cases) element for element.
    cs = _case(gold, case)
    N, nW, C, _ = (int(v) for v in cs["sizes"].reshape(-1))
    hk = da._compute_history_terms(cs["dN"], cs["delta"], cs["windowTimes"])
    assert hk.shape == (N, nW, C)
    assert np.array_equal(hk, cs["HkAll"].reshape(N, nW, C))


def test_compute_history_terms_counts_only_unit_bins() -> None:
    # MATLAB builds the spike train from find(dN(c,:)==1): a bin holding 2 is
    # not a spike there, so it adds nothing to the history.
    dN = np.array([[1.0, 2.0, 0.0, 1.0, 0.0]])
    hk = da._compute_history_terms(dN, 0.001, [0.0, 0.001, 0.003])
    assert hk[:, :, 0].tolist() == [[0, 0], [1, 0], [0, 1], [0, 1], [1, 0]]


@pytest.mark.parametrize("force_numba", PATHS)
@pytest.mark.parametrize("case", FILTER_CASES)
def test_pp_decode_filter_linear_matches_matlab_gold_at_every_step(gold, case, force_numba, monkeypatch) -> None:
    if not force_numba:
        monkeypatch.setattr("nstat.extras._numba_kernels._NUMBA_AVAILABLE", False)
    cs = _case(gold, case)
    for name, actual in zip(("x_p", "W_p", "x_u", "W_u"), _run_filter(cs)):
        _assert_matches(actual, cs[name], f"{case} {name}")


def test_pp_fixed_interval_smoother_matches_matlab_gold(gold) -> None:
    cs = _case(gold, "pfis_pois_sq")
    x_pLag, W_pLag, x_uLag, W_uLag = _run_smoother(cs)
    _assert_matches(x_uLag, cs["x_uLag"], "x_uLag")
    _assert_matches(W_uLag, cs["W_uLag"], "W_uLag")
    _assert_matches(x_pLag, cs["x_pLag"], "x_pLag")
    _assert_matches(W_pLag, cs["W_pLag"], "W_pLag")


def test_pp_fixed_interval_smoother_lag1_first_prediction_column(gold) -> None:
    # MATLAB runs its output block on the first step too (x_K = 0 there), and
    # with lags == 1 that sets x_pLag(:,2) = x_u(:,1), W_pLag(:,:,2) = W_u(:,:,1).
    cs = _case(gold, "pfis_pois_sq")
    x_pLag, W_pLag, _, _ = _run_smoother(cs)
    assert np.any(cs["x_pLag"][:, 1] != 0)
    _assert_matches(np.asarray(x_pLag)[:, 1], cs["x_pLag"][:, 1], "x_pLag[:, 1]")
    _assert_matches(np.asarray(W_pLag)[:, :, 1], cs["W_pLag"][:, :, 1], "W_pLag[:, :, 1]")


@pytest.mark.parametrize("case", ["estep_pois_sq", "estep_pois_NeqC", "estep_binom_C1"])
def test_pp_estep_matches_matlab_gold_at_every_step(gold, case) -> None:
    cs = _case(gold, case)
    N, nW, C, dx = (int(v) for v in cs["sizes"].reshape(-1))
    x_K, W_K, _, _ = DecodingAlgorithms.PP_EStep(
        cs["A"], cs["Q"], cs["dN"], cs["mu"], cs["beta"], cs["fitType"], cs["gamma"],
        cs["HkAll"], cs["x0"], cs["Px0"],
    )
    assert cs["x_K"].shape == (dx, N) and cs["W_K"].shape == (dx, dx, N)
    _assert_matches(x_K, cs["x_K"], f"{case} x_K")
    _assert_matches(W_K, cs["W_K"], f"{case} W_K")


def test_pp_estep_one_cell_2d_history_equals_explicit_3d(gold) -> None:
    # Every output -- including logll and the sufficient statistics, which
    # PP_EStep computes from HkAll itself -- is identical for MATLAB's
    # N x nW one-cell history and the explicit (N, nW, 1) tensor.
    cs = _case(gold, "estep_binom_C1")
    N, nW, C, _ = (int(v) for v in cs["sizes"].reshape(-1))
    assert C == 1 and cs["HkAll"].shape == (N, nW)
    args = (cs["A"], cs["Q"], cs["dN"], cs["mu"], cs["beta"], cs["fitType"], cs["gamma"])
    x_K2, W_K2, ll2, es2 = DecodingAlgorithms.PP_EStep(*args, cs["HkAll"], cs["x0"], cs["Px0"])
    x_K3, W_K3, ll3, es3 = DecodingAlgorithms.PP_EStep(*args, cs["HkAll"].reshape(N, nW, 1), cs["x0"], cs["Px0"])
    assert np.array_equal(x_K2, x_K3) and np.array_equal(W_K2, W_K3) and ll2 == ll3
    assert sorted(es2) == sorted(es3)
    for key in es2:
        assert np.array_equal(np.asarray(es2[key]), np.asarray(es3[key])), key


# ---------------------------------------------------------------------------
# _normalize_history_tensor layout resolution
# ---------------------------------------------------------------------------


def test_normalize_history_tensor_keeps_canonical_layout_when_n_equals_c() -> None:
    hk = np.random.default_rng(0).normal(size=(6, 2, 6))  # N == C == 6, nW = 2
    assert np.array_equal(da._normalize_history_tensor(hk, 6, 2, 6), hk)


def test_normalize_history_tensor_accepts_2d_history_only_for_one_cell() -> None:
    hk = np.random.default_rng(1).normal(size=(7, 2))
    out = da._normalize_history_tensor(hk, 7, 2, 1)
    assert out.shape == (7, 2, 1) and np.array_equal(out[:, :, 0], hk)
    with pytest.raises(ValueError, match="HkAll must align"):
        da._normalize_history_tensor(np.zeros((7, 2)), 7, 2, 3)


def test_normalize_history_tensor_other_layouts_resolve_as_before() -> None:
    N, nW, C = 7, 2, 3
    hk = np.random.default_rng(2).normal(size=(N, nW, C))
    for permuted in (np.transpose(hk, (2, 0, 1)), np.transpose(hk, (2, 1, 0)), np.transpose(hk, (1, 2, 0))):
        # (C, N, nW), (C, nW, N) and MATLAB's permute(HkAll,[2 3 1]) = (nW, C, N).
        assert np.array_equal(da._normalize_history_tensor(permuted, N, nW, C), hk)
    # The helper maps each of these back given the true numWindows.  The public
    # update steps infer numWindows from axis 1, so through them a permuted
    # layout only resolves when that inference happens to be right.


def test_normalize_history_tensor_rejects_ambiguous_square_layout() -> None:
    # (C, C, N) with nW == C fits both MATLAB's permute(HkAll,[2 3 1]) =
    # (nW, C, N) and (C, nW, N); MATLAB's PPAF and PPLFP updates read such a
    # square slice differently, so it is refused rather than guessed.
    sq = np.random.default_rng(3).normal(size=(3, 3, 7))
    with pytest.raises(ValueError, match="ambiguous"):
        da._normalize_history_tensor(sq, 7, 3, 3)
    with pytest.raises(ValueError, match="only the canonical|Only the canonical"):
        DecodingAlgorithms.PPDecode_updateLinear(
            np.zeros(2), 0.1 * np.eye(2), np.zeros((3, 7)), -np.ones(3), np.ones((2, 3)),
            "poisson", -0.5 * np.ones((3, 3)), sq, 2,
        )


def test_compute_history_terms_rejects_windows_reaching_the_current_bin() -> None:
    # MATLAB History indexes b(ceil(t*sampleRate)+1 : ...), which fails for an
    # edge at or before -delta; Python would otherwise sum current/future bins.
    dN = np.array([[1.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="history windows must lie"):
        da._compute_history_terms(dN, 0.001, [-0.001, 0.001])
    # An edge in (-delta, 0] still starts at one bin back, as in MATLAB.
    assert np.array_equal(
        da._compute_history_terms(dN, 0.001, [-0.0005, 0.001]),
        da._compute_history_terms(dN, 0.001, [0.0, 0.001]),
    )


def test_matlab_colon_exact_matches_matlab_bitwise(gold) -> None:
    from nstat.core import _matlab_colon_exact

    a, d, b = (gold[f"colon_{k}"].reshape(-1).astype(float) for k in "adb")
    expected = gold["colon_v"].reshape(-1)
    assert a.size == d.size == b.size == expected.size == 487
    for i in range(a.size):
        want = np.asarray(expected[i], dtype=float).reshape(-1)
        got = _matlab_colon_exact(a[i], d[i], b[i])
        assert got.shape == want.shape and np.array_equal(got, want), (i, a[i], d[i], b[i])


def test_pplfp_em_default_history_matches_matlab(gold, monkeypatch) -> None:
    """PPLFP_EM with windowTimes omitted builds MATLAB's default history exactly.

    MATLAB: windowTimes = 0:delta:(length(gamma)+1)*delta (PPLFP.m:1613), then
    History.computeHistory per cell.  The E-step is stubbed to capture the
    HkAll that PPLFP_EM hands it and stop there.
    """
    from nstat.decoding.PPLFP import PPLFP

    class _Captured(Exception):
        pass

    seen: dict = {}

    def _capture_estep(*args, **_kwargs):
        seen["HkAll"] = np.asarray(args[12], dtype=float)
        raise _Captured

    monkeypatch.setattr(PPLFP, "PPLFP_EStep", staticmethod(_capture_estep))
    dN = gold["emdef_dN"].astype(float)
    gamma = gold["emdef_gamma"].astype(float)
    delta = float(gold["emdef_delta"].reshape(-1)[0])
    C, N = dN.shape
    assert gamma.shape == (8, 2) and C == 2

    def _run(g):
        with pytest.raises(_Captured):
            PPLFP.PPLFP_EM(
                np.zeros((1, N)), dN, 0.99 * np.eye(2), 0.01 * np.eye(2), np.ones((1, 2)), np.eye(1),
                np.zeros(1), -2.0 * np.ones(C), 0.5 * np.ones((2, C)), "poisson", delta, g,
            )
        return seen.pop("HkAll")

    hk = _run(gamma)
    ml = gold["emdef_HkAll"].astype(float)
    assert hk.shape == ml.shape == (N, 9, C)  # length(gamma) + 1 = 9 windows
    assert np.array_equal(hk, ml)
    # A scalar-zero gamma means "no history" (MATLAB PPLFP.m FIX #98).
    assert np.array_equal(_run(0.0), np.zeros((N, 1, C)))

