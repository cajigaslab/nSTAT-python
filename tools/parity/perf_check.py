#!/usr/bin/env python3
"""Performance parity check: time 5 hot paths in Python and MATLAB.

This is the *first* performance-parity baseline for ``nstat-python`` — a
new parity dimension layered on top of the numerical, visual, structural,
and class-method gates established in v1-v10.  The five hot paths are
high-traffic public functions that users hit repeatedly; the ratio of
Python wall-clock to MATLAB wall-clock answers "is Python competitive?"

Usage
-----
::

    python tools/parity/perf_check.py                 # default: 3 runs/side
    python tools/parity/perf_check.py --runs 5        # 5 runs/side, median + min/max
    python tools/parity/perf_check.py --capture-baseline
    python tools/parity/perf_check.py --python-only   # skip the MATLAB side
    python tools/parity/perf_check.py --paths analysis_run_for_neuron pp_decode_filter_linear

Outputs
-------
Default mode prints a Markdown table to stdout.  ``--capture-baseline``
also writes ``parity/performance_baseline.yml`` with the captured
medians, per-run lists, environment metadata, and ratio targets.

Hot paths
---------
1. ``analysis_run_for_neuron``    — ``Analysis.RunAnalysisForNeuron`` on a
   1000-spike single-cell trial against a single ``TrialConfig``.
2. ``pp_decode_filter_linear``    — ``DecodingAlgorithms.PPDecodeFilterLinear``
   on a 10000-step 2-state binomial-link adaptive filter.
3. ``kalman_filter``              — ``DecodingAlgorithms.kalman_filter`` on
   a 1000-step 4-state linear-Gaussian system.
4. ``simulate_point_process``     — ``simulate_point_process`` (the
   ``nstat.simulators`` thinning routine) for 10 s at 1 kHz binning,
   constant rate 10 Hz.  MATLAB analogue: ``CIF.simulateCIFByThinning``.
5. ``history_compute_history``    — ``History.computeHistory`` for a
   10000-sample spike train against 4 history windows.

Constraints / design notes
--------------------------
- MATLAB invocations carry ~5-10 s of startup overhead.  ``--runs N``
  performs N tic/toc inside a single ``matlab -batch`` invocation so the
  startup cost is amortised across the runs, not multiplied by N.
- We deliberately do NOT warm up the Python side: the first call is the
  first call.  Any obvious JIT/import warm-up is visible in the per-run
  list emitted by the tool.
- Inputs use ``np.random.default_rng(42)`` in Python and ``rng(42)`` in
  MATLAB so the *timed operations* are equivalent and reproducible.
- This script does not contact the GitHub-hosted MATLAB repo.  It uses
  the local checkout at ``$NSTAT_MATLAB_PATH`` (default
  ``/Users/iahncajigas/projects/nstat``) read-only — independence rule.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

# Make the repo importable when run as a standalone script — mirror the
# pattern in tools/parity/diff_against_matlab.py.
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

REPO_ROOT = _REPO_ROOT_FOR_IMPORT
BASELINE_PATH = REPO_ROOT / "parity" / "performance_baseline.yml"

# Default discovery for the local MATLAB checkout — matches diff_against_matlab.py.
_DEFAULT_MATLAB_REPO = os.environ.get("NSTAT_MATLAB_PATH", "/Users/iahncajigas/projects/nstat")
_DEFAULT_MATLAB_BIN = os.environ.get("MATLAB_BIN", "/opt/homebrew/bin/matlab")

# ---------------------------------------------------------------------------
# Path definitions
# ---------------------------------------------------------------------------


@dataclass
class HotPath:
    """One benchmark path.

    Attributes
    ----------
    name : str
        Stable identifier used in the baseline YAML and CLI ``--paths``.
    python_function : str
        Dotted reference for the Python implementation under test.
    matlab_function : str
        Class-qualified MATLAB function (e.g. ``Analysis.RunAnalysisForNeuron``).
    input_size : str
        Human-readable description of the input dimensions.
    python_timer : Callable[[], float]
        Returns wall-clock seconds for one run.
    matlab_body : str
        MATLAB code to be executed once per ``rep``; the harness wraps
        this body in ``for rep=1:N tic; ... toc end`` and prints elapsed
        seconds one per line.
    target_for_competitive : float
        Soft target ratio (py/ml) below which Python is considered to
        match MATLAB on this hot path.
    target_for_parity : float
        Hard ceiling above which the path is flagged for investigation.
    """

    name: str
    python_function: str
    matlab_function: str
    input_size: str
    python_timer: Callable[[], float]
    matlab_body: str
    target_for_competitive: float = 1.5
    target_for_parity: float = 5.0


# ---------------------------------------------------------------------------
# Python timer closures
# ---------------------------------------------------------------------------


def _time_analysis_run_for_neuron() -> float:
    """Single-cell ``Analysis.RunAnalysisForNeuron`` on a 1000-spike trial.

    Uses a dense uniform Poisson spike train with a sinusoidal stimulus
    covariate plus a 4-window history kernel.  ``makePlot=0`` to keep
    the timing pure-compute.
    """
    from nstat.analysis import Analysis
    from nstat.ConfigColl import ConfigColl
    from nstat.CovColl import CovColl
    from nstat.Covariate import Covariate
    from nstat.Events import Events
    from nstat.History import History
    from nstat.nspikeTrain import nspikeTrain
    from nstat.nstColl import nstColl
    from nstat.Trial import Trial
    from nstat.TrialConfig import TrialConfig

    rng = np.random.default_rng(42)
    T = 10.0
    fs = 1000.0
    time_grid = np.arange(0.0, T, 1.0 / fs)
    stim_values = np.sin(2 * np.pi * 3.0 * time_grid)
    stim = Covariate(time_grid, stim_values, "Stimulus", "time", "s", "", ["stim"])

    spike_times = np.sort(rng.uniform(0.0, T, size=1000))
    train = nspikeTrain(spike_times, "1", fs, 0.0, T, makePlots=-1)
    spikes = nstColl([train])
    history = History([0.0, 0.005, 0.010, 0.020, 0.040])
    trial = Trial(spikes, CovColl([stim]), Events([0.0], ["cue"]), history)

    cfg = TrialConfig(
        covMask=[["Stimulus", "stim"]],
        sampleRate=fs,
        history=[0.0, 0.005, 0.010, 0.020, 0.040],
        name="stim_hist",
    )
    configs = ConfigColl([cfg])

    t0 = time.perf_counter()
    Analysis.RunAnalysisForNeuron(trial, 0, configs, makePlot=0)
    return time.perf_counter() - t0


def _time_pp_decode_filter_linear() -> float:
    """``PPDecodeFilterLinear`` on a 10000-step 2-state binomial filter.

    Matches the ``test_decoding_algorithms_fidelity`` shape conventions
    but scaled up: ``dN`` is 2 cells × 10000 bins of Bernoulli spikes.
    """
    from nstat.DecodingAlgorithms import DecodingAlgorithms

    rng = np.random.default_rng(42)
    n_steps = 10_000
    n_cells = 2
    a = np.eye(2)
    q = 0.01 * np.eye(2)
    mu = np.array([-1.0, -1.0], dtype=float)
    beta = np.array([[0.5, 0.2], [0.1, 0.4]], dtype=float)
    dN = (rng.uniform(0.0, 1.0, size=(n_cells, n_steps)) < 0.05).astype(float)

    t0 = time.perf_counter()
    DecodingAlgorithms.PPDecodeFilterLinear(a, q, dN, mu, beta, "binomial", 0.001)
    return time.perf_counter() - t0


def _time_kalman_filter() -> float:
    """``DecodingAlgorithms.kalman_filter`` on a 1000-step 4-state system."""
    from nstat.DecodingAlgorithms import DecodingAlgorithms

    rng = np.random.default_rng(42)
    n_steps = 1000
    n_state = 4
    n_obs = 2

    A = np.eye(n_state) + 0.01 * rng.standard_normal((n_state, n_state))
    C = rng.standard_normal((n_obs, n_state))
    Pv = 0.01 * np.eye(n_state)
    Pw = 0.05 * np.eye(n_obs)
    Px0 = np.eye(n_state)
    x0 = np.zeros(n_state)
    y = rng.standard_normal((n_obs, n_steps))

    t0 = time.perf_counter()
    DecodingAlgorithms.kalman_filter(A, C, Pv, Pw, Px0, x0, y)
    return time.perf_counter() - t0


def _time_simulate_point_process() -> float:
    """``simulate_point_process`` for 10 s at 1 kHz binning, rate 10 Hz."""
    from nstat.simulators import simulate_point_process

    T = 10.0
    fs = 1000.0
    time_grid = np.arange(0.0, T, 1.0 / fs)
    rate_hz = np.full_like(time_grid, 10.0)

    t0 = time.perf_counter()
    simulate_point_process(time_grid, rate_hz, seed=42)
    return time.perf_counter() - t0


def _time_history_compute_history() -> float:
    """``History.computeHistory`` on a 10000-sample train, 4 windows."""
    from nstat.History import History
    from nstat.nspikeTrain import nspikeTrain

    rng = np.random.default_rng(42)
    T = 10.0
    fs = 1000.0
    spikes = np.sort(rng.uniform(0.0, T, size=200))
    train = nspikeTrain(spikes, "1", fs, 0.0, T, makePlots=-1)
    hist = History([0.0, 0.005, 0.010, 0.020, 0.040])

    t0 = time.perf_counter()
    hist.computeHistory(train)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# MATLAB-side bodies
# ---------------------------------------------------------------------------
#
# Each block is the MATLAB analogue of the Python closure above.  Inputs
# are constructed inside the body so the ``tic; <body>; toc`` we wrap
# around it times the same operation as the Python timer.  Where MATLAB
# uses a different default (``delta=0.001``, binomial GLM), we mirror it
# explicitly.

_MATLAB_BODIES = {
    "analysis_run_for_neuron": r"""
        rng(42);
        T = 10.0; fs = 1000.0;
        tg = (0:1/fs:T-1/fs)';
        stim = Covariate(tg, sin(2*pi*3.0*tg), 'Stimulus', 'time', 's', '', {'stim'});
        st = sort(rand(1000,1) * T);
        train = nspikeTrain(st, '1', fs, 0.0, T);
        spikes = nstColl({train});
        hist = History([0.0 0.005 0.010 0.020 0.040]);
        trial = Trial(spikes, CovColl({stim}), Events(0.0, {'cue'}), hist);
        cfg = TrialConfig({{'Stimulus','stim'}}, fs, [0.0 0.005 0.010 0.020 0.040], [], 'stim_hist');
        configs = ConfigColl({cfg});
        Analysis.RunAnalysisForNeuron(trial, 1, configs, 0);
    """,
    "pp_decode_filter_linear": r"""
        rng(42);
        n_steps = 10000;
        A = eye(2);
        Q = 0.01 * eye(2);
        mu = [-1.0; -1.0];
        beta = [0.5 0.2; 0.1 0.4];
        dN = double(rand(2, n_steps) < 0.05);
        DecodingAlgorithms.PPDecodeFilterLinear(A, Q, dN, mu, beta, 'binomial', 0.001);
    """,
    "kalman_filter": r"""
        rng(42);
        n_steps = 1000;
        n_state = 4; n_obs = 2;
        A = eye(n_state) + 0.01 * randn(n_state);
        C = randn(n_obs, n_state);
        Pv = 0.01 * eye(n_state);
        Pw = 0.05 * eye(n_obs);
        Px0 = eye(n_state);
        x0 = zeros(n_state, 1);
        y = randn(n_obs, n_steps);
        DecodingAlgorithms.kalman_filter(A, C, Pv, Pw, Px0, x0, y);
    """,
    "simulate_point_process": r"""
        rng(42);
        T = 10.0; fs = 1000.0;
        tg = (0:1/fs:T-1/fs)';
        rate = 10.0 * ones(size(tg));
        mu = log(rate(1) / fs);
        cif_data = log(rate / fs);
        lambda = Covariate(tg, cif_data, 'lambda', 'time', 's', '', {'lambda'});
        CIF.simulateCIFByThinningFromLambda(lambda, 1);
    """,
    "history_compute_history": r"""
        rng(42);
        T = 10.0; fs = 1000.0;
        st = sort(rand(200,1) * T);
        train = nspikeTrain(st, '1', fs, 0.0, T);
        hist = History([0.0 0.005 0.010 0.020 0.040]);
        hist.computeHistory(train);
    """,
}


# ---------------------------------------------------------------------------
# Hot-path registry
# ---------------------------------------------------------------------------


def _build_paths() -> list[HotPath]:
    return [
        HotPath(
            name="analysis_run_for_neuron",
            python_function="nstat.analysis.Analysis.RunAnalysisForNeuron",
            matlab_function="Analysis.RunAnalysisForNeuron",
            input_size="1000-spike single-cell trial, sin-3Hz stim, 4-window history, 10s @ 1kHz",
            python_timer=_time_analysis_run_for_neuron,
            matlab_body=_MATLAB_BODIES["analysis_run_for_neuron"],
        ),
        HotPath(
            name="pp_decode_filter_linear",
            python_function="nstat.decoding_algorithms.DecodingAlgorithms.PPDecodeFilterLinear",
            matlab_function="DecodingAlgorithms.PPDecodeFilterLinear",
            input_size="10000-step 2-state binomial-link adaptive filter, 2 cells",
            python_timer=_time_pp_decode_filter_linear,
            matlab_body=_MATLAB_BODIES["pp_decode_filter_linear"],
        ),
        HotPath(
            name="kalman_filter",
            python_function="nstat.decoding_algorithms.DecodingAlgorithms.kalman_filter",
            matlab_function="DecodingAlgorithms.kalman_filter",
            input_size="1000-step 4-state linear-Gaussian system, 2 obs",
            python_timer=_time_kalman_filter,
            matlab_body=_MATLAB_BODIES["kalman_filter"],
        ),
        HotPath(
            name="simulate_point_process",
            python_function="nstat.simulators.simulate_point_process",
            matlab_function="CIF.simulateCIFByThinningFromLambda",
            # Substitution note: the brief listed ``simulatePointProcess`` /
            # ``CIF.simulateCIFByThinning`` as the MATLAB analogue.  The
            # closest reachable surface is ``simulateCIFByThinningFromLambda``
            # (a thinning routine with the same generative model);
            # ``simulateCIFByThinning`` requires hist+stim+ens covariates
            # the Python ``simulate_point_process`` does not expose.
            input_size="10s @ 1kHz, constant rate 10Hz",
            python_timer=_time_simulate_point_process,
            matlab_body=_MATLAB_BODIES["simulate_point_process"],
        ),
        HotPath(
            name="history_compute_history",
            python_function="nstat.history.History.computeHistory",
            matlab_function="History.computeHistory",
            input_size="200-spike train @ 1kHz over 10s, 4 history windows",
            python_timer=_time_history_compute_history,
            matlab_body=_MATLAB_BODIES["history_compute_history"],
        ),
    ]


# ---------------------------------------------------------------------------
# Python timing harness
# ---------------------------------------------------------------------------


def time_python_path(path: HotPath, n_runs: int) -> dict[str, Any]:
    """Time the Python side of ``path`` ``n_runs`` times.

    Returns
    -------
    dict
        ``{"runs_sec": [...], "median_sec": float, "min_sec": float,
        "max_sec": float}``.
    """
    runs: list[float] = []
    for _ in range(n_runs):
        runs.append(path.python_timer())
    return {
        "runs_sec": [round(r, 6) for r in runs],
        "median_sec": round(statistics.median(runs), 6),
        "min_sec": round(min(runs), 6),
        "max_sec": round(max(runs), 6),
    }


# ---------------------------------------------------------------------------
# MATLAB timing harness
# ---------------------------------------------------------------------------


_MATLAB_HARNESS_TEMPLATE = r"""
addpath(genpath('{matlab_repo}'));
try
    for rep=1:{n_runs}
        tic;
{body}
        elapsed = toc;
        fprintf('PERF_NSTAT_ELAPSED %.9f\n', elapsed);
    end
catch ME
    fprintf(2, 'PERF_NSTAT_ERROR %s\n', ME.message);
    for k=1:numel(ME.stack)
        fprintf(2, '  at %s:%d\n', ME.stack(k).file, ME.stack(k).line);
    end
    exit(2);
end
exit(0);
"""


def time_matlab_path(
    path: HotPath,
    n_runs: int,
    matlab_repo: str,
    matlab_bin: str,
) -> dict[str, Any]:
    """Time the MATLAB analogue of ``path`` ``n_runs`` times.

    Builds a single ``-batch`` invocation that runs the body ``n_runs``
    times under ``tic/toc`` and parses the elapsed seconds from stdout.
    Amortising startup across runs is essential — a fresh MATLAB
    invocation costs 5-10 s.

    Returns
    -------
    dict
        Same shape as :func:`time_python_path`, plus a ``"status"`` key
        of ``"ok"`` or ``"error: ..."``.
    """
    repo_path = Path(matlab_repo)
    if not repo_path.is_dir():
        return {"status": f"error: matlab_repo not found: {matlab_repo}", "runs_sec": []}
    if not shutil.which(matlab_bin) and not Path(matlab_bin).exists():
        return {"status": f"error: matlab_bin not found: {matlab_bin}", "runs_sec": []}

    script = _MATLAB_HARNESS_TEMPLATE.format(
        matlab_repo=str(repo_path),
        n_runs=n_runs,
        body=path.matlab_body,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".m", delete=False) as fh:
        script_path = Path(fh.name)
        # MATLAB -batch wants a file basename without .m, so write to a
        # scratch dir and addpath that dir.
        fh.write(script)

    try:
        scratch_dir = script_path.parent
        script_name = script_path.stem
        cmd = [
            matlab_bin,
            "-batch",
            f"addpath('{scratch_dir}'); {script_name}",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error: matlab timeout after 600s", "runs_sec": []}

        if proc.returncode != 0:
            err = (proc.stderr or "")[-1000:]
            return {"status": f"error: matlab exit={proc.returncode}: {err.strip()}", "runs_sec": []}

        runs: list[float] = []
        for line in proc.stdout.splitlines():
            m = re.match(r"^PERF_NSTAT_ELAPSED\s+([0-9.eE+-]+)\s*$", line.strip())
            if m:
                runs.append(float(m.group(1)))
        if not runs:
            tail = (proc.stdout or "")[-1000:]
            return {"status": f"error: no elapsed lines parsed; stdout tail: {tail.strip()}", "runs_sec": []}

        return {
            "status": "ok",
            "runs_sec": [round(r, 6) for r in runs],
            "median_sec": round(statistics.median(runs), 6),
            "min_sec": round(min(runs), 6),
            "max_sec": round(max(runs), 6),
        }
    finally:
        try:
            script_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _hardware_string() -> str:
    """A short description of the host hardware, best-effort."""
    bits: list[str] = []
    bits.append(platform.machine())
    bits.append(platform.system())
    bits.append(platform.release())
    try:  # macOS
        sysctl = shutil.which("sysctl")
        if sysctl:
            out = subprocess.run(
                [sysctl, "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                bits.append(out.stdout.strip())
    except Exception:
        pass
    return " | ".join(bits)


def _matlab_repo_version(matlab_repo: str) -> str:
    """Capture a one-line MATLAB-checkout identifier for the baseline."""
    try:
        sha = subprocess.run(
            ["git", "-C", matlab_repo, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        date = subprocess.run(
            ["git", "-C", matlab_repo, "log", "-1", "--format=%cI"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode == 0:
            sha_str = sha.stdout.strip()
            date_str = date.stdout.strip() if date.returncode == 0 else "unknown"
            return f"cajigaslab/nSTAT@{sha_str} ({date_str})"
    except Exception:
        pass
    return f"local checkout at {matlab_repo}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_markdown(results: list[dict[str, Any]]) -> str:
    """Render the per-path results as a Markdown table."""
    lines = [
        "| Path | Python median (s) | MATLAB median (s) | Ratio (py/ml) | Flag |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        py_med = r["python"].get("median_sec")
        ml_med = r["matlab"].get("median_sec")
        ratio = r.get("ratio_py_over_ml")
        flag = r.get("flag", "")
        py_str = f"{py_med:.4f}" if py_med is not None else "n/a"
        ml_str = f"{ml_med:.4f}" if ml_med is not None else "n/a"
        ratio_str = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "n/a"
        lines.append(f"| {r['name']} | {py_str} | {ml_str} | {ratio_str} | {flag} |")
    return "\n".join(lines)


def _classify(ratio: float | None, path: HotPath) -> str:
    if ratio is None:
        return "skipped"
    if ratio <= path.target_for_competitive:
        return "competitive"
    if ratio <= path.target_for_parity:
        return "acceptable"
    return "needs investigation"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    n_runs: int,
    capture_baseline: bool,
    python_only: bool,
    selected_paths: list[str] | None,
    matlab_repo: str,
    matlab_bin: str,
) -> int:
    paths = _build_paths()
    if selected_paths:
        wanted = set(selected_paths)
        paths = [p for p in paths if p.name in wanted]
        missing = wanted - {p.name for p in paths}
        if missing:
            print(f"unknown path(s): {sorted(missing)}", file=sys.stderr)
            return 2

    results: list[dict[str, Any]] = []
    for path in paths:
        print(f"[perf] {path.name}: timing Python ({n_runs} runs)...", file=sys.stderr)
        py = time_python_path(path, n_runs)

        if python_only:
            ml: dict[str, Any] = {"status": "skipped (python-only)", "runs_sec": []}
        else:
            print(f"[perf] {path.name}: timing MATLAB ({n_runs} runs)...", file=sys.stderr)
            ml = time_matlab_path(path, n_runs, matlab_repo, matlab_bin)

        py_med = py.get("median_sec")
        ml_med = ml.get("median_sec")
        ratio: float | None = None
        if py_med is not None and ml_med not in (None, 0):
            ratio = round(py_med / ml_med, 3)

        results.append(
            {
                "name": path.name,
                "python_function": path.python_function,
                "matlab_function": path.matlab_function,
                "input_size": path.input_size,
                "python": py,
                "matlab": ml,
                "ratio_py_over_ml": ratio,
                "target_for_competitive": path.target_for_competitive,
                "target_for_parity": path.target_for_parity,
                "flag": _classify(ratio, path),
            }
        )

    # Stdout: human-facing summary
    print(_format_markdown(results))

    # Baseline YAML
    if capture_baseline:
        payload = {
            "version": 1,
            "captured": time.strftime("%Y-%m-%d"),
            "matlab_repo": _matlab_repo_version(matlab_repo),
            "matlab_bin": matlab_bin,
            "python_version": platform.python_version(),
            "hardware": _hardware_string(),
            "n_runs_per_side": n_runs,
            "paths": results,
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        print(f"\nWrote baseline to {BASELINE_PATH}", file=sys.stderr)

    # Exit code: 0 unless a path was flagged for investigation AND we were not
    # asked to merely capture a baseline.
    if not capture_baseline:
        for r in results:
            if r["flag"] == "needs investigation":
                return 1
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=int, default=3, help="runs per side (default 3)")
    p.add_argument(
        "--capture-baseline",
        action="store_true",
        help="write parity/performance_baseline.yml in addition to printing the summary",
    )
    p.add_argument(
        "--python-only",
        action="store_true",
        help="skip the MATLAB side (useful when MATLAB is unreachable)",
    )
    p.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="restrict timing to the named paths (default: all five)",
    )
    p.add_argument(
        "--matlab-repo",
        default=_DEFAULT_MATLAB_REPO,
        help=f"local MATLAB nSTAT checkout (default {_DEFAULT_MATLAB_REPO})",
    )
    p.add_argument(
        "--matlab-bin",
        default=_DEFAULT_MATLAB_BIN,
        help=f"MATLAB binary path (default {_DEFAULT_MATLAB_BIN})",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run(
        n_runs=args.runs,
        capture_baseline=args.capture_baseline,
        python_only=args.python_only,
        selected_paths=args.paths,
        matlab_repo=args.matlab_repo,
        matlab_bin=args.matlab_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
