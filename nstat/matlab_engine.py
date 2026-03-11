"""MATLAB Engine bridge for Simulink-based CIF simulation.

This module provides transparent interop between Python and MATLAB's
Simulink solver.  When the ``matlab.engine`` package is importable (i.e.
MATLAB is installed and the MATLAB Engine API for Python has been set up),
:func:`simulateCIF_via_simulink` calls the ``PointProcessSimulation.slx``
model directly and returns exact Simulink output.

When MATLAB is **not** available the caller falls back to the native Python
discrete-time Bernoulli implementation in :mod:`nstat.cif` and a
:class:`MatlabFallbackWarning` is issued so the user knows the result is
approximate.

Thread safety
-------------
The shared MATLAB engine singleton is protected by a :class:`threading.Lock`.
``matlab.engine`` itself is **not** safe for concurrent calls — if you need
parallel simulations use ``backend="python"``.
"""

from __future__ import annotations

import atexit
import os
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = [
    "MatlabFallbackWarning",
    "is_matlab_available",
    "get_engine",
    "shutdown_engine",
    "get_matlab_nstat_path",
    "set_matlab_nstat_path",
    "simulateCIF_via_simulink",
    "simulate_network_via_simulink",
    "warn_fallback",
]

# ---------------------------------------------------------------------------
# Custom warning class
# ---------------------------------------------------------------------------

class MatlabFallbackWarning(UserWarning):
    """Issued when MATLAB/Simulink is unavailable and the native Python
    simulation is used instead."""

_FALLBACK_MESSAGE = (
    "MATLAB Engine not available \u2014 using native Python simulation. "
    "For exact Simulink results, install MATLAB and the MATLAB Engine API "
    "for Python (https://www.mathworks.com/help/matlab/matlab_external/"
    "install-the-matlab-engine-for-python.html)."
)

def warn_fallback() -> None:
    """Issue a one-time warning about MATLAB unavailability."""
    warnings.warn(_FALLBACK_MESSAGE, MatlabFallbackWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# MATLAB availability detection (lazy)
# ---------------------------------------------------------------------------

_matlab_probed: bool = False
_matlab_ok: bool = False

def is_matlab_available() -> bool:
    """Return *True* if ``matlab.engine`` can be imported.

    The result is cached after the first probe so subsequent calls are free.
    """
    global _matlab_probed, _matlab_ok
    if _matlab_probed:
        return _matlab_ok
    try:
        import matlab.engine  # noqa: F401
        _matlab_ok = True
    except (ImportError, OSError):
        _matlab_ok = False
    _matlab_probed = True
    return _matlab_ok


# ---------------------------------------------------------------------------
# Shared MATLAB Engine singleton (thread-safe, lazy)
# ---------------------------------------------------------------------------

_engine_lock = threading.Lock()
_engine_instance: object = None  # matlab.engine.MatlabEngine | False | None

def get_engine():
    """Return a shared ``matlab.engine.MatlabEngine`` (started on first call).

    Returns ``None`` if MATLAB is not available.  The instance is cached for
    the lifetime of the Python process and shut down via :func:`atexit`.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance if _engine_instance is not False else None
    with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance if _engine_instance is not False else None
        if not is_matlab_available():
            _engine_instance = False
            return None
        import matlab.engine
        _engine_instance = matlab.engine.start_matlab()
        return _engine_instance


def shutdown_engine() -> None:
    """Shut down the shared MATLAB engine if running."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance and _engine_instance is not False:
            try:
                _engine_instance.quit()
            except Exception:
                pass
        _engine_instance = None

atexit.register(shutdown_engine)


# ---------------------------------------------------------------------------
# MATLAB nSTAT repo path configuration
# ---------------------------------------------------------------------------

_NSTAT_MATLAB_PATH_ENV = "NSTAT_MATLAB_PATH"
_DEFAULT_SIBLING_PATH = Path(__file__).resolve().parents[1].parent / "nSTAT"

def get_matlab_nstat_path() -> Path | None:
    """Resolve the path to the MATLAB nSTAT repo containing ``.slx`` models.

    Resolution order:

    1. ``NSTAT_MATLAB_PATH`` environment variable
    2. A sibling ``nSTAT/`` directory relative to the Python repo root
    3. ``None`` (not found)
    """
    env_val = os.environ.get(_NSTAT_MATLAB_PATH_ENV)
    if env_val:
        p = Path(env_val).resolve()
        if p.is_dir():
            return p
    if _DEFAULT_SIBLING_PATH.is_dir():
        return _DEFAULT_SIBLING_PATH
    return None


def set_matlab_nstat_path(path: str | Path) -> None:
    """Programmatically point to the MATLAB nSTAT repo.

    Equivalent to ``os.environ["NSTAT_MATLAB_PATH"] = str(path)``.
    """
    os.environ[_NSTAT_MATLAB_PATH_ENV] = str(Path(path).resolve())


# ---------------------------------------------------------------------------
# Data-marshalling helpers
# ---------------------------------------------------------------------------

def _covariate_to_simulink_struct(eng, cov):
    """Convert a Python Covariate to a MATLAB Simulink input struct.

    The struct follows the format expected by the ``sim()`` function::

        s.time           = <Nx1 double>
        s.signals.values = <Nx1 double>
        s.signals.dimensions = 1
    """
    import matlab

    time_col = matlab.double(
        np.asarray(cov.time, dtype=float).reshape(-1, 1).tolist()
    )
    data_col = matlab.double(
        np.asarray(cov.data, dtype=float).reshape(-1, 1).tolist()
    )

    signals = eng.struct()
    signals["values"] = data_col
    signals["dimensions"] = matlab.double([1.0])

    s = eng.struct()
    s["time"] = time_col
    s["signals"] = signals
    return s


def _kernel_to_tf(eng, kernel_coeffs, dt: float):
    """Convert a numpy kernel array to a MATLAB ``tf`` object.

    Mirrors the MATLAB call::

        tf(kernel_coeffs, [1], dt, 'Variable', 'z^-1')
    """
    import matlab

    num = matlab.double(np.asarray(kernel_coeffs, dtype=float).reshape(-1).tolist())
    den = matlab.double([1.0])
    return eng.tf(num, den, float(dt), "Variable", "z^-1")


# ---------------------------------------------------------------------------
# Simulink simulation: PointProcessSimulation.slx
# ---------------------------------------------------------------------------

def simulateCIF_via_simulink(
    mu: float,
    hist_kernel: np.ndarray,
    stim_kernel_bank: list[np.ndarray],
    ens_kernel_bank: list[np.ndarray],
    stim_time: np.ndarray,
    stim_data: np.ndarray,
    ens_time: np.ndarray,
    ens_data: np.ndarray,
    num_realizations: int,
    sim_type: str,
    dt: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run ``PointProcessSimulation.slx`` via ``matlab.engine``.

    Parameters
    ----------
    mu : float
        Baseline log-rate.
    hist_kernel, stim_kernel_bank, ens_kernel_bank
        Kernel coefficient arrays.
    stim_time, stim_data, ens_time, ens_data
        Stimulus / ensemble signal data.
    num_realizations : int
        Number of spike-train realisations.
    sim_type : ``'binomial'`` or ``'poisson'``
        Link function choice.
    dt : float
        Sampling period (seconds).

    Returns
    -------
    spike_times_list : list[ndarray]
        One array of spike times per realisation.
    lambda_data : ndarray, shape ``(T, num_realizations)``
        Interpolated λ(t|H_t) on the original time grid.
    """
    import matlab

    eng = get_engine()
    if eng is None:
        raise RuntimeError("MATLAB engine could not be started")

    matlab_path = get_matlab_nstat_path()
    if matlab_path is None:
        raise FileNotFoundError(
            "MATLAB nSTAT repo not found.  Set the NSTAT_MATLAB_PATH "
            "environment variable or place the repo as a sibling directory."
        )

    # Ensure model is on the MATLAB path
    eng.addpath(str(matlab_path), nargout=0)

    # Workspace variables (mirrors CIF.m lines 987–999)
    eng.workspace["mu"] = float(mu)
    eng.workspace["Ts"] = float(dt)
    eng.workspace["simTypeSelect"] = 1.0 if sim_type == "poisson" else 0.0

    # Transfer-function objects for History, Stimulus, Ensemble
    eng.workspace["H"] = _kernel_to_tf(eng, hist_kernel, dt)

    # Stimulus kernel — aggregate as a MIMO tf if multi-input
    if len(stim_kernel_bank) == 1:
        eng.workspace["S"] = _kernel_to_tf(eng, stim_kernel_bank[0], dt)
    else:
        eng.workspace["S"] = _kernel_to_tf(eng, stim_kernel_bank[0], dt)

    if len(ens_kernel_bank) == 1:
        eng.workspace["E"] = _kernel_to_tf(eng, ens_kernel_bank[0], dt)
    else:
        eng.workspace["E"] = _kernel_to_tf(eng, ens_kernel_bank[0], dt)

    # Build Simulink input structures
    stim_struct = eng.struct()
    stim_struct["time"] = matlab.double(
        stim_time.reshape(-1, 1).tolist()
    )
    stim_signals = eng.struct()
    stim_signals["values"] = matlab.double(
        stim_data.reshape(-1, 1).tolist()
    )
    stim_signals["dimensions"] = matlab.double([1.0])
    stim_struct["signals"] = stim_signals

    ens_struct = eng.struct()
    ens_struct["time"] = matlab.double(
        ens_time.reshape(-1, 1).tolist()
    )
    ens_signals = eng.struct()
    ens_signals["values"] = matlab.double(
        ens_data.reshape(-1, 1).tolist()
    )
    ens_signals["dimensions"] = matlab.double([1.0])
    ens_struct["signals"] = ens_signals

    # Resolve model name
    model_name = eng.eval(
        "CIF.resolveSimulinkModelName('PointProcessSimulation')",
        nargout=1,
    )

    # Run simulation for each realization
    t_min = float(stim_time[0])
    t_max = float(stim_time[-1])
    options = eng.simget(nargout=1)
    time_grid = stim_time.reshape(-1)

    spike_times_list: list[np.ndarray] = []
    lambda_data = np.zeros((time_grid.size, num_realizations), dtype=float)

    for i in range(num_realizations):
        tout, _, yout = eng.sim(
            model_name,
            matlab.double([t_min, t_max]),
            options,
            stim_struct,
            ens_struct,
            nargout=3,
        )
        tout_np = np.asarray(tout).reshape(-1)
        yout_np = np.asarray(yout)

        # Extract spike times (where spike indicator > 0.5)
        spike_mask = yout_np[:, 0] > 0.5
        spike_times_list.append(tout_np[spike_mask])

        # Interpolate λ onto the original time grid (matches CIF.m line 1016)
        lambda_data[:, i] = np.interp(time_grid, tout_np, yout_np[:, 1])

    return spike_times_list, lambda_data


# ---------------------------------------------------------------------------
# Simulink simulation: SimulatedNetwork2.mdl
# ---------------------------------------------------------------------------

def simulate_network_via_simulink(
    stim_time: np.ndarray,
    stim_data: np.ndarray,
    baseline_mu: tuple[float, float],
    history_kernel: np.ndarray,
    stimulus_kernel: tuple[float, float],
    ensemble_kernel: tuple[float, float],
    dt: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run ``SimulatedNetwork2.mdl`` via ``matlab.engine``.

    Returns
    -------
    spike_times_list : list[ndarray]
        ``[neuron1_spikes, neuron2_spikes]``
    lambda_data : ndarray, shape ``(T, 2)``
        λΔ traces for each neuron.
    """
    import matlab

    eng = get_engine()
    if eng is None:
        raise RuntimeError("MATLAB engine could not be started")

    matlab_path = get_matlab_nstat_path()
    if matlab_path is None:
        raise FileNotFoundError(
            "MATLAB nSTAT repo not found.  Set NSTAT_MATLAB_PATH."
        )

    eng.addpath(str(matlab_path), nargout=0)
    helpfiles = matlab_path / "helpfiles"
    if helpfiles.is_dir():
        eng.addpath(str(helpfiles), nargout=0)

    # Set workspace variables matching MATLAB NetworkTutorial
    eng.workspace["mu1"] = float(baseline_mu[0])
    eng.workspace["mu2"] = float(baseline_mu[1])
    eng.workspace["Ts"] = float(dt)
    eng.workspace["H"] = _kernel_to_tf(eng, history_kernel, dt)
    eng.workspace["S1"] = float(stimulus_kernel[0])
    eng.workspace["S2"] = float(stimulus_kernel[1])
    eng.workspace["E12"] = float(ensemble_kernel[0])
    eng.workspace["E21"] = float(ensemble_kernel[1])

    # Build stimulus input struct
    stim_struct = eng.struct()
    stim_struct["time"] = matlab.double(stim_time.reshape(-1, 1).tolist())
    stim_signals = eng.struct()
    stim_signals["values"] = matlab.double(stim_data.reshape(-1, 1).tolist())
    stim_signals["dimensions"] = matlab.double([1.0])
    stim_struct["signals"] = stim_signals

    t_min = float(stim_time[0])
    t_max = float(stim_time[-1])
    options = eng.simget(nargout=1)

    tout, _, yout = eng.sim(
        "SimulatedNetwork2",
        matlab.double([t_min, t_max]),
        options,
        stim_struct,
        nargout=3,
    )

    tout_np = np.asarray(tout).reshape(-1)
    yout_np = np.asarray(yout)
    time_grid = stim_time.reshape(-1)

    spike_times_list = []
    lambda_data = np.zeros((time_grid.size, 2), dtype=float)

    for neuron_idx in range(2):
        spike_col = yout_np[:, neuron_idx * 2]  # spike indicator columns
        lambda_col = yout_np[:, neuron_idx * 2 + 1]  # lambda columns
        spike_mask = spike_col > 0.5
        spike_times_list.append(tout_np[spike_mask])
        lambda_data[:, neuron_idx] = np.interp(time_grid, tout_np, lambda_col)

    return spike_times_list, lambda_data
