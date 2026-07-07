# `nstat.extras.continuous_cif` — continuous-time CIF simulator

Native-Python port of the MATLAB Simulink model
`PointProcessSimulationCont.slx`: the stimulus and ensemble drives are
realised as *continuous* LTI filters (`S`, `E`) integrated with
`scipy.signal.lsim`, while the self-history feedback (`H`) is
discretized to the spike-generation sample time via
`scipy.signal.cont2discrete` (zero-order hold) and stepped once per bin.
This is a Python-only `nstat.extras` feature — the discrete
`CIF.simulateCIF` in core `nstat` has no continuous-transfer-function
path, so there is no MATLAB-parity obligation to keep this in `nstat.cif`.

## Install

No optional dependency — `continuous_cif` ships with the core
`nstat-toolbox` package. The module has zero runtime dependencies beyond
NumPy and SciPy, which `nstat-toolbox` already requires.

```bash
pip install nstat-toolbox
```

## Usage

```python
import numpy as np
from nstat.extras.continuous_cif import simulate_cif_continuous

Ts = 0.001
t = np.arange(5000) * Ts
u_stim = np.sin(2.0 * np.pi * 1.0 * t)   # slow "LFP" stimulus drive
u_ens = np.zeros_like(t)

# Continuous first-order lowpass stimulus filter S(s) = 1.5 / (0.05*s + 1)
stim_filter = (np.array([1.5]), np.array([0.05, 1.0]))
# Continuous inhibitory self-history filter H(s) = -4 / (0.01*s + 1)
hist_filter = (np.array([-4.0]), np.array([0.01, 1.0]))

result = simulate_cif_continuous(
    -4.0, stim_filter, 0.0, hist_filter, (t, u_stim), (t, u_ens),
    Ts=Ts, simType="binomial", seed=0,
)
result.spikes.spikeTimes      # realised spike times (SpikeTrain)
result.rate_hz                # lambda_delta / Ts, per-bin instantaneous rate
```

Pass `return_details=True` to also attach `eta`, `stim_drive`,
`ens_drive`, and `history_effect` diagnostic traces onto the returned
`PointProcessSimulation`.

See `examples/extras/continuous_cif_demo.py` for a runnable end-to-end
demo (continuous lowpass stimulus filtering + inhibitory continuous
self-history refractoriness), including a ground-truth-vs-recovered
check of the continuous filtering against closed-form linear-systems
theory.

## What it ports

`PointProcessSimulationCont.slx` is the continuous-time companion to the
discrete `PointProcessSimulation.slx` Simulink model — it has zero
MATLAB-side callers (no MATLAB helpfile or paper workflow invokes it),
so porting it is a Python-only `nstat.extras` enhancement, not a
core-parity obligation. `simulate_cif_continuous` follows the same
`eta = mu + S*stim + E*ens + H*pp_delayed` structure, with
`lambdaDelta = exp(eta)` (`simType='poisson'`) or `sigmoid(eta)`
(`simType='binomial'`), but realises `S`/`E` as continuous LTI systems
integrated across the full input grid rather than pre-sampled discrete
sequences.

## Validation strategy

The model's stochasticity only enters through the self-history feedback
term, so validation splits along that line:

- **`H = 0` (deterministic):** `eta` and `lambda_delta` do not depend on
  the realised spikes, so they are bit-close deterministic and are
  validated directly against the MATLAB gold fixture
  `tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat`.
- **`H != 0` (stochastic history feedback):** `lambda_delta` depends on
  the realised spike sequence, and MATLAB's DSP Random Source block is
  not reproducible in NumPy. This path is instead validated Python-side
  using injected `uniform_values` for exact reproducibility (matching
  the injected-uniform pattern used by
  `nstat.simulators.simulate_two_neuron_network`).

## Related

- `parity/simulink_fidelity.yml` — the `PointProcessSimulationCont` entry
  records this port's `native_python` / `high_fidelity_native_python`
  classification.
- `nstat.extras.matlab_rng` — a similar zero-optional-dependency
  `nstat.extras` module; see its help file for the same install/wiring
  pattern.
