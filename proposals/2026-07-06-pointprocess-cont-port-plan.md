# PointProcessSimulationCont.slx → Python port — implementation plan

> Dev-track plan (subagent-driven). Execution: superpowers:subagent-driven-development.
> Branch: `feat/cif-continuous-port`. Package version 0.6.0.

**Goal:** Add a native-Python continuous-time conditional-intensity (CIF)
point-process simulator, `nstat.extras.simulate_cif_continuous`, that reproduces
the deterministic λ output of the MATLAB `PointProcessSimulationCont.slx` model,
and reclassify that model in the Simulink-fidelity ledger.

**Architecture:** Python-only `nstat.extras` feature (no MATLAB runtime backend).
The MATLAB model is realized natively: continuous LTI filters S/E/H via
`scipy.signal`, the same exp/logistic link + uniform-thinning + 1-step history
feedback as the already-ported discrete `CIF.simulateCIF`.

**Tech stack:** NumPy, `scipy.signal` (lsim, cont2discrete). No new runtime deps.

## Global Constraints (bind every task)

- **Home is `nstat.extras`, not core.** The Cont model has **zero callers** in
  MATLAB (`CIF.simulateCIF` dispatches only to `PointProcessSimulationThinning`
  and the discrete `PointProcessSimulation`, and rejects continuous transfer
  functions). It is therefore a Python-only enhancement, not a parity obligation.
  Putting a continuous simulator in core `CIF` would *violate* parity.
- **Reuse, don't reinvent.** Return the existing
  `nstat.simulators.PointProcessSimulation` dataclass. Mirror the step-loop +
  injected-uniform pattern of `nstat.simulators.simulate_two_neuron_network`.
- **No new runtime dependency.** `scipy` is already a core dep.
- **RNG:** `np.random.default_rng(seed)` only. Never legacy `np.random`.
- **Time in seconds, rates in Hz.** `from __future__ import annotations`.
- **Gold fixture is canonical:** `tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat`
  (already committed, Task 1). Do not regenerate/alter it.

## Verified model semantics (MATLAB 2025b, to 1e-13)

```
eta(t)  = mu + S*stim(t) + E*ens(t) + H*pp_delayed(t)     # S,E,H continuous LTI; pp_delayed = z^-1 * spike
lambdaDelta(t) = exp(eta)               if simType == 'poisson'   (simTypeSelect=1)
               = exp(eta)/(1+exp(eta))  if simType == 'binomial'  (simTypeSelect=0)
spike(t) = 1  iff  U(0,1) < lambdaDelta(t)
"lambdaDelta" output port = lambdaDelta / Ts = lambda (Hz)   # what the gold fixture stores
```

**Validation split (hard constraint, discovered empirically):**
- **H = 0** → λ(t) is deterministic in the inputs → **bit-close vs the gold fixture**.
- **H ≠ 0** → λ depends on the realised (stochastic) spikes → validate **Python-side
  with injected uniforms** (MATLAB DSP Random Source RNG is not reproducible in NumPy).

Gold fixture fields: `t` (1001,), `u`, `mu`, `Ts`, `Snum`, `Sden`, `eta` (1001,),
`lambda_poisson` (1000,), `lambda_binom` (1000,). **Alignment:** the model logs
output on `t[1:]` (drops t=0), so `lambda_*[k]` corresponds to `eta[k+1]`; a Python
λ trace computed on the full `t` grid matches the gold via `python_lambda[1:]`.

## File structure

- Create: `nstat/extras/continuous_cif.py` — `simulate_cif_continuous(...)`.
- Create: `tests/extras/test_continuous_cif.py` — unit + gold-fixture tests.
- Modify: `nstat/extras/__init__.py` — document the new module in the header list.
- Modify: `tests/test_api_surface.py` — add the new public symbol row.
- Modify: `AGENT_GUIDE.md` — add a usage recipe (helpfile-check gate).
- Modify: `docs/api.rst` (+ `docs/extras/…` if the pattern requires).
- Modify: `parity/simulink_fidelity.yml` — reclassify `PointProcessSimulationCont`.
- Modify: `tests/test_simulink_fidelity_audit.py` — update expected classification.
- Create: `examples/extras/continuous_cif_demo.py` (+ `examples/extras/manifest.yml`
  and `tools/extras_build/extras_descriptions.yml` entries).

## Public API

```python
def simulate_cif_continuous(
    mu: float,
    stim,                      # continuous LTI: (num, den) | scipy.signal.lti | array-like FIR
    ens,
    hist,
    input_stim,                # Covariate | (time, values) — stimulus time series
    input_ens,
    *,
    Ts: float,                 # spike-generation bin / random-source sample time (s)
    simType: str = "binomial", # {'binomial','poisson'}
    seed: int | None = None,
    uniform_values: np.ndarray | None = None,
    return_details: bool = False,
) -> PointProcessSimulation
```

## Algorithm (implementer spec)

1. Build inputs on the grid `t` (from `input_stim`), `dt = Ts`.
2. Precompute the **deterministic** drives (do not depend on spikes):
   `stim_drive = scipy.signal.lsim((Snum,Sden), u_stim, t)[1]`;
   `ens_drive` likewise for E. (A zero LTI → all-zeros drive.)
3. Discretize the **history** filter H to sample time `Ts` with
   `scipy.signal.cont2discrete((Hnum,Hden), Ts, method='zoh')` → `(Ad,Bd,Cd,Dd)`;
   carry state `x_h` across steps.
4. Step loop `k = 0..N-1` (sequential — history feedback):
   - `pp_delayed = spike[k-1]` (0 at `k=0`).
   - `h_effect = Cd @ x_h + Dd * pp_delayed`;  then `x_h = Ad @ x_h + Bd * pp_delayed`.
   - `eta = mu + stim_drive[k] + ens_drive[k] + h_effect`.
   - `lambda_delta = exp(eta)` (poisson) or `sigmoid(eta)` (binomial); clip eta to
     a safe range as `simulate_two_neuron_network` does.
   - `u_k = draws[k]` (from `uniform_values` or `rng.random(N)`).
   - `spike[k] = 1.0 if u_k < lambda_delta else 0.0`.
   - `rate_hz[k] = lambda_delta / Ts`.
5. Return `PointProcessSimulation(time=t, rate_hz=rate_hz, spikes=SpikeTrain(t[spike>0.5]),
   lambda_delta=lambda_delta_arr, spike_indicator=spike, uniform_values=draws)`.

Accept LTI systems as `(num, den)` tuples, `scipy.signal.lti`, or objects exposing
`.num`/`.den`; a plain array is treated as an FIR numerator over `den=[1]`.

## Tasks

- [x] **Task 1 — gold fixtures (DONE, commit d0c374e).** `tools/parity/matlab/capture_cif_cont.m`
  + `tests/parity/fixtures/matlab_gold/cif_cont_lambda.mat`.

- [ ] **Task 2 — core module + tests.** Implement `simulate_cif_continuous` and
  `tests/extras/test_continuous_cif.py`. Tests (TDD): (a) `lsim` filter of a known
  continuous S matches `scipy.signal.lsim` and the gold `eta - mu`; (b) **deterministic
  λ vs gold** for BOTH links (`python_lambda[1:]` vs `lambda_poisson`/`lambda_binom`,
  rtol≈1e-6); (c) injected-uniform history recursion reproduces an exact spike sequence
  with `H≠0`; (d) shape/`simType`/error-path checks; (e) mean-rate statistical sanity.

- [ ] **Task 3 — wiring + governance + demo.** Register the symbol (extras
  `__all__`/import path per repo convention), `tests/test_api_surface.py` row,
  `AGENT_GUIDE.md` recipe, `docs/api.rst`; reclassify `PointProcessSimulationCont`
  in `parity/simulink_fidelity.yml` (`reference_only` → `high_fidelity_native_python`,
  `python_equivalent: nstat.extras.simulate_cif_continuous`) and update
  `tests/test_simulink_fidelity_audit.py`; add `examples/extras/continuous_cif_demo.py`
  (ground-truth-vs-recovered figure) + manifest + description entries. Run
  `make test`, `make helpfile-check`, `make readme-check`, `make freshness-check`.

- [ ] **Final whole-branch review**, then finish the branch.

## Risks

- **R2 (residual):** continuous-vs-discrete output differs only for genuinely
  continuous S/H/E; document the feature's value (s-domain synaptic/recovery kernels
  at fine resolution) honestly in the docstring + demo.
- **History discretization:** the `cont2discrete(zoh)` stepping of H approximates the
  continuous filter's response to the ZOH-held delayed spike train; document as a
  faithful discretization, validated by the injected-uniform test rather than vs MATLAB.
