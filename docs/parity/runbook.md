# Parity maintenance runbook

The parity push completed in 48 iterations across 10 bundled PRs. This
runbook covers ongoing maintenance, including post-upstream-merge
reconciliation cycles.

> **Status (v10 — 2026-06-19):** parity push complete; maintenance mode
> active. 9/10 priority topics at holistic `matches`, 36/36 numerical
> drift PASS at tightened tolerances, 27/27 gold fixtures PASS,
> all 9 filed MATLAB issues (`cajigaslab/nSTAT#78–#86`) adopted upstream.

## When something parity-affecting changes

If your PR touches `nstat/*.py`, `notebooks/*.ipynb`, or
`tools/parity/*.py`:

### Local checks (pre-commit hooks will run these automatically)

- `make parity-check-quick` — composite + SSIM (~30 s)
- `python tools/parity/numerical_drift.py` — algorithmic outputs
- `python tools/parity/class_method_parity.py --all` — class-method order
- `python tools/parity/code_structure_diff.py --all` — line-for-line parity
- `python tools/parity/helper_coverage.py` — % of notebooks using `matlab_*` helpers

### Manual remote dispatch

For a full sweep (~30 min including notebook execution):

```bash
gh workflow run parity-check.yml --ref main
```

## When drift appears

1. Check which gate failed in `.parity-review/SUMMARY_run_<UTC>.md`
2. Read the per-topic or per-class report under `.parity-review/`
3. Apply minimum-diff fix or add ledger entry

## Ledgers (single source of truth = YAML)

- `parity/matlab_defects.yml` — Python improvements over MATLAB
- `parity/matlab_pedagogical_gaps.yml` — Python pedagogical extras
- `parity/code_structure_exemptions.yml` — per-topic MATLAB-only idiom allowlist
- `parity/numerical_drift_spec.yml` — function-level numeric comparisons
- `parity/visual_fidelity.yml` — SSIM gates (regression net only)

Markdown copies auto-render via `tools/parity/render_ledger.py`.

## Adding new helpers

If a recurring pattern surfaces across notebooks:

1. Add to `nstat/notebook_figures.py` as `matlab_<name>()` or similar
2. Migrate at least 2 notebooks to use it
3. Update `tools/parity/helper_coverage.py` if needed (the tool auto-discovers)

## Documented residuals

After v10, 1 topic remains at holistic `minor`:

- `PPThinning` — sinusoidal lambda overlay on dense raster rows 3-4
  renders with slightly different visual weight than MATLAB. ISI
  histograms, axis labels, and figure counts all match. Documented in
  `parity/matlab_pedagogical_gaps.yml` as accepted renderer-level
  divergence.

All other priority topics (`nstCollExamples`, `StimulusDecode2D`,
`ExplicitStimulusWhiskerData`, `mEPSCAnalysis`, `nSTATPaperExamples`,
`HippocampalPlaceCellExample`, `SignalObjExamples`, `NetworkTutorial`,
`DecodingExample`) are at `matches`.

## Responding to a MATLAB upstream merge

When upstream `cajigaslab/nSTAT` merges fixes or enhancements (as
happened in v10 when our 9 filed issues were adopted), run a
reconciliation cycle:

### Step 1 — Re-capture gold fixtures

```bash
/opt/homebrew/bin/matlab -batch "addpath('tools/parity/matlab'); \
  export_matlab_gold_fixtures(pwd, '/path/to/local/nstat')"
```

This refreshes the 25 cohort-A fixtures. For the 26 ad-hoc fixtures
(4 PPLFP + 22 v9_*), the capture-script stubs at
`tools/parity/matlab/export_pplfp_gold_fixtures.m` and
`export_v9_gold_fixtures.m` have per-fixture TODO recipes that need
in-MATLAB seed/dim reconstruction before they're usable.

### Step 2 — Diff old vs new fixtures

Back up the pre-update fixtures, then load both and compare. The
v10 iter 44 diff harness lives at
`.parity-review/iter44_diff_fixtures.py` as a reference template.

### Step 3 — Classify each changed fixture

| Fixture diff | Action |
|---|---|
| Identical | No action |
| Numeric drift within current tolerance | No action; consider tightening |
| Numeric drift outside tolerance, Python now correct | Add `Case A` entry to `matlab_defects.yml`; update Python |
| Numeric drift outside tolerance, MATLAB now correct | Mark `Case B/C`; refresh fixture; relax tolerance with rationale |
| Shape/structure change | Make Python loaders shape-agnostic |
| Cosmetic field rename / metadata change | Update test assertions |

### Step 4 — Update ledgers

For each upstream-driven change, mark the relevant ledger entry:

```yaml
upstream_status: adopted-upstream
resolved_in: "cajigaslab/nSTAT@<sha>"
resolved_iter: "iter NN / YYYY-MM-DD"
resolved_notes: |
  <one-line summary of the fix's visible effect>
```

The `render_ledger.py` tool surfaces these fields in the rendered
`.md` files automatically.

### Step 5 — Tighten tolerances where headroom appeared

After upstream fixes land, Case C entries that were "RNG-sensitive"
or "init-convention-sensitive" often tighten to strict precision.
Re-run `python tools/parity/numerical_drift.py` and look for entries
where `max_abs_err` / `max_rel_err` is many orders below `rtol`/`atol`.

### Step 6 — Re-run holistic Reviewer

```bash
python tools/parity/build_composites.py --all
```

Then dispatch a Reviewer per topic (visual inspection of the
composite PNGs). A minor verdict may now promote to `matches` if the
upstream change closed a structural gap (e.g., MATLAB adding a
figure that Python already had as a "pedagogical extra").

### Step 7 — Update `tests/test_notebook_fidelity_audit.py` tolerances

Upstream MATLAB adding pedagogical figures (as in v10) pushes
`figure_delta` past historic `FIGURE_TOLERANCE` constants. Per
AGENT_GUIDE.md §0, widen the constant rather than reverting the
parity-improving change.

## Things not committed across reconciliation

When iterating in a reconciliation cycle:
- `.parity-review/` is gitignored; backup directories there don't
  enter the PR.
- The 53 `.mat` fixtures DO change; commit them with a defect-ledger
  entry citing the upstream commit.
- The 11 adopted-upstream ledger entries are durable history — never
  delete them, just toggle `upstream_status`.

## When to add a defect entry

Per `AGENT_GUIDE.md` §0:

- Defect fix (MATLAB bug) → `parity/matlab_defects.yml` + refresh gold fixture
- Stability improvement (`-expm1`, etc) → same
- Efficiency improvement (bit-equivalent) → no fixture refresh; just code comment

## Performance parity

> **Status (v12 — 2026-06-19):** 10 hot paths in baseline. 8 of 10 paths
> at-or-faster than MATLAB. 2 paths (`pp_decode_filter_linear` 2.78×,
> `kalman_filter` 2.27×) sit just above the 2× target due to per-step
> `np.linalg.solve` dispatch overhead on tiny matrices — documented
> algorithmic blockers requiring Numba/Cython (v13 candidate).

Numerical, visual, structural, and class-method parity were the gates
through v10. v11 adds a fifth dimension — wall-clock performance — by
timing five high-traffic public functions on both implementations and
recording the ratio. The five paths and the current baseline live in
[`parity/performance_baseline.yml`](../../parity/performance_baseline.yml).

### The five hot paths

| Path | Target (py/ml) | Investigate above |
|---|---|---|
| `analysis_run_for_neuron` (single neuron, 1000 spikes) | ≤ 1.5x | 5.0x |
| `pp_decode_filter_linear` (10000 steps, 2 cells) | ≤ 1.5x | 5.0x |
| `kalman_filter` (1000 steps, 4 states) | ≤ 1.5x | 5.0x |
| `simulate_point_process` (10 s @ 1 kHz, 10 Hz rate) | ≤ 1.5x | 5.0x |
| `history_compute_history` (200 spikes, 4 windows) | ≤ 1.5x | 5.0x |

Each MATLAB-side analogue is the same-name MATLAB function except
`simulate_point_process`, which compares against
`CIF.simulateCIFByThinningFromLambda` — the closest reachable surface
in the MATLAB checkout (the brief's preferred `simulatePointProcess` is
a Python-only convenience and `CIF.simulateCIFByThinning` requires
hist+stim+ens covariates the Python entry point does not expose).

### Re-running the baseline

```bash
make perf-check               # 3 runs/side, prints Markdown table  (~2-3 min)
make perf-check-full          # 10 runs/side, statistically tighter (~5-10 min)
make perf-check-capture       # 5 runs/side AND rewrites parity/performance_baseline.yml
```

The runner needs `/opt/homebrew/bin/matlab` and the local MATLAB
checkout (override with `--matlab-bin` and `--matlab-repo`, or set
`MATLAB_BIN` / `NSTAT_MATLAB_PATH`). All MATLAB tic/toc reps run inside
*one* `matlab -batch` invocation so the 5-10 s startup is amortised
across the runs, not multiplied by N.

The first MATLAB tic/toc per process is significantly slower than the
subsequent reps (MATLAB JIT warm-up). The runner reports both
`median_sec` (robust) and `runs_sec` (the per-rep list) so warm-up
spikes are visible without skewing the headline number.

CI does NOT run this gauntlet — MATLAB on the runner isn't available
and the wall-clock numbers are host-specific. The schema of
`parity/performance_baseline.yml` is validated by
`tests/test_performance_parity.py`, which runs in `make test` and
`make test-smoke`.

### When perf ratios regress

If a PR pushes a path's ratio above the previous baseline, investigate
before tightening or widening the target — performance ratios are
stability indicators, and a sudden 2x slowdown is a real signal. Common
causes:

1. **An extra defensive copy in a hot loop** — frequent for numerical
   work; usually fixable in-place with a view.
2. **A new Python-side validation pass on the input** — fine if rare,
   bad if it scales with the input length.
3. **A new sympy-backed CIF evaluation in a previously-vectorised
   path** — sympy is correct but slow; cache the lambdified callable.

Only after the cause is understood is it appropriate to either
re-capture the baseline (genuine algorithmic change, ratio acceptable)
or widen `target_for_parity` (genuine algorithmic complexity increase,
documented in the commit message).
