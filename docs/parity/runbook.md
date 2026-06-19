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
