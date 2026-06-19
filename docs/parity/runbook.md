# Parity maintenance runbook

The parity push completed in 38 iterations across 8 bundled PRs. This
runbook covers ongoing maintenance.

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

After v8, 4 topics remain at holistic 'minor' verdict:

- `StimulusDecode2D` — Python row 3 differs from MATLAB row 3 (composite ordering)
- `PPThinning` — blank ISI panel persists
- `ExplicitStimulusWhiskerData` — varying-richness Python figures
- `DecodingExample` — Python ships 7 figures vs MATLAB's 5 (intermediate steps)

Each is documented in `parity/matlab_pedagogical_gaps.yml` as accepted
divergence from MATLAB.

## When to add a defect entry

Per `AGENT_GUIDE.md` §0:

- Defect fix (MATLAB bug) → `parity/matlab_defects.yml` + refresh gold fixture
- Stability improvement (`-expm1`, etc) → same
- Efficiency improvement (bit-equivalent) → no fixture refresh; just code comment
