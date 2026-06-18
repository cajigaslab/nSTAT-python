# MATLAB ↔ Python parity

This Python toolbox (`nstat-python`) produces functionally equivalent output to
the MATLAB `nSTAT` toolbox (https://github.com/cajigaslab/nSTAT). Parity is
verified at four levels:

1. **Numerical output** — every public function's output is verified against
   MATLAB baselines via `tools/parity/numerical_drift.py` and the
   `tests/parity/fixtures/matlab_gold/*.mat` gold fixtures.
2. **Figure content** — every Python notebook produces the same per-figure
   content as the corresponding MATLAB helpfile.
3. **Figure appearance** — MATLAB-default colors (`#0072BD`/`#D95319`/...),
   `jet` cmap, MATLAB-style axes, ticks, grid, font, layout.
4. **Notebook structure** — `notebooks/<Topic>.ipynb` mirrors
   `helpfiles/<Topic>.mlx` sections + narrative text.

## Current parity state

| Level | Coverage | Status |
|---|---|---|
| Numerical (gold fixtures) | 27 / 27 | all pass |
| Numerical (drift tool) | 10 priority functions | all within tolerance |
| Figure counts | 17/23 exact + 5 surplus (ledger) + 1 deficit | OK |
| SSIM gates | 23+ entries | all pass thresholds |

## Re-running the parity check

```bash
make parity-check           # full sweep: extract .mlx → execute notebooks → SSIM → composites
make parity-check-quick     # composite + SSIM only (~30s, skips notebook execution)
```

Both write to `.parity-review/` (gitignored). Open
`.parity-review/parity_index.html` in any browser for a clickable side-by-side
comparison.

## Documenting divergences

When Python intentionally diverges from MATLAB:
- **Numerical/algorithmic divergence** (Python is more stable / efficient /
  correct) → record in `parity/matlab_defects.md`
- **Figure surplus** (Python adds a pedagogical extra) → record in
  `parity/matlab_pedagogical_gaps.md` with a MATLAB upstream action line

Gold fixtures may be refreshed when justified; refresh must be paired with a
defects-ledger entry per AGENT_GUIDE.md §0.

## Future verification

The parity check is **on-demand**, not gated per PR:
- `.github/workflows/parity-check.yml` is `workflow_dispatch` only.
- Trigger via `gh workflow run parity-check.yml --ref main` after any
  parity-affecting change.

See also:
- `parity/matlab_defects.md` — documented Python improvements over MATLAB
- `parity/matlab_pedagogical_gaps.md` — documented Python pedagogical extras
- `parity/visual_fidelity.yml` — SSIM gate manifest
- `parity/numerical_drift_spec.yml` — numerical drift entries
- `AGENT_GUIDE.md` §0 — the binding parity principle
