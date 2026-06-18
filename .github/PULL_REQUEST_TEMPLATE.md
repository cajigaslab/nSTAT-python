## Summary

<!-- one or two sentences -->

## Test plan

<!-- bulleted checklist -->

## Parity checklist (for parity-affecting PRs)

- [ ] If you touched library numerics (`nstat/*.py`), ran `make parity-check-quick`
      and confirmed SSIM non-regression.
- [ ] If you intentionally diverged from MATLAB output, added an entry to
      `parity/matlab_defects.md` (improvements) or
      `parity/matlab_pedagogical_gaps.md` (pedagogical extras).
- [ ] If you refreshed a `tests/parity/fixtures/matlab_gold/*.mat`, the PR body
      links to the relevant defects-ledger entry.
- [ ] Figure surplus (Python > MATLAB count) is either documented in
      `matlab_pedagogical_gaps.md` or removed.

Generated with [Claude Code](https://claude.com/claude-code)
