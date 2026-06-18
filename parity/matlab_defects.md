# MATLAB defects and Python improvements ledger

This file records every place where Python's behavior intentionally diverges
from MATLAB nSTAT. Per AGENT_GUIDE.md §0, three reasons justify divergence:

1. **Defect fix** — MATLAB has a bug (off-by-one, wrong sign, instability)
2. **Stability improvement** — Python uses a more numerically robust algorithm
3. **Efficiency improvement** — Python uses a faster algorithm with bit-equivalent output

Schema for each entry:

```
## Defect: <one-line title>
- **MATLAB location:** `<file>:<line>` in `cajigaslab/nSTAT@<sha-or-tag>`
- **Defect class:** Bug | Stability | Efficiency
- **MATLAB behavior:** <what the original code does>
- **Correct behavior:** <what the science demands; cite reference>
- **Python implementation:** `<file>:<line>` in this repo
- **Fixture impact:** `tests/parity/fixtures/matlab_gold/<file>.mat` refreshed in commit `<sha>` (or "no fixture impact")
- **Discovered:** <iter # / date>
```

---

## Open entries

(none yet — iteration 1 of the parity push has not yet uncovered any)

---

## Reviewer checklist for parity-affecting PRs

- [ ] Every modified gold fixture has a defects-ledger entry
- [ ] Every "MATLAB does X but I changed Python to do Y" claim has a citation
- [ ] No silent fixture refresh (every `.mat` change has a commit message)
- [ ] No reverting a MATLAB-style convention thinking it's a bug — when in
      doubt, ask the maintainer
