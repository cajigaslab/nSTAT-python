# MATLAB ↔ Python parity diff — 2026-05-21

- MATLAB checkout: `/Users/iahncajigas/projects/nstat`
- Inventoried `.m` files: **23**
- Diff entries: **2**

## NEW IN MATLAB (not in `parity/manifest.yml`)

- **LinearCIF** (`LinearCIF.m`)
  - LinearCIF.m is a class file with 12 function declaration(s) but has no entry in parity/manifest.yml::public_api.
- **nSTAT_ExampleDataInfo** (`nSTAT_ExampleDataInfo.m`)
  - nSTAT_ExampleDataInfo.m is a function file with 1 function declaration(s) but has no entry in parity/manifest.yml::public_api.

## REMOVED FROM MATLAB

_(no findings)_

## MISMATCHED PYTHON PATH

_(no findings)_

---

**Note**: this report only inspects the public-API surface listed in `parity/manifest.yml`.  Behavioural parity is tracked separately via gold fixtures in `tests/parity/fixtures/matlab_gold/` and the class-fidelity audit in `parity/class_fidelity.yml`.
