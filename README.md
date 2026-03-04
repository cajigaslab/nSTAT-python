# nSTAT-python

`nSTAT-python` is a clean-room Python implementation of the nSTAT toolbox.

[![test-and-build](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/ci.yml)
[![parity-gate](https://github.com/cajigaslab/nSTAT-python/actions/workflows/parity-gate.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/parity-gate.yml)
[![performance-parity](https://github.com/cajigaslab/nSTAT-python/actions/workflows/performance-parity.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/performance-parity.yml)
[![image-mode-parity](https://github.com/cajigaslab/nSTAT-python/actions/workflows/image-mode-parity.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/image-mode-parity.yml)
[![pages](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/pages.yml)
[![validation-pdf](https://github.com/cajigaslab/nSTAT-python/actions/workflows/validation-pdf.yml/badge.svg)](https://github.com/cajigaslab/nSTAT-python/actions/workflows/validation-pdf.yml)

## Design goals
- Zero MATLAB runtime dependency
- Class-structure parity with MATLAB nSTAT
- Python-native implementation and docs
- Searchable help pages on GitHub Pages
- Executable learning notebooks

## Installation

```bash
python -m pip install nstat
```

From source:

```bash
git clone git@github.com:cajigaslab/nSTAT-python.git
cd nSTAT-python
python -m pip install -e .[dev,docs,notebooks]
```

## How to install nSTAT (post-install setup)

Run the Python-native setup helper `nstat_install` (no MATLAB required):

```bash
nstat-install
```

Equivalent Python API:

```python
from nstat.install import nstat_install

report = nstat_install()
print(report.cache_dir)
```

## Quick start

```python
import numpy as np
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain

t = np.linspace(0.0, 1.0, 1001)
x = np.sin(2 * np.pi * 5 * t)
cov = Covariate(time=t, data=x, name="stimulus", labels=["stim"])
spikes = SpikeTrain(spike_times=np.array([0.11, 0.42, 0.77]), t_start=0.0, t_end=1.0)
print(cov.sample_rate_hz, spikes.firing_rate_hz())
```

## Documentation and help pages
- Docs home: [cajigaslab.github.io/nSTAT-python](https://cajigaslab.github.io/nSTAT-python/)
- Help index: [cajigaslab.github.io/nSTAT-python/help](https://cajigaslab.github.io/nSTAT-python/help/)
- Canonical validation artifacts: [CANONICAL_VALIDATION_ARTIFACTS.md](./CANONICAL_VALIDATION_ARTIFACTS.md)

## Data policy
Only example data may be shared with MATLAB nSTAT. All non-data files are unique to this repository.

## MATLAB Data Mirror

Use the bundled workflow to mirror MATLAB toolbox example data into this repo with checksums:

```bash
python tools/data_mirror/run_mirror_workflow.py \
  --source-root /path/to/matlab/nSTAT/data \
  --version 20260302 \
  --clean
```

This command performs:
1. Source snapshot manifest generation.
2. Byte-for-byte mirrored copy into `data/shared/matlab_gold_<version>/`.
3. Shared-data allowlist regeneration.
4. Dataset API manifest regeneration (`data/datasets_manifest.json`).
5. Strict checksum verification.

To re-verify later:

```bash
python tools/data_mirror/verify_matlab_data.py \
  --manifest data/shared/matlab_gold_20260302.manifest.json \
  --strict
```

## MATLAB parity workflow

Generate parity inventories and the machine-readable gap report:

```bash
python tools/parity/build_parity_snapshot.py \
  --matlab-root /path/to/matlab/nSTAT \
  --fail-on high
```

Artifacts are written to:
- `parity/matlab_api_inventory.json`
- `parity/python_api_inventory.json`
- `parity/parity_gap_report.json`
- `parity/TIER1_PORT_BACKLOG.md`

Tier-1 progress gate:

```bash
python tools/parity/check_tier1_progress.py \
  --report parity/parity_gap_report.json \
  --policy parity/tier1_gate_policy.yml
```

MATLAB-style adapters are available under:
- `nstat.compat.matlab`

Sync parity artifacts (functional audit + sprint backlog + help dashboard/docs):

```bash
python tools/parity/sync_parity_artifacts.py \
  --matlab-root /path/to/matlab/nSTAT
```

## RC Release Automation

Use the GitHub Actions workflow `.github/workflows/release-rc.yml` to:
1. Rebuild parity artifacts and enforce gates.
2. Generate a full validation PDF.
3. Auto-generate RC release notes from parity reports.
4. Publish/update a pre-release with the latest PDF asset attached.

You can trigger it from GitHub Actions (`release-rc`) with an input tag
like `v1.0.0-rc3`.

## Stable Release Promotion

Use `.github/workflows/release-stable.yml` to promote a validated RC to a stable release.
The workflow:
1. Checks out the RC tag commit.
2. Runs hard checks (lint, typing, unit tests, docs build).
3. Runs parity and numeric-drift gates.
4. Regenerates the validation PDF.
5. Creates/pushes the stable tag and publishes a non-prerelease release.

Inputs:
- `rc_tag` (for example `v1.0.0-rc3`)
- `stable_tag` (for example `v1.0.0`)

## PR-Native Parity Gate

`.github/workflows/parity-gate.yml` runs on every pull request and enforces:
- Parity snapshot gate (`--fail-on medium`)
- Numeric drift thresholds
- Functional parity policy
- Example-output parity policy
- Synchronized parity artifacts (`parity/*`, `docs/help/*`, `docs/notebooks.md`, `baseline/help_mapping.json`)
- Full gate-mode validation PDF generation with canonical artifact names:
  - `output/pdf/validation_gate_mode_latest.pdf`
  - `output/pdf/validation_gate_mode_latest.json`
  - `output/pdf/validation_gate_mode_latest.csv`

## Function-Level Performance Parity

Run deterministic Python workload benchmarks:

```bash
python tools/performance/run_python_benchmarks.py \
  --tiers S,M,L \
  --repeats 7 \
  --warmup 2 \
  --out-json output/performance/python_performance_report.json \
  --out-csv output/performance/python_performance_report.csv
```

Compare Python runtime/memory metrics against MATLAB baseline fixtures:

```bash
python tools/performance/compare_matlab_python_performance.py \
  --python-report output/performance/python_performance_report.json \
  --matlab-report tests/performance/fixtures/matlab/performance_baseline_470fde8.json \
  --policy parity/performance_gate_policy.yml \
  --previous-python-report tests/performance/fixtures/python/performance_baseline_linux_latest.json \
  --report-out parity/performance_parity_report.json \
  --csv-out parity/performance_parity_report.csv \
  --fail-on-regression \
  --require-regression-env-match
```

Generate MATLAB baseline report (controlled environment):

```bash
matlab -batch "addpath('matlab/benchmark'); run_matlab_performance_benchmarks( ...
  'tests/performance/fixtures/matlab/performance_baseline_470fde8.json', ...
  'tests/performance/fixtures/matlab/performance_baseline_470fde8.csv', ...
  '/path/to/nSTAT')"
```

## Branch Protection Automation

To apply required checks on `main` (admin token required):

Current required checks on `main`:
- `unit-lint (3.11)`
- `unit-lint (3.12)`
- `docs-smoke-notebooks`
- `matlab-data-integrity`
- `cleanroom-compliance`
- `parity-checks`
- `build-validation-pdf`
- `image-mode-parity`
- `performance-parity`

## Paper reference
Cajigas I, Malik WQ, Brown EN. nSTAT: Open-source neural spike train analysis toolbox for Matlab. *J Neurosci Methods* (2012), DOI: `10.1016/j.jneumeth.2012.08.009`, PMID: `22981419`.
