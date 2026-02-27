# Python nSTAT

This repository contains the standalone Python implementation of nSTAT.

## Canonical API modules

- `nstat.signal`: `Signal`, `Covariate`
- `nstat.spikes`: `SpikeTrain`, `SpikeTrainCollection`
- `nstat.events`: `Events`
- `nstat.history`: `HistoryBasis`
- `nstat.trial`: `CovariateCollection`, `TrialConfig`, `ConfigCollection`, `Trial`
- `nstat.cif`: `CIFModel`
- `nstat.analysis`: `Analysis`
- `nstat.fit`: `FitResult`, `FitSummary`
- `nstat.decoding`: `DecoderSuite`
- `nstat.datasets`: dataset registry and checksum verification

MATLAB-style entry points remain importable as compatibility adapters with `DeprecationWarning` messages.

## Install

```bash
python3 -m pip install -e .
```

## Run paper examples equivalent

```bash
python3 examples/nstat_paper_examples.py --repo-root .
```

## Generate docs and notebooks

```bash
python3 tools/generate_help_topic_docs.py
python3 tools/generate_example_notebooks.py
```

## Notebook figure parity

```bash
# One-time baseline sync from local MATLAB helpfiles PNG assets.
python3 tools/sync_matlab_reference_figures.py \
  --matlab-helpfiles /path/to/nSTAT/helpfiles --clean

# Validate expected MATLAB-topic figure counts in help assets.
python3 tools/validate_matlab_topic_figure_counts.py \
  --matlab-helpfiles /path/to/nSTAT/helpfiles

# Extract MATLAB Live Script title/heading metadata for notebook alignment.
python3 tools/extract_mlx_metadata.py \
  --matlab-helpfiles /path/to/nSTAT/helpfiles

# Regenerate notebooks with required narrative sections.
python3 tools/generate_example_notebooks.py

# Execute notebooks in-place and persist outputs for GitHub rendering.
python3 tools/publish_example_notebooks.py

# Verify published notebooks include required sections and embedded outputs.
python3 tools/verify_published_notebooks.py --enforce-gate

# Execute 25/25 example notebooks and enforce figure-count contracts.
python3 tools/verify_examples_notebooks.py

# Compare generated notebook figures to vendored MATLAB baselines.
python3 tools/compare_notebook_figures_to_matlab.py --enforce-gate
```

Published notebook policy:
- Keep outputs embedded in `notebooks/helpfiles/*.ipynb` on the default branch so GitHub web rendering shows figures and execution context.
- Do not strip notebook outputs in PRs that update help examples.
- Keep terminology consistent with Cajigas et al. (2012) and retain paper section references in notebook/help narratives.
- Keep notebook narrative aligned to MATLAB Live Script headings for each corresponding topic.

## Validation

```bash
python3 tools/freeze_port_baseline.py
python3 tools/generate_method_parity_matrix.py
python3 tools/generate_implemented_method_coverage.py
python3 tools/generate_example_notebooks.py
python3 tools/publish_example_notebooks.py
python3 tools/verify_published_notebooks.py --enforce-gate
python3 tools/verify_examples_notebooks.py
python3 tools/compare_notebook_figures_to_matlab.py --enforce-gate
NSTAT_MATLAB_EXTRA_ARGS='-maca64 -nodisplay -noFigureWindows -softwareopengl' \
  python3 tools/verify_python_vs_matlab_similarity.py --enforce-gate
python3 tools/freeze_similarity_baseline.py
python3 tools/verify_offline_standalone.py
python3 -m pytest
```

If Git LFS assets are unavailable (for example, CI quota exhaustion), set
`NSTAT_ALLOW_SYNTHETIC_DATA=1` to use deterministic synthetic fallbacks for
data-heavy paper example loaders.

### Local parity block debugging

```bash
python3 tools/debug_parity_blocks.py \
  --set-actions-runner-svc \
  --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"
```

Single wrapper command (fail-fast ladder):

```bash
tools/run_parity_ladder.sh
```

Single preflight command (Stage A ladder + selected Stage B topics):

```bash
tools/run_parity_preflight.sh
```

Notes:

- Runs blocks in order: `core_smoke -> timeout_front -> graphics_mid -> heavy_tail -> full_suite`.
- Exits immediately if a block regresses (`python_ok`, `matlab_ok`, scalar overlap, parity contract, or regression gate).
- Includes runtime regression guard using machine baseline block times with multiplier `NSTAT_PARITY_RUNTIME_MULTIPLIER` (default `2.5`).
- Set `NSTAT_PARITY_RUNTIME_MULTIPLIER=0` to disable runtime regression checks.
- Pass specific block names as args to run subset ladders, e.g.:
  `tools/run_parity_ladder.sh core_smoke timeout_front`.
- Ladder writes retry telemetry to `reports/parity_retry_summary.json` (block, attempt count, retry reason, timeout-topic list).
- Retry behavior is controlled by `NSTAT_PARITY_RETRY_TIMEOUT_BLOCKS` and `NSTAT_PARITY_TIMEOUT_RETRY_BLOCKS`.
- Set `NSTAT_MATLAB_TOPIC_MAX_ATTEMPTS=2` to retry per-topic MATLAB timeouts/crashes once before failing.
- Set `NSTAT_PARITY_RETRY_RECOVERABLE_BLOCKS=1` and `NSTAT_PARITY_RECOVERABLE_RETRY_BLOCKS` to retry block failures caused by recoverable MATLAB failures (timeouts/crash signatures).
- Preflight topic selection can be overridden with `NSTAT_PARITY_PREFLIGHT_STAGEB_TOPICS`.

See `docs/parity_runbook.rst` for the exact locally validated parity command set.

Use targeted blocks to debug delays locally before running remote CI:

```bash
# 1) Fast API/parity smoke
python3 tools/debug_parity_blocks.py --blocks core_smoke \
  --set-actions-runner-svc --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"

# 2) Former timeout-prone front topics
python3 tools/debug_parity_blocks.py --blocks timeout_front \
  --set-actions-runner-svc --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"

# 3) Graphics-sensitive middle topics
python3 tools/debug_parity_blocks.py --blocks graphics_mid \
  --set-actions-runner-svc --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"

# 4) Heavy tail topics
python3 tools/debug_parity_blocks.py --blocks heavy_tail \
  --set-actions-runner-svc --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"

# 5) Full gate-equivalent suite
python3 tools/debug_parity_blocks.py --blocks full_suite \
  --set-actions-runner-svc --matlab-extra-args "-maca64 -nodisplay -noFigureWindows -softwareopengl"
```

Summarize a parity report quickly:

```bash
python3 tools/summarize_parity_report.py reports/parity_block_full_suite.json
```

Recent local baseline on this machine (MATLAB R2025b, no figure windows, software OpenGL):

- `core_smoke`: ~47s
- `timeout_front`: ~122s
- `graphics_mid`: ~291s
- `heavy_tail`: ~385s
- `full_suite` (25 topics): ~826s

## CI

- `.github/workflows/python-ci.yml` runs docs, notebook verification, offline standalone checks, and `pytest`.
- `.github/workflows/matlab-parity-gate.yml` runs MATLAB/Python parity gate on self-hosted macOS runners with MATLAB installed.

## Repo split inventory

Snapshot MATLAB/Python help and example coverage before splitting `nSTAT` and `nSTAT-python`:

```bash
python3 tools/generate_repo_split_inventory.py
```

Outputs:
- `reports/repo_split_inventory/summary.json`
- `reports/repo_split_inventory/topic_coverage_matrix.json`
- `reports/repo_split_inventory/split_readiness_gates.json`
