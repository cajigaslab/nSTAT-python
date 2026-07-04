# nstat-python — Tech Stack

- Python `>=3.10`; CI matrix runs 3.11 and 3.12. Package manager is plain
  `pip` (no uv/poetry/pipenv, no lockfile) — install via
  `pip install -e ".[dev]"` (setuptools build backend, `setuptools>=68`).
- Test framework: pytest `>=8.0`, no `pytest.ini`/`[tool.pytest.ini_options]` —
  defaults only, selection done via `-k` in Makefile targets.
- Core runtime deps (pinned floors in `pyproject.toml`): `numpy>=1.24`,
  `scipy>=1.10`, `matplotlib>=3.7`, `sympy>=1.13` (CIFs are sympy-backed),
  `PyYAML>=6.0`, `nbformat>=5.10`, `nbclient>=0.10` (notebook execution/fidelity).
- `dev` extra: pytest, `scikit-image>=0.22` (image/visual-parity audits),
  `sphinx>=8.0` + `sphinx-rtd-theme>=3.0` + `myst-parser>=4.0` (docs),
  `packaging>=23.0` (used by `tests/test_version_sync.py`).
- `nstat.extras.*` opt-in dependency groups (each backs one bridge module,
  install via `pip install nstat-toolbox[<group>]`): `neo`, `nwb` (pynwb),
  `pynapple`, `metrics` (pyspike), `nemos`, `dynamax`, `test-parity`
  (nemos+pykalman+statsmodels+nitime, requires `numpy>=2.0`), `clusterless`
  (`replay_trajectory_classification>=1.3,<1.4` — pinned, 1.4+ breaks the
  bridge API, tracked in issue #128), `spatial-gp` (gpflow), `hawkes` (tick),
  `dpp` (dppy), `latents` (elephant), `numba` (JIT for PPAF/Kalman hot loops).
  `spikeinterface` and `deep-learning` groups are placeholders (empty lists,
  modules not yet shipped).
- `all-extras` must stay the union of every *functional* group, EXCEPT
  `dynamax` and `clusterless` (JAX-heavy, ~200MB) and `spatial-gp`
  (gpflow/TensorFlow) which are deliberately excluded — install those explicitly.
- Version pins that matter: `nemos`/`replay_trajectory_classification` pull
  JAX which needs `numpy>=2.0` (`np.dtypes.StringDType()`), so `test-parity`
  and `all-extras` pin `numpy>=2.0` even though core only requires `>=1.24`.
  `statsmodels>=0.14.6` specifically (older imports removed
  `scipy._lib._util._lazywhere`).
- Package version currently `0.5.7` in `pyproject.toml` (cross-checked against
  `CITATION.cff`/`RELEASE_NOTES.md` by `tests/test_version_sync.py` — treat
  `pyproject.toml` as the live source, other docs like `CLAUDE.md` may lag by
  a patch version).
