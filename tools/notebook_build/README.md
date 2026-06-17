# `tools/notebook_build/` — notebook tooling

Tooling for the parity notebooks under [`notebooks/`](../../notebooks/). The
notebooks themselves are the **source of truth**; everything here operates on
them or describes them.

## Live tooling (safe to run; used by CI / `make`)

| File | Role |
|---|---|
| `run_notebooks.py` | Execute notebooks by group (`--group ci_smoke` / `parity_core` / `helpfile_full`) and validate figure contracts. |
| `build_notebook_galleries.py` | Execute notebooks and (re)build the generated galleries under `docs/notebook_galleries/`. |
| `embed_figures.py` | Backfill committed gallery PNGs into notebook cell outputs so figures render on GitHub. Remediation target for `tests/test_notebook_gallery_figures.py::test_notebooks_embed_their_figures`. |
| `sanitize_notebooks.py` | Normalize notebook metadata / strip MATLAB-port markers in place. |
| `sync_parity_notes.py` | Sync the MATLAB parity markdown cells from `parity_notes.yml` into the notebooks. |
| `changed_topics.py` | Map a git diff to affected notebook topics (CI test selection). |
| `notebook_manifest.yml` | The single notebook **catalogue** (topic → file → run_group), kept 1:1 with `notebooks/` (guarded by `tests/test_structure_hygiene.py`). |
| `topic_groups.yml` | Notebook **CI groups** (which topics run in which group). |
| `parity_notes.yml` | Per-notebook **MATLAB fidelity metadata** (source `.mlx`, fidelity status, remaining differences). |

These three YAMLs are intentionally kept as separate single-purpose files.

## ⚠️ Historical bootstrap generator — DO NOT RE-RUN

`build_network_tutorial_notebook.py` scaffolded `NetworkTutorial.ipynb` under
`notebooks/`. That notebook has since been hand-refined, sanitized,
parity-annotated and executed, and has **diverged from generator output**.
Re-running the generator **overwrites and corrupts** the committed notebook
(deletes ~320 lines of curated content). It is kept for provenance and is
covered by `tests/test_network_tutorial_builder.py`; to add or change a
notebook, edit the `.ipynb` directly (then `sanitize_notebooks.py` /
`sync_parity_notes.py` as needed).

The earlier helpfile/decoding/paper bootstrap generators
(`build_analysis_help_notebooks.py`, `build_decoding_fidelity_notebooks.py`,
`build_foundational_help_notebooks.py`, `build_helpfile_fidelity_notebooks.py`,
`build_nstat_paper_notebook.py`, and the shared `_help_notebook_writer.py`)
were removed once their output had fully diverged; the committed `.ipynb`
files they originally scaffolded remain the source of truth.
