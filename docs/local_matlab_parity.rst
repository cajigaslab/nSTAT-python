Local MATLAB Parity
===================

Use this workflow when strict helpfile image parity needs MATLAB reference figures on a
machine that has MATLAB installed locally.

Requirements
------------

- ``matlab`` available on ``PATH``
- A clean MATLAB ``nSTAT`` checkout at the pinned parity SHA
- A materialized MATLAB ``helpfiles`` directory for Git LFS-backed ``.mat`` assets
- Python example data installed through ``nstat.data_manager.ensure_example_data``

Environment
-----------

Set the MATLAB checkout and the materialized helpfile source explicitly:

.. code-block:: bash

   export NSTAT_MATLAB_REPO=/tmp/upstream-nstat-matlab
   export NSTAT_MATLAB_HELPFILES_SOURCE=/path/to/materialized/nSTAT/helpfiles

Hydrate LFS-backed helpfile assets
----------------------------------

Fresh GitHub clones may contain Git LFS pointers for ``helpfiles/*.mat``. Hydrate those
files before exporting MATLAB reference images:

.. code-block:: bash

   python3 tools/reports/hydrate_matlab_helpfile_assets.py \
     --matlab-repo "$NSTAT_MATLAB_REPO" \
     --report-json output/matlab_help_images/hydrate_helpfiles_report.json

The hydrator verifies each copied file against the SHA256 recorded in the LFS pointer.
It does not commit or download raw example data.

Run local parity
----------------

Export smoke-group MATLAB figures, regenerate notebook metadata, execute the notebooks,
and run strict ordinal parity:

.. code-block:: bash

   python3 tools/reports/export_matlab_helpfile_figures_group.py \
     --repo-root . \
     --matlab-repo "$NSTAT_MATLAB_REPO" \
     --group smoke \
     --report-json output/matlab_help_images/report_smoke_group.json \
     --batch-timeout-seconds 1800

   python3 tools/notebooks/generate_helpfile_notebooks.py \
     --repo-root . \
     --matlab-help-root "$NSTAT_MATLAB_REPO/helpfiles"

   python3 tools/notebooks/check_notebook_consistency.py --repo-root .
   python3 tools/notebooks/run_notebooks.py --group smoke --timeout 900
   python3 tools/reports/check_helpfile_ordinal_image_parity.py --group smoke

Notes
-----

- ``publish_all_helpfiles`` remains execution-tested but is excluded from strict figure gating.
- The upstream MATLAB repository's Git LFS budget is external to ``nSTAT-python``. When
  a fresh MATLAB clone checks out pointer files instead of materialized ``.mat`` assets,
  this hydration step is the supported local workaround.
