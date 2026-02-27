Notebook Figure Parity
======================

This runbook enforces figure generation and MATLAB visual parity for the 25
example topics listed under ``helpfiles/helptoc.xml`` (``nstat_examples`` node).

Baseline assets
---------------

- Contract: ``examples/help_topics/figure_contract.json``
- Paper terminology map: ``examples/help_topics/paper_nomenclature.json``
- MATLAB Live Script metadata: ``examples/help_topics/matlab_mlx_metadata.json``
- Baselines: ``reference/matlab_helpfigures/<Topic>/fig_###.png``
- Baseline manifest: ``reference/matlab_helpfigures/manifest.json``
- Thresholds: ``reference/matlab_helpfigures/thresholds.json``

Sync baselines from MATLAB helpfiles
------------------------------------

.. code-block:: bash

   python3 tools/sync_matlab_reference_figures.py \
     --matlab-helpfiles /path/to/nSTAT/helpfiles --clean

Validate MATLAB helpfile figure inventory
-----------------------------------------

.. code-block:: bash

   python3 tools/validate_matlab_topic_figure_counts.py \
     --matlab-helpfiles /path/to/nSTAT/helpfiles

Extract MATLAB Live Script metadata
-----------------------------------

.. code-block:: bash

   python3 tools/extract_mlx_metadata.py \
     --matlab-helpfiles /path/to/nSTAT/helpfiles

Generate and validate notebooks
-------------------------------

.. code-block:: bash

   python3 tools/generate_example_notebooks.py
   python3 tools/publish_example_notebooks.py
   python3 tools/verify_published_notebooks.py --enforce-gate
   python3 tools/verify_examples_notebooks.py

Compare notebook figures to MATLAB baseline
-------------------------------------------

.. code-block:: bash

   python3 tools/compare_notebook_figures_to_matlab.py --enforce-gate

Artifacts
---------

- Published execution report: ``reports/published_notebook_execution.json``
- Published notebook check: ``reports/published_notebook_check.json``
- Notebook verification report: ``reports/examples_notebook_verification.json``
- Figure parity report: ``reports/notebook_figure_parity.json``
- Generated figures: ``reports/figures/notebooks/<Topic>/fig_###.png``
- Failure diffs: ``reports/figures/diffs/<Topic>/fig_###_diff.png``

Publishing policy
-----------------

- Keep notebook outputs committed in ``notebooks/helpfiles/*.ipynb`` so figures render on GitHub.
- Preserve required notebook narrative sections for each help example.
