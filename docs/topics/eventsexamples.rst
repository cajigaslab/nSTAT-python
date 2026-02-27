Using the Events Class
======================

MATLAB help target: ``EventsExamples.html``

Concept
-------
This page mirrors the corresponding MATLAB help topic and documents the Python standalone equivalent.

API Mapping (MATLAB -> Python)
------------------------------
.. list-table::
   :header-rows: 1

   * - MATLAB API
     - Python API
   * - ``Events``
     - ``nstat.events.Events``

Migration Callout
-----------------
- MATLAB-style compatibility adapters remain importable for one major cycle and emit ``DeprecationWarning``.
- Prefer canonical Python modules under ``nstat`` for new code.

Python Usage
------------
.. code-block:: python

   import nstat
   print(nstat.__all__[:5])

Data Requirements
-----------------
Use ``nstat.datasets.list_datasets()`` and ``nstat.datasets.get_dataset_path(...)`` to access bundled datasets.

Expected Outputs
----------------
This topic should execute without MATLAB and produce deterministic summary metrics where applicable.

Debugging Notes
---------------
- Confirm dataset paths with ``nstat.datasets.list_datasets()`` and ``nstat.datasets.get_dataset_path(...)``.
- For notebook execution in CI/headless runs, set ``MPLBACKEND=Agg``.
- If parity checks fail, inspect generated reports under ``reports/`` for topic-level details.

Known Differences
-----------------
- Some legacy plotting helpers are represented via notebooks/docs instead of full method parity.
- Numerical outputs may vary if random seeds, bin widths, or sample rates differ from MATLAB defaults.

Example Utility
---------------
This example demonstrates how `Events` workflows map to standalone Python execution and why the resulting outputs are useful for model debugging and interpretation.

Paper Nomenclature
------------------
Use terminology consistent with Cajigas et al. (2012): conditional intensity function (CIF), point process generalized linear model (PP-GLM), maximum likelihood estimation (MLE), and related decoding/network terms where applicable.
Primary paper URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3491120/

Workflow Summary
----------------
1. Load data (or deterministic synthetic fallback) and configure the example pipeline.
2. Execute the Python topic workflow from ``examples/help_topics``.
3. Review structured outputs and generated notebook figures.
4. Compare behavior against MATLAB intent using parity reports when needed.

MATLAB MLX Alignment
--------------------
Reference Live Script: ``nSTAT_currentRelease_Local/helpfiles/EventsExamples.mlx``
MATLAB Live Script title: ``Events``
Paper Section Alignment
-----------------------
- Section 2.2.1
- Section 2.1.2

Topic-specific paper terms:
- event
- external covariate

Section-aligned Interpretation
-----------------------------
Encoding event-aligned structure for use in conditional intensity models.

Notebook
--------
A generated executable notebook is available at ``notebooks/helpfiles/EventsExamples.ipynb``.
