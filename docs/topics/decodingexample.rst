Example Data Analysis - Decoding Univariate Simulated Stimuli (No History Effect)
=================================================================================

MATLAB help target: ``DecodingExample.html``

Concept
-------
This page mirrors the corresponding MATLAB help topic and documents the Python standalone equivalent.

API Mapping (MATLAB -> Python)
------------------------------
.. list-table::
   :header-rows: 1

   * - MATLAB API
     - Python API
   * - ``DecodingExample``
     - ``nstat.decoding.DecoderSuite``

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
This example demonstrates how `DecodingExample` workflows map to standalone Python execution and why the resulting outputs are useful for model debugging and interpretation.

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
Reference Live Script: ``nSTAT_currentRelease_Local/helpfiles/DecodingExample.mlx``
MATLAB Live Script title: ``STIMULUS DECODING``
Key Live Script headings:
- Generate the conditional Intensity Function
- Fit a model to the spikedata to obtain a model CIF

Paper Section Alignment
-----------------------
- Section 2.1.6
- Section 2.3.5
- Section 3.5

Topic-specific paper terms:
- decoding
- posterior estimate
- stimulus reconstruction

Section-aligned Interpretation
-----------------------------
Univariate stimulus decoding without history terms as baseline decoder.

Notebook
--------
A generated executable notebook is available at ``notebooks/helpfiles/DecodingExample.ipynb``.
