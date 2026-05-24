API Reference
=============

This page is **auto-generated** from the package source by Sphinx's
``autosummary`` extension.  Adding a name to :data:`nstat.__all__`
automatically makes it appear here on the next docs build — there is
no manual list to maintain.

The grouping below mirrors the categories in
:doc:`ClassDefinitions` and the public-API section of
``AGENT_GUIDE.md``.  Click any symbol to jump to its rendered
docstring (with parameter tables, type hints, and links to source).

.. currentmodule:: nstat


Core data primitives
--------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   SignalObj
   Signal
   Covariate
   nspikeTrain
   SpikeTrain
   Events
   ConfidenceInterval


Collections
-----------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   nstColl
   SpikeTrainCollection
   CovColl
   CovariateCollection
   ConfigColl
   ConfigCollection


Experiment and configuration
----------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   Trial
   TrialConfig
   History
   HistoryBasis


Modeling and inference
----------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   Analysis
   CIF
   CIFModel
   LinearCIF
   FitResult
   FitSummary
   FitResSummary

.. autosummary::
   :toctree: _autosummary

   psth


Decoding
--------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   DecodingAlgorithms
   DecoderSuite
   PoissonGLMResult

.. autosummary::
   :toctree: _autosummary

   fit_poisson_glm


Simulation
----------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   PointProcessSimulation
   NetworkSimulationResult

.. autosummary::
   :toctree: _autosummary

   simulate_poisson_from_rate
   simulate_cif_from_stimulus
   simulate_point_process
   simulate_two_neuron_network
   run_full_paper_examples


Plot style
----------

.. autosummary::
   :toctree: _autosummary

   set_plot_style
   get_plot_style
   apply_plot_style


Installation and data
---------------------

.. autosummary::
   :toctree: _autosummary

   nstat_install
   nSTAT_Install
   get_dataset_path
   list_datasets
   verify_checksums
   getPaperDataDirs
   get_paper_data_dirs


MATLAB bridge (optional)
------------------------

.. autosummary::
   :toctree: _autosummary

   is_matlab_available
   get_matlab_nstat_path
   set_matlab_nstat_path


Exceptions and warnings
-----------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst

   DataNotFoundError
   MatlabEngineError
   MatlabFallbackWarning
   ParityValidationError
   UnsupportedWorkflowError
