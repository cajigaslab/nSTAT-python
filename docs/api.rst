API Reference
=============

Core modules:

- ``nstat.core`` — ``SignalObj``, ``Covariate``, ``nspikeTrain``
- ``nstat.trial`` — ``CovariateCollection`` (CovColl), ``SpikeTrainCollection``
  (nstColl), ``Trial``, ``TrialConfig``, ``ConfigCollection`` (ConfigColl)
- ``nstat.events`` — ``Events``
- ``nstat.history`` — ``History``
- ``nstat.cif`` — ``CIF``
- ``nstat.analysis`` — ``Analysis``
- ``nstat.fit`` — ``FitResult``, ``FitSummary`` (FitResSummary)
- ``nstat.decoding_algorithms`` — ``DecodingAlgorithms``
- ``nstat.confidence_interval`` — ``ConfidenceInterval``
- ``nstat.data_manager`` — dataset download and cache management

MATLAB-compatible public imports (``from nstat import ...``):

- ``SignalObj``, ``Covariate``, ``nspikeTrain``
- ``CovColl``, ``nstColl``
- ``Trial``, ``TrialConfig``, ``ConfigColl``
- ``History``, ``Events``, ``CIF``
- ``Analysis``, ``FitResult``, ``FitResSummary``
- ``DecodingAlgorithms``, ``ConfidenceInterval``
- ``getPaperDataDirs``, ``get_paper_data_dirs``
- ``nSTAT_Install``, ``nstat_install``
