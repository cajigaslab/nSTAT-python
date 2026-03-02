from __future__ import annotations

import numpy as np

from nstat.compat.matlab import Analysis
from nstat.compat.matlab import ConfigColl
from nstat.compat.matlab import CovColl
from nstat.compat.matlab import DecodingAlgorithms
from nstat.compat.matlab import FitResSummary
from nstat.compat.matlab import FitResult
from nstat.compat.matlab import SignalObj
from nstat.compat.matlab import nspikeTrain
from nstat.compat.matlab import nstColl
from nstat.compat.matlab import TrialConfig
from nstat.signal import Covariate
from nstat.fit import FitResult as NativeFitResult



def test_signalobj_alias_methods() -> None:
    t = np.linspace(0.0, 1.0, 5)
    x = np.sin(2.0 * np.pi * t)
    sig = SignalObj(time=t, data=x, name="sig")

    assert sig.getNumSamples() == 5
    assert sig.getNumSignals() == 1
    assert np.isclose(sig.getSampleRate(), 4.0)
    assert np.isclose(sig.getDuration(), 1.0)
    assert sig.getData().shape == (5,)



def test_trialconfig_matlab_style_constructor() -> None:
    cfg = TrialConfig(covariateLabels=["stim"], Fs=500.0, fitType="binomial", name="cfg")
    assert cfg.getSampleRate() == 500.0
    assert cfg.getFitType() == "binomial"
    assert cfg.getCovariateLabels() == ["stim"]



def test_configcoll_matlab_aliases() -> None:
    cfg1 = TrialConfig(covariateLabels=["stim"], Fs=1000.0, fitType="poisson", name="cfg_a")
    cfg2 = TrialConfig(covariateLabels=["ctx"], Fs=500.0, fitType="binomial", name="cfg_b")
    coll = ConfigColl([cfg1, cfg2])

    assert coll.getConfigNames() == ["cfg_a", "cfg_b"]
    assert coll.getConfig(2).name == "cfg_b"
    assert coll.getConfig("cfg_a").fit_type == "poisson"

    cfg3 = TrialConfig(covariateLabels=["stim", "ctx"], Fs=250.0, fitType="poisson", name="cfg_c")
    coll.addConfig(cfg3)
    assert coll.getConfig(3).name == "cfg_c"

    coll.setConfigNames(["a", "b", "c"])
    assert coll.getConfigNames() == ["a", "b", "c"]

    subset = coll.getSubsetConfigs([1, 3])
    assert [cfg.name for cfg in subset.configs] == ["a", "c"]

    payload = coll.toStructure()
    restored = ConfigColl.fromStructure(payload)
    assert restored.getConfigNames() == ["a", "b", "c"]


def test_analysis_fitglm_alias() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=1200)
    X = x[:, None]
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.9 * x)))
    y = rng.binomial(1, p).astype(float)

    result = Analysis.fitGLM(X=X, y=y, fitType="binomial")
    assert isinstance(result, NativeFitResult)
    assert np.isfinite(result.log_likelihood)



def test_decoding_aliases() -> None:
    mat = np.array(
        [
            [0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    rates, pvals, sig = DecodingAlgorithms.computeSpikeRateCIs(mat)
    assert rates.shape[0] == mat.shape[0]
    assert pvals.shape == (mat.shape[0], mat.shape[0])
    assert sig.shape == (mat.shape[0], mat.shape[0])


def test_collection_and_covariate_aliases() -> None:
    t = np.linspace(0.0, 1.0, 101)
    c1 = Covariate(time=t, data=np.sin(2 * np.pi * t), name="stim", labels=["stim"])
    c2 = Covariate(time=t, data=np.cos(2 * np.pi * t), name="ctx", labels=["ctx"])
    coll = CovColl([c1, c2])
    X, labels = coll.dataToMatrix()
    assert X.shape[1] == 2
    assert labels == ["stim", "ctx"]
    assert coll.getCovIndFromName("ctx") == 1
    assert coll.isCovPresent("stim")


def test_covcoll_extended_aliases() -> None:
    t = np.linspace(0.0, 1.0, 101)
    c1 = Covariate(time=t, data=np.sin(2 * np.pi * t), name="stim", labels=["stim"])
    c2 = Covariate(time=t, data=np.cos(2 * np.pi * t), name="ctx", labels=["ctx"])
    coll = CovColl([c1, c2])

    assert coll.containsChars("stimulus", "tu")
    assert coll.isaSelectorCell(["stim", "ctx"])
    assert coll.covIndFromSelector(["ctx"]) == [1]
    assert coll.getCovMaskFromSelector([2]) == [1]
    assert coll.generateRemainingIndex(["stim"]) == [1]
    assert coll.generateSelectorCell([0, 1]) == ["stim", "ctx"]
    assert coll.getSelectorFromMasks([1]) == ["ctx"]

    coll.setMasksFromSelector(["stim"])
    assert coll.getCovDataMask() == [0]
    coll.maskAwayCov(["stim"])
    assert coll.getCovDataMask() == [1]
    coll.maskAwayAllExcept(["ctx"])
    assert coll.getCovDataMask() == [1]

    coll.setCovShift(0.1)
    assert np.isclose(coll.getTime()[0], 0.1)
    coll.resetCovShift()
    assert np.isclose(coll.getTime()[0], 0.0)
    assert np.isclose(coll.findMinTime(), 0.0)
    assert np.isclose(coll.findMaxTime(), 1.0)

    payload = coll.dataToStructure()
    restored = CovColl.fromStructure(payload)
    assert restored.getAllCovLabels() == ["stim", "ctx"]


def test_nspiketrain_extended_aliases() -> None:
    st = nspikeTrain(spike_times=np.array([0.1, 0.3, 0.55]), t_start=0.0, t_end=1.0, name="u1")
    assert np.isclose(st.computeRate(), st.getFiringRate())
    assert np.isclose(st.getMinISI(), 0.2)
    assert st.getMaxBinSizeBinary() > 0.0
    assert st.getLStatistic() >= 0.0
    assert st.computeStatistics()["n_spikes"] == 3.0

    st.setSigRep(np.array([0, 1, 0, 1], dtype=float))
    assert st.isSigRepBinary()
    st.clearSigRep()
    assert st.getSigRep(binSize_s=0.1).ndim == 1

    st_copy = st.nstCopy()
    assert np.allclose(st_copy.spike_times, st.spike_times)
    parts = st.partitionNST([0.0, 0.5, 1.0])
    assert len(parts) == 2

    payload = st.toStructure()
    restored = nspikeTrain.fromStructure(payload)
    assert np.allclose(restored.spike_times, st.spike_times)


def test_spike_collection_aliases() -> None:
    st1 = nspikeTrain(spike_times=np.array([0.1, 0.3]), t_start=0.0, t_end=1.0, name="u1")
    st2 = nspikeTrain(spike_times=np.array([0.2, 0.4]), t_start=0.0, t_end=1.0, name="u2")
    coll = nstColl([st1, st2])
    assert coll.getNumUnits() == 2
    assert np.isclose(coll.getFirstSpikeTime(), 0.1)
    assert np.isclose(coll.getLastSpikeTime(), 0.4)
    assert coll.getNSTnameFromInd(1) == "u2"
    merged = coll.toSpikeTrain()
    assert merged.spike_times.size == 4

    assert coll.ensureConsistancy()
    assert coll.getMaxBinSizeBinary() > 0.0
    assert coll.findMaxSampleRate() > 0.0
    assert coll.estimateVarianceAcrossTrials(binSize_s=0.1).ndim == 1

    coll.setNeuronMaskFromInd([1])
    assert coll.isNeuronMaskSet()
    assert coll.getIndFromMask() == [0]
    assert coll.getIndFromMaskMinusOne() == [0]
    coll.resetMask()
    assert coll.getIndFromMask() == [0, 1]

    coll.setNeighbors([[1], [0]])
    assert coll.areNeighborsSet()
    assert coll.getNeighbors() == [[1], [0]]

    basis = nstColl.generateUnitImpulseBasis(basisWidth_s=0.2, sampleRate_hz=100.0, totalTime_s=1.0)
    assert basis.data.ndim == 2
    ens = coll.getEnsembleNeuronCovariates(binSize_s=0.1, mode="count")
    assert ens.nActCovar() == 2


def test_fit_aliases() -> None:
    fit1 = FitResult(
        coefficients=np.array([0.2]),
        intercept=-1.0,
        fit_type="binomial",
        log_likelihood=-9.0,
        n_samples=100,
        n_parameters=2,
        parameter_labels=["stim"],
    )
    fit2 = FitResult(
        coefficients=np.array([0.4]),
        intercept=-0.7,
        fit_type="binomial",
        log_likelihood=-8.0,
        n_samples=100,
        n_parameters=2,
        parameter_labels=["stim"],
    )
    assert fit1.getCoeffIndex("stim") == 0
    assert np.isclose(fit1.getParam("aic"), fit1.getAIC())

    summary = FitResSummary([fit1, fit2])
    diff = summary.getDiffAIC()
    assert diff.shape == (2,)
    mat = summary.computeDiffMat("bic")
    assert mat.shape == (2, 2)
