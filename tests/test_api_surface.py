from __future__ import annotations

import warnings

import nstat


def test_canonical_api_imports() -> None:
    assert nstat.Signal is not None
    assert nstat.SpikeTrain is not None
    assert nstat.SpikeTrainCollection is not None
    assert nstat.Trial is not None
    assert nstat.Analysis is not None
    assert nstat.Events is not None
    assert nstat.FitResult is not None
    assert nstat.FitSummary is not None
    assert nstat.CIFModel is not None
    assert nstat.DecoderSuite is not None
    assert nstat.getPaperDataDirs is not None
    assert nstat.get_paper_data_dirs is not None
    assert nstat.nSTAT_Install is not None


def test_matlab_facing_class_imports_are_canonical() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from nstat.ConfigColl import ConfigColl
        from nstat.CovColl import CovColl
        from nstat.DecodingAlgorithms import DecodingAlgorithms
        from nstat.SignalObj import SignalObj
        from nstat.TrialConfig import TrialConfig
        from nstat.Covariate import Covariate
        from nstat.nspikeTrain import nspikeTrain

        _ = SignalObj([0.0, 1.0], [1.0, 2.0])
        _ = Covariate([0.0, 1.0], [1.0, 2.0])
        _ = nspikeTrain([0.25, 0.5], makePlots=-1)
        _ = CovColl([])
        _ = ConfigColl([])
        assert DecodingAlgorithms is not None
        _ = TrialConfig()
        assert not any("deprecated" in str(item.message).lower() for item in w)
