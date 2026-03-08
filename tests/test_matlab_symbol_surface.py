from __future__ import annotations

import inspect

from nstat import Analysis, CIF, DecodingAlgorithms


EXPECTED_SYMBOLS = {
    Analysis: {
        "RunAnalysisForNeuron",
        "RunAnalysisForAllNeurons",
        "GLMFit",
        "KSPlot",
        "plotFitResidual",
        "computeFitResidual",
        "computeKSStats",
        "plotInvGausTrans",
        "plotSeqCorr",
        "plotCoeffs",
    },
    CIF: {
        "setSpikeTrain",
        "setHistory",
        "simulateCIFByThinningFromLambda",
        "simulateCIF",
        "evalGradient",
        "evalGradientLog",
        "evalJacobian",
        "evalJacobianLog",
        "evalGradientLDGamma",
        "evalJacobianLDGamma",
    },
    DecodingAlgorithms: {
        "PPDecode_predict",
        "PPDecode_update",
        "PPDecode_updateLinear",
        "PPDecodeFilterLinear",
        "PPDecodeFilter",
        "PP_fixedIntervalSmoother",
        "PPHybridFilterLinear",
        "PPHybridFilter",
    },
}


def test_expected_matlab_symbol_surface_exists_and_is_callable() -> None:
    for obj, expected in EXPECTED_SYMBOLS.items():
        missing = sorted(name for name in expected if not callable(getattr(obj, name, None)))
        assert not missing, f"{obj.__name__} is missing MATLAB-facing callables: {missing}"


def test_expected_symbol_surface_has_python_runtime_signatures() -> None:
    for obj, expected in EXPECTED_SYMBOLS.items():
        for name in expected:
            signature = inspect.signature(getattr(obj, name))
            assert signature is not None
