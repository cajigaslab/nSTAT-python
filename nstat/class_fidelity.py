from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


EXPECTED_RUNTIME_MEMBERS: dict[str, tuple[str, ...]] = {
    "nstat.SignalObj": (
        "shift",
        "shiftMe",
        "alignTime",
        "power",
        "sqrt",
        "xcov",
        "periodogram",
        "MTMspectrum",
        "spectrogram",
        "plotVariability",
        "plotAllVariability",
        "plotPropsSet",
        "areDataLabelsEmpty",
        "isLabelPresent",
        "convertNamesToIndices",
        "clearPlotProps",
    ),
    "nstat.Trial": (
        "findMinSampleRate",
        "getAllLabels",
        "getDesignMatrix",
        "getNumHist",
        "getEnsCovMatrix",
        "getTrialPartition",
        "plotCovariates",
        "plotRaster",
        "toStructure",
        "fromStructure",
    ),
    "nstat.nstColl": (
        "psthBars",
        "estimateVarianceAcrossTrials",
        "ssglm",
    ),
    "nstat.Analysis": (
        "GLMFit",
        "RunAnalysisForNeuron",
        "RunAnalysisForAllNeurons",
        "KSPlot",
        "computeKSStats",
        "computeFitResidual",
        "plotFitResidual",
        "plotInvGausTrans",
        "plotSeqCorr",
        "plotCoeffs",
    ),
    "nstat.CIF": (
        "setSpikeTrain",
        "setHistory",
        "simulateCIF",
        "simulateCIFByThinning",
        "simulateCIFByThinningFromLambda",
        "evalGradient",
        "evalGradientLog",
        "evalJacobian",
        "evalJacobianLog",
        "evalGradientLDGamma",
        "evalJacobianLDGamma",
    ),
    "nstat.DecodingAlgorithms": (
        "PPDecode_predict",
        "PPDecode_update",
        "PPDecode_updateLinear",
        "PPDecodeFilterLinear",
        "PPDecodeFilter",
        "PP_fixedIntervalSmoother",
        "PPHybridFilterLinear",
        "PPHybridFilter",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_class_fidelity_audit(repo_root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if repo_root is None else repo_root.resolve()
    path = base / "parity" / "class_fidelity.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_public_symbol(dotted_name: str | None) -> Any | None:
    if not dotted_name:
        return None
    parts = [part for part in str(dotted_name).split(".") if part]
    if not parts:
        return None
    obj: Any = importlib.import_module(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _coerce_verified_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def row_runtime_symbol_verified(row: dict[str, Any]) -> bool:
    public_name = row.get("python_public_name")
    symbol = resolve_public_symbol(public_name)
    if symbol is None:
        return False
    required_members = EXPECTED_RUNTIME_MEMBERS.get(str(public_name), ())
    return all(callable(getattr(symbol, member, None)) for member in required_members)


def row_audit_symbol_verified(row: dict[str, Any]) -> bool:
    return _coerce_verified_flag(row.get("symbol_presence_verified"))


def iter_symbol_presence_mismatches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in payload.get("items", []):
        expected = row_runtime_symbol_verified(row)
        if row_audit_symbol_verified(row) != expected:
            mismatches.append(row)
    return mismatches


def summarize_symbol_presence(payload: dict[str, Any]) -> dict[str, int]:
    counts = {"verified": 0, "unverified": 0, "not_applicable": 0}
    for row in payload.get("items", []):
        if not row.get("python_public_name") or row.get("status") == "not_applicable":
            counts["not_applicable"] += 1
        elif row_runtime_symbol_verified(row):
            counts["verified"] += 1
        else:
            counts["unverified"] += 1
    return counts


__all__ = [
    "EXPECTED_RUNTIME_MEMBERS",
    "iter_symbol_presence_mismatches",
    "load_class_fidelity_audit",
    "resolve_public_symbol",
    "row_audit_symbol_verified",
    "row_runtime_symbol_verified",
    "summarize_symbol_presence",
]
