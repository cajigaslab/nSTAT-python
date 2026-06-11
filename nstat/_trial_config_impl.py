"""Canonical implementation of ``TrialConfig`` and ``ConfigCollection``.

Previously a section of :mod:`nstat.trial`; split out for readability.
``nstat.trial`` continues to re-export both classes for back-compat — all
existing import paths
(``from nstat.trial import TrialConfig``, ``from nstat.TrialConfig import
TrialConfig``, ``from nstat import TrialConfig``) still work.

These classes only reference :class:`nstat.trial.Trial` via duck-typed
method calls (``trial.setHistory(...)``, ``trial.resample(...)``, etc.) so
there is no runtime import dependency on ``trial.py``.  The string forward
reference ``"Trial"`` is preserved in type annotations.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # only for type checkers — no runtime import
    from .trial import Trial


def _is_empty_config_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return len(value) == 0
    return False


class TrialConfig:
    """Description of a single GLM fit configuration.

    A ``TrialConfig`` specifies which covariates, history, ensemble
    history, and sample rate to apply to a :class:`Trial` before fitting.
    Multiple ``TrialConfig`` objects are collected in a
    :class:`ConfigCollection` to run a batch of nested-model comparisons.

    Parameters
    ----------
    covMask : sequence of str or nested sequences, or None
        Covariate labels to include in the design matrix.
        ``'all'`` includes every covariate.
    sampleRate : float or None
        If provided, the trial is resampled to this rate before fitting.
    history : History or array_like or None
        Self-history specification (History object or window-times).
    ensCovHist : History or array_like or None
        Ensemble-history specification.
    ensCovMask : array_like or None
        Binary mask selecting which neighbours contribute ensemble history.
    covLag : array_like or None
        Covariate shift / lag specification.
    name : str
        Human-readable name for this configuration.
    """

    def __init__(
        self,
        covMask: Sequence[Sequence[str]] | Sequence[str] | None = None,
        sampleRate: float | None = None,
        history: object | None = None,
        ensCovHist: object | None = None,
        ensCovMask: object | None = None,
        covLag: object | None = None,
        name: str = "",
    ) -> None:
        """Construct a single GLM-fit configuration (Matlab ``TrialConfig``).

        Parameters
        ----------
        covMask : sequence of str, nested sequences, or None, optional
            Names of the covariates (and channels) to include in the
            design matrix.  ``None`` or ``[]`` (the empty list) is the
            "unset" sentinel and leaves the mask empty.  Use the literal
            string ``'all'`` to include every covariate in a
            :class:`~nstat.trial.CovariateCollection`.
        sampleRate : float or None, optional
            Target sample rate in **Hz** to resample the trial to
            before fitting.  ``None`` leaves the trial's existing rate
            unchanged.
        history : History, array_like, or None, optional
            Self-history specification.  May be a
            :class:`~nstat.history.History` object or a vector of window
            boundary times in **seconds**.
        ensCovHist : History, array_like, or None, optional
            Ensemble-history specification.
        ensCovMask : array_like or None, optional
            Binary mask selecting which neighbour neurons contribute
            ensemble-history terms.
        covLag : array_like or None, optional
            Per-covariate time shifts (covariate lags) in **seconds**.
        name : str, optional
            Human-readable name for this configuration (appears in
            :class:`~nstat.fit.FitResult` plots).  Default ``""``.

        Notes
        -----
        ``None`` arguments are normalised to the empty list ``[]`` to
        match the MATLAB sentinel convention recognised by
        :func:`_is_empty_config_value`.

        See Also
        --------
        ConfigCollection : Ordered collection used to compare configurations.
        Trial.setConfig : Apply a single ``TrialConfig`` to a trial.
        """
        self.covMask = [] if covMask is None else covMask
        self.sampleRate = [] if sampleRate is None else sampleRate
        self.history = [] if history is None else history
        self.ensCovHist = [] if ensCovHist is None else ensCovHist
        self.ensCovMask = [] if ensCovMask is None else ensCovMask
        self.covLag = [] if covLag is None else covLag
        self.name = str(name)

    @property
    def covariate_names(self) -> list[str]:
        """Return the name of each covariate group in the mask."""
        if not self.covMask:
            return []
        names: list[str] = []
        for item in self.covMask:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Sequence) and item:
                names.append(str(item[0]))
        return names

    def getName(self) -> str:
        """Return this configuration's human-readable name."""
        return self.name

    def setName(self, name: str) -> None:
        """Set this configuration's human-readable name."""
        self.name = str(name)

    def setConfig(self, trial: "Trial") -> None:
        """Apply this configuration to a Trial (in place).

        Sets the covariate mask, history, ensemble history, sample rate,
        and covariate lag on the trial.
        """
        if not _is_empty_config_value(self.history):
            trial.setHistory(self.history)
        else:
            trial.resetHistory()

        if not _is_empty_config_value(self.sampleRate):
            sampleRate = float(self.sampleRate)
            if round(trial.sampleRate, 3) != round(sampleRate, 3):
                trial.resample(sampleRate)

        trial.setCovMask(self.covMask)

        if not _is_empty_config_value(self.covLag):
            trial.shiftCovariates(self.covLag)

        if not _is_empty_config_value(self.ensCovHist):
            trial.setEnsCovHist(self.ensCovHist)
            trial.setEnsCovMask(self.ensCovMask)
        else:
            trial.setEnsCovHist()
            trial.setEnsCovMask()

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``TrialConfig.toStructure``)."""
        return {
            "covMask": self.covMask,
            "sampleRate": self.sampleRate,
            "history": self.history,
            "ensCovHist": self.ensCovHist,
            "ensCovMask": self.ensCovMask,
            "covLag": self.covLag,
            "name": self.name,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "TrialConfig":
        """Reconstruct from a dict produced by :meth:`toStructure`.

        .. note:: Follows Matlab's omission of ``ensCovMask``."""
        # MATLAB's `TrialConfig.fromStructure` omits `ensCovMask` and shifts
        # the remaining trailing arguments left by one position.
        return TrialConfig(
            structure.get("covMask"),
            structure.get("sampleRate"),
            structure.get("history"),
            structure.get("ensCovHist"),
            structure.get("covLag"),
            structure.get("name", ""),
        )


class ConfigCollection:
    """Ordered collection of :class:`TrialConfig` objects.

    Used by :class:`Analysis` to iterate over multiple model
    specifications (e.g. baseline, baseline + stimulus,
    baseline + stimulus + history) and compare their fits.

    Parameters
    ----------
    configs : TrialConfig, sequence of TrialConfig, or None
        Initial configuration(s).  ``None`` creates a single
        ``"Empty Config"`` entry (Matlab parity).
    """

    def __init__(self, configs: Sequence[TrialConfig] | TrialConfig | str | None = None) -> None:
        """Construct an ordered collection of GLM-fit configurations (Matlab ``ConfigColl``).

        Parameters
        ----------
        configs : TrialConfig, sequence of TrialConfig, str, or None, optional
            Initial configuration(s).

            - A single :class:`TrialConfig` becomes the first entry.
            - A sequence (list/tuple) is iterated and each entry added.
            - ``None`` (default) or an empty sequence creates a single
              ``"Empty Config"`` placeholder entry — matches MATLAB
              ``ConfigColl()`` which routes through ``addConfig([])``.

        Raises
        ------
        TypeError
            If *configs* contains an unsupported entry type.

        Notes
        -----
        Empty-config placeholders (strings/sequences) are tracked
        separately from real :class:`TrialConfig` instances — see
        :attr:`configs` for the filtered list.

        See Also
        --------
        TrialConfig : Single GLM-fit configuration.
        Analysis.RunAnalysisForAllNeurons : Iterates over a ``ConfigCollection``.
        """
        self.numConfigs = 0
        self.configNames: list[str] = []
        self.configArray: list[TrialConfig | str | list[str]] = []
        # MATLAB ConfigColl() routes through addConfig([]), which creates
        # a single "Empty Config" entry by default.
        self.addConfig([] if configs is None else configs)

    @property
    def configs(self) -> list[TrialConfig]:
        """List of actual ``TrialConfig`` entries (excludes empty placeholders)."""
        return [cfg for cfg in self.configArray if isinstance(cfg, TrialConfig)]

    def add_config(self, cfg: TrialConfig) -> None:
        """Pythonic alias for :meth:`addConfig`."""
        self.addConfig(cfg)

    def addConfig(self, cfg: Sequence[TrialConfig] | TrialConfig | str | None) -> None:
        """Append one or more configurations to this collection."""
        if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes, TrialConfig, np.ndarray)):
            if len(cfg) == 0:
                self.numConfigs += 1
                self.configNames.append("Empty Config")
                self.configArray.append(["Empty Config"])
                return
            for item in cfg:
                self.addConfig(item)
            return
        if _is_empty_config_value(cfg):
            self.numConfigs += 1
            self.configNames.append("Empty Config")
            self.configArray.append(["Empty Config"])
            return
        if isinstance(cfg, TrialConfig):
            self.numConfigs += 1
            self.configArray.append(cfg)
            self.setConfigNames(cfg.name, [self.numConfigs])
            return
        if isinstance(cfg, str):
            # MATLAB's string branch dereferences tcObj.name and errors.
            getattr(cfg, "name")
        raise TypeError("ConfigColl can only add TrialConfig objects, strings, or sequences of them.")

    def get_config(self, idx: int) -> TrialConfig | str | list[str]:
        """Return a config by 0-based index (Pythonic API)."""
        if idx < 0 or idx >= self.numConfigs:
            raise IndexError("ConfigCollection index out of bounds (0-based indexing).")
        return self.configArray[idx]

    def getConfig(self, idx: int):
        """Return a config by 0-based index."""
        if idx < 0 or idx >= self.numConfigs:
            raise IndexError("Index Out of Bounds")
        return self.configArray[idx]

    def setConfig(self, trial: "Trial", index: int) -> None:
        """Apply configuration *index* (0-based) to the given Trial."""
        config = self.getConfig(index)
        if isinstance(config, TrialConfig):
            config.setConfig(trial)
            return
        raise ValueError("Cannot Set Empty Configs")

    def getConfigNames(self, index: Sequence[int] | None = None) -> list[str]:
        """Return the names for selected configs (0-based), or all if *index* is ``None``."""
        if index is None:
            index = list(range(self.numConfigs))
        out: list[str] = []
        for i in index:
            if i < 0 or i >= self.numConfigs:
                raise IndexError("Index Out of Bounds")
            tempName = self.configNames[i]
            out.append(tempName if tempName else f"Fit {i + 1}")
        return out

    def setConfigNames(self, names, index: Sequence[int] | None = None) -> None:
        """Set the human-readable names for configs at 0-based *index* positions."""
        if index is None:
            index = list(range(self.numConfigs))
        if isinstance(names, str):
            if len(index) != 1:
                raise ValueError("If specifying a single name, index must be length 1.")
            target = int(index[0])
            while len(self.configNames) < self.numConfigs:
                self.configNames.append("")
            self.configNames[target] = names if names else f"Fit {self.numConfigs}"
            return
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            if len(index) != len(names):
                raise ValueError("If specifying multiple names, names and index must match in length.")
            for idx, name in zip(index, names):
                self.setConfigNames(str(name), [int(idx)])
            return
        raise TypeError("names must be a string or sequence of strings.")

    def getSubsetConfigs(self, subset: Sequence[int]) -> "ConfigCollection":
        """Return a new collection containing only configs at 1-based *subset* indices."""
        tempconfigs = [self.getConfig(int(i)) for i in subset]
        return ConfigCollection(tempconfigs)

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``ConfigColl.toStructure``)."""
        structure = {
            "numConfigs": self.numConfigs,
            "configNames": list(self.configNames),
            "configArray": [],
        }
        for cfg in self.configArray:
            if isinstance(cfg, TrialConfig):
                structure["configArray"].append(cfg.toStructure())
            else:
                structure["configArray"].append(cfg)
        return structure

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "ConfigCollection":
        """Reconstruct from a dict produced by :meth:`toStructure`."""
        configs = []
        for row in structure.get("configArray", []):
            if isinstance(row, dict):
                configs.append(TrialConfig.fromStructure(row))
            else:
                configs.append(row)
        return ConfigCollection(configs)


__all__ = ["TrialConfig", "ConfigCollection"]
