from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import Covariate, nspikeTrain
from .events import Events


def _is_string_sequence(values: object) -> bool:
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Sequence):
        return False
    return all(isinstance(item, str) for item in values)


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


def _copy_covariate(cov: Covariate) -> Covariate:
    copied = cov.copySignal()
    if not isinstance(copied, Covariate):
        copied = Covariate(
            copied.time,
            copied.data,
            copied.name,
            copied.xlabelval,
            copied.xunits,
            copied.yunits,
            copied.dataLabels,
            copied.plotProps,
        )
    return copied


class CovariateCollection:
    """MATLAB-style CovColl implementation with collection-level masks and timing."""

    def __init__(self, covariates: Sequence[Covariate] | Covariate | None = None, *more_covariates: Covariate) -> None:
        self.covArray: list[Covariate] = []
        self.covDimensions: list[int] = []
        self.numCov = 0
        self.minTime = float("inf")
        self.maxTime = float("-inf")
        self.covMask: list[np.ndarray] = []
        self.covShift = 0.0
        self.sampleRate = float("nan")
        self.originalSampleRate: float | None = None
        self.originalMinTime: float | None = None
        self.originalMaxTime: float | None = None
        if covariates is not None:
            self.addToColl(covariates)
        for cov in more_covariates:
            self.addToColl(cov)

    @property
    def covariates(self) -> list[Covariate]:
        return [self.getCov(i) for i in range(1, self.numCov + 1)]

    @property
    def names(self) -> list[str]:
        return [cov.name for cov in self.covArray]

    def _capture_originals_if_needed(self) -> None:
        if self.numCov == 0:
            return
        if self.originalSampleRate is None:
            self.originalSampleRate = float(self.sampleRate)
        if self.originalMinTime is None:
            self.originalMinTime = float(self.minTime)
        if self.originalMaxTime is None:
            self.originalMaxTime = float(self.maxTime)

    def _refresh_summary(self) -> None:
        self.numCov = len(self.covArray)
        self.covDimensions = [cov.dimension for cov in self.covArray]
        if self.numCov == 0:
            self.minTime = float("inf")
            self.maxTime = float("-inf")
            self.sampleRate = float("nan")
            self.covMask = []
            return

        if len(self.covMask) != self.numCov:
            self.covMask = [np.ones(cov.dimension, dtype=int) for cov in self.covArray]
        else:
            normalized_mask: list[np.ndarray] = []
            for cov, mask in zip(self.covArray, self.covMask):
                arr = np.asarray(mask, dtype=int).reshape(-1)
                if arr.size != cov.dimension:
                    arr = np.ones(cov.dimension, dtype=int)
                normalized_mask.append(arr)
            self.covMask = normalized_mask

        if not np.isfinite(self.sampleRate):
            self.sampleRate = self.findMaxSampleRate()
        self.minTime = self.findMinTime() + float(self.covShift)
        self.maxTime = self.findMaxTime() + float(self.covShift)
        self._capture_originals_if_needed()

    def _covariate_from_identifier(self, identifier: int | str) -> int:
        if isinstance(identifier, str):
            return self.getCovIndFromName(identifier)
        index = int(identifier)
        if index < 1 or index > self.numCov:
            raise IndexError("Covariate index out of bounds (1-based indexing).")
        return index

    def _apply_collection_state(self, cov: Covariate, index: int) -> Covariate:
        out = _copy_covariate(cov)
        if self.covShift != 0:
            out.time = out.time + float(self.covShift)
            out.minTime = float(np.min(out.time))
            out.maxTime = float(np.max(out.time))
        if np.isfinite(self.sampleRate) and self.sampleRate > 0 and round(out.sampleRate, 3) != round(self.sampleRate, 3):
            out = out.resample(self.sampleRate)
        if np.isfinite(self.minTime) and np.isfinite(self.maxTime) and out.time.size > 0:
            out = out.getSigInTimeWindow(self.minTime, self.maxTime, holdVals=1)
        out.setMask(self.covMask[index - 1])
        return out

    def add(self, covariate: Covariate) -> None:
        self.addToColl(covariate)

    def addCovariate(self, covariate: Covariate) -> None:
        self.addToColl(covariate)

    def addToColl(self, covariates: Sequence[Covariate] | Covariate | "CovariateCollection" | None) -> None:
        if covariates is None:
            return
        if isinstance(covariates, CovariateCollection):
            for cov in covariates.covArray:
                self.addToColl(cov)
            return
        if isinstance(covariates, Covariate):
            self.covArray.append(_copy_covariate(covariates))
            self.covMask.append(np.ones(covariates.dimension, dtype=int))
            self._refresh_summary()
            return
        if isinstance(covariates, Sequence) and not isinstance(covariates, (str, bytes, np.ndarray)):
            for cov in covariates:
                self.addToColl(cov)
            return
        raise TypeError("CovColl can only add Covariate instances or sequences of Covariates.")

    def removeCovariate(self, identifier: int | str) -> None:
        index = self._covariate_from_identifier(identifier)
        del self.covArray[index - 1]
        del self.covMask[index - 1]
        self._refresh_summary()

    def get(self, name: str) -> Covariate:
        return self.getCov(name)

    def getCov(self, identifier: int | str | Sequence[int] | Sequence[str]):
        if isinstance(identifier, str):
            return self._apply_collection_state(self.covArray[self.getCovIndFromName(identifier) - 1], self.getCovIndFromName(identifier))
        if isinstance(identifier, Sequence) and not isinstance(identifier, (str, bytes, np.ndarray)):
            if _is_string_sequence(identifier):
                return [self.getCov(item) for item in identifier]
            return [self.getCov(int(item)) for item in identifier]
        if isinstance(identifier, np.ndarray) and identifier.ndim > 0:
            return [self.getCov(int(item)) for item in identifier.reshape(-1)]
        index = self._covariate_from_identifier(identifier)
        return self._apply_collection_state(self.covArray[index - 1], index)

    def getCovIndFromName(self, name: str) -> int:
        for idx, cov in enumerate(self.covArray, start=1):
            if cov.name == name:
                return idx
        raise KeyError(f"Covariate '{name}' not found")

    def getCovIndicesFromNames(self, name: Sequence[str] | str):
        if isinstance(name, str):
            return self.getCovIndFromName(name)
        return [self.getCovIndFromName(item) for item in name]

    def findMinTime(self) -> float:
        if self.numCov == 0:
            return float("inf")
        return float(min(cov.minTime for cov in self.covArray))

    def findMaxTime(self) -> float:
        if self.numCov == 0:
            return float("-inf")
        return float(max(cov.maxTime for cov in self.covArray))

    def findMaxSampleRate(self) -> float:
        if self.numCov == 0:
            return float("nan")
        return float(max(cov.sampleRate for cov in self.covArray if np.isfinite(cov.sampleRate)))

    def setMinTime(self, minTime: float | None = None) -> None:
        if minTime is None:
            minTime = self.findMinTime() + float(self.covShift)
        self.minTime = float(minTime)

    def setMaxTime(self, maxTime: float | None = None) -> None:
        if maxTime is None:
            maxTime = self.findMaxTime() + float(self.covShift)
        self.maxTime = float(maxTime)

    def restrictToTimeWindow(self, wMin: float, wMax: float) -> None:
        self.setMinTime(wMin)
        self.setMaxTime(wMax)

    def setSampleRate(self, sampleRate: float) -> None:
        if self.originalSampleRate is None and np.isfinite(self.sampleRate):
            self.originalSampleRate = float(self.sampleRate)
        self.sampleRate = float(sampleRate)
        self.enforceSampleRate()

    def resample(self, sampleRate: float) -> None:
        self.setSampleRate(sampleRate)

    def enforceSampleRate(self) -> None:
        if not np.isfinite(self.sampleRate) or self.sampleRate <= 0:
            self.sampleRate = self.findMaxSampleRate()

    def resetMask(self) -> None:
        self.covMask = [np.ones(cov.dimension, dtype=int) for cov in self.covArray]

    def getCovDataMask(self, identifier: int | str) -> np.ndarray:
        index = self._covariate_from_identifier(identifier)
        return np.asarray(self.covMask[index - 1], dtype=int).copy()

    def isCovMaskSet(self) -> bool:
        return any(np.any(mask == 0) for mask in self.covMask)

    def flattenCovMask(self) -> np.ndarray:
        if not self.covMask:
            return np.array([], dtype=int)
        return np.concatenate([np.asarray(mask, dtype=int).reshape(-1) for mask in self.covMask])

    def getSelectorFromMasks(self, covMask: list[np.ndarray] | None = None) -> list[list[int]]:
        current = self.covMask if covMask is None else covMask
        selector: list[list[int]] = []
        for mask in current:
            active = np.flatnonzero(np.asarray(mask, dtype=int) == 1) + 1
            selector.append(active.astype(int).tolist())
        return selector

    def _selector_cell_from_names(self, dataSelector: Sequence[Any]) -> list[list[int]]:
        selectorCell = [[] for _ in range(self.numCov)]
        if not dataSelector:
            return selectorCell
        if isinstance(dataSelector[0], str):
            covName = str(dataSelector[0])
            covIndex = self.getCovIndFromName(covName)
            currCov = self.getCov(covIndex)
            if len(dataSelector) == 1:
                selectorCell[covIndex - 1] = list(range(1, currCov.dimension + 1))
            else:
                selectorCell[covIndex - 1] = currCov.getIndicesFromLabels([str(v) for v in dataSelector[1:]])
            return selectorCell

        for item in dataSelector:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                raise ValueError("dataSelector specified incorrectly")
            parsed = list(item)
            if not parsed:
                continue
            covName = str(parsed[0])
            covIndex = self.getCovIndFromName(covName)
            currCov = self.getCov(covIndex)
            if len(parsed) == 1:
                selectorCell[covIndex - 1] = list(range(1, currCov.dimension + 1))
            else:
                selectorCell[covIndex - 1] = currCov.getIndicesFromLabels([str(v) for v in parsed[1:]])
        return selectorCell

    def generateSelectorCell(self, dataSelector) -> list[list[int]]:
        if dataSelector is None:
            return [[] for _ in range(self.numCov)]
        if isinstance(dataSelector, str):
            return self._selector_cell_from_names([dataSelector])
        if isinstance(dataSelector, np.ndarray):
            dataSelector = dataSelector.tolist()
        if not isinstance(dataSelector, Sequence) or isinstance(dataSelector, (str, bytes)):
            raise ValueError("dataSelector specified incorrectly")
        values = list(dataSelector)
        if not values:
            return [[] for _ in range(self.numCov)]
        looks_like_numeric_selector = self.numCov == len(values) and all(
            isinstance(item, np.ndarray)
            or (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and all(not isinstance(v, str) for v in item)
            )
            or isinstance(item, (int, np.integer, float, np.floating))
            for item in values
        )
        if looks_like_numeric_selector:
            selectorCell: list[list[int]] = []
            for item in values:
                if isinstance(item, np.ndarray):
                    selectorCell.append(np.asarray(item, dtype=int).reshape(-1).tolist())
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    selectorCell.append([int(v) for v in item])
                else:
                    selectorCell.append([int(item)])
            return selectorCell
        return self._selector_cell_from_names(values)

    def _selector_to_cov_mask(self, selectorCell: list[list[int]]) -> list[np.ndarray]:
        if len(selectorCell) != self.numCov:
            raise ValueError("selectorCell size must match number of covariates.")
        masks: list[np.ndarray] = []
        for cov, selector in zip(self.covArray, selectorCell):
            mask = np.zeros(cov.dimension, dtype=int)
            if selector:
                arr = np.asarray(selector, dtype=int).reshape(-1)
                if arr.size == cov.dimension and np.all(np.isin(arr, [0, 1])):
                    mask = arr.astype(int)
                else:
                    if np.any(arr < 1) or np.any(arr > cov.dimension):
                        raise IndexError("Covariate selector index out of bounds.")
                    mask[arr - 1] = 1
            masks.append(mask)
        return masks

    def setMasksFromSelector(self, selectorCell: list[list[int]]) -> None:
        self.covMask = self._selector_to_cov_mask(selectorCell)

    def setMask(self, cellInput) -> None:
        if isinstance(cellInput, str) and cellInput == "all":
            self.resetMask()
            return
        selectorCell = self.generateSelectorCell(cellInput)
        self.setMasksFromSelector(selectorCell)

    def nActCovar(self) -> int:
        return int(sum(1 for selector in self.getSelectorFromMasks() if selector))

    def maskAwayCov(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        identifiers = identifier
        if isinstance(identifier, (int, str)):
            identifiers = [identifier]
        for item in identifiers:
            index = self._covariate_from_identifier(item)
            self.covMask[index - 1] = np.zeros(self.covArray[index - 1].dimension, dtype=int)

    def maskAwayOnlyCov(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        self.resetMask()
        self.maskAwayCov(identifier)

    def maskAwayAllExcept(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        if isinstance(identifier, (int, str)):
            keep = {self._covariate_from_identifier(identifier)}
        else:
            keep = {self._covariate_from_identifier(item) for item in identifier}
        for idx, cov in enumerate(self.covArray, start=1):
            if idx not in keep:
                self.covMask[idx - 1] = np.zeros(cov.dimension, dtype=int)

    def setCovShift(self, deltaT: float, identifier=None) -> "CovariateCollection":
        self.covShift = float(deltaT)
        if np.isfinite(self.minTime):
            self.minTime = float(self.minTime + self.covShift)
        if np.isfinite(self.maxTime):
            self.maxTime = float(self.maxTime + self.covShift)
        return self

    def resetCovShift(self) -> None:
        self.covShift = 0.0
        self.setMinTime()
        self.setMaxTime()

    def restoreToOriginal(self) -> None:
        self.covShift = 0.0
        if self.originalSampleRate is not None:
            self.sampleRate = float(self.originalSampleRate)
        else:
            self.sampleRate = self.findMaxSampleRate()
        self.setMinTime(self.findMinTime())
        self.setMaxTime(self.findMaxTime())
        self.resetMask()

    def plot(self, *_, handle=None, **__):
        selected = [idx for idx in range(1, self.numCov + 1)]
        fig = handle if handle is not None else plt.figure(figsize=(8.5, max(2.5, 2.2 * max(len(selected), 1))))
        fig.clear()
        axes = fig.subplots(len(selected), 1, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for ax, cov_index in zip(axes.reshape(-1), selected, strict=False):
            cov = self.getCov(cov_index)
            cov.plot(handle=ax)
            ax.set_title(cov.name)
        fig.tight_layout()
        return fig

    def getAllCovLabels(self) -> list[str]:
        labels: list[str] = []
        for index in range(1, self.numCov + 1):
            labels.extend(self.getCov(index).dataLabels)
        return labels

    def getCovLabelsFromMask(self) -> list[str]:
        labels: list[str] = []
        for index in range(1, self.numCov + 1):
            cov = self.getCov(index)
            mask = self.covMask[index - 1]
            labels.extend([label for keep, label in zip(mask, cov.dataLabels) if keep == 1])
        return labels

    def matrixWithTime(self, repType: str = "standard", dataSelector=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self.numCov == 0:
            raise ValueError("CovariateCollection is empty")
        if dataSelector is None:
            selectorCell = self.getSelectorFromMasks() if self.isCovMaskSet() else [
                list(range(1, self.getCov(i).dimension + 1)) for i in range(1, self.numCov + 1)
            ]
        else:
            selectorCell = self.generateSelectorCell(dataSelector)

        active_cov = [i + 1 for i, selector in enumerate(selectorCell) if selector]
        if not active_cov:
            time = self.getCov(1).time
            return time.copy(), np.zeros((time.size, 0), dtype=float), []

        time = self.getCov(active_cov[0]).getSigRep(repType).time
        parts: list[np.ndarray] = []
        labels: list[str] = []
        for covIndex in active_cov:
            cov = self.getCov(covIndex).getSigRep(repType)
            selector = selectorCell[covIndex - 1]
            data = cov.dataToMatrix(selector)
            endInd = min(time.size, data.shape[0])
            block = np.zeros((time.size, data.shape[1]), dtype=float)
            block[:endInd, :] = data[:endInd, :]
            parts.append(block)
            labels.extend([cov.dataLabels[idx - 1] for idx in selector])
        return time.copy(), np.hstack(parts) if parts else np.zeros((time.size, 0), dtype=float), labels

    def dataToMatrix(self, repType: str | Sequence[str] | None = "standard", dataSelector=None, *_) -> np.ndarray:
        if repType not in {"standard", "zero-mean"}:
            dataSelector = repType
            repType = "standard"
        _, matrix, _ = self.matrixWithTime(str(repType), dataSelector)
        return matrix


class SpikeTrainCollection:
    """MATLAB-style nstColl implementation."""

    def __init__(self, trains: Sequence[nspikeTrain] | nspikeTrain | None = None) -> None:
        self.nstrain: list[nspikeTrain] = []
        self.numSpikeTrains = 0
        self.minTime = float("inf")
        self.maxTime = float("-inf")
        self.sampleRate = float("-inf")
        self.neuronMask = np.array([], dtype=int)
        self.neighbors: np.ndarray | list[list[int]] = []
        if trains is not None:
            self.addToColl(trains)

    @property
    def num_spike_trains(self) -> int:
        return self.numSpikeTrains

    @property
    def uniqueNeuronNames(self) -> list[str]:
        return self.getUniqueNSTnames()

    def __iter__(self):
        for tr in self.nstrain:
            yield tr

    def _refresh_summary(self) -> None:
        self.numSpikeTrains = len(self.nstrain)
        if self.numSpikeTrains == 0:
            self.minTime = float("inf")
            self.maxTime = float("-inf")
            self.sampleRate = float("-inf")
            self.neuronMask = np.array([], dtype=int)
            self.neighbors = []
            return
        self.minTime = float(min(train.minTime for train in self.nstrain))
        self.maxTime = float(max(train.maxTime for train in self.nstrain))
        self.sampleRate = self.findMaxSampleRate()
        if self.neuronMask.size != self.numSpikeTrains:
            self.neuronMask = np.ones(self.numSpikeTrains, dtype=int)

    def addSingleSpikeToColl(self, nst: nspikeTrain) -> None:
        self.nstrain.append(nst.nstCopy())
        self._refresh_summary()

    def addToColl(self, nst: Sequence[nspikeTrain] | nspikeTrain | "SpikeTrainCollection") -> None:
        if isinstance(nst, SpikeTrainCollection):
            for train in nst.nstrain:
                self.addSingleSpikeToColl(train)
            return
        if isinstance(nst, nspikeTrain):
            self.addSingleSpikeToColl(nst)
            return
        if isinstance(nst, Sequence) and not isinstance(nst, (str, bytes, np.ndarray)):
            for item in nst:
                if not isinstance(item, nspikeTrain):
                    raise TypeError("nstColl requires a sequence of nspikeTrain objects.")
                self.addSingleSpikeToColl(item)
            return
        raise TypeError("nstColl can only add nspikeTrain instances or sequences of nspikeTrain.")

    def merge(self, nstColl2: "SpikeTrainCollection") -> "SpikeTrainCollection":
        self.addToColl(nstColl2)
        return self

    def get_nst(self, idx: int) -> nspikeTrain:
        if idx < 0 or idx >= self.numSpikeTrains:
            raise IndexError("SpikeTrainCollection index out of bounds (0-based indexing).")
        return self.nstrain[idx]

    def getNST(self, idx) -> nspikeTrain | list[nspikeTrain]:
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes, np.ndarray)):
            return [self.getNST(int(item)) for item in idx]
        index = int(idx)
        if index < 1 or index > self.numSpikeTrains:
            raise IndexError("nstColl index out of bounds (1-based indexing).")
        return self.nstrain[index - 1]

    def getNSTnames(self) -> list[str]:
        return [train.name for train in self.nstrain]

    def getUniqueNSTnames(self) -> list[str]:
        names = [name for name in self.getNSTnames() if name]
        return list(dict.fromkeys(names))

    def getNSTIndicesFromName(self, name: Sequence[str] | str):
        if isinstance(name, str):
            matches = [i + 1 for i, value in enumerate(self.getNSTnames()) if value == name]
            if not matches:
                raise KeyError(f"Neuron '{name}' not found")
            return matches if len(matches) > 1 else matches[0]
        return [self.getNSTIndicesFromName(item) for item in name]

    def toSpikeTrain(
        self,
        selectorArray: Sequence[int] | Sequence[str] | str | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
        windowTimes: Sequence[float] | None = None,
    ) -> nspikeTrain:
        if self.numSpikeTrains == 0:
            raise ValueError("nstColl.toSpikeTrain requires at least one spike train")

        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime

        if selectorArray is None:
            selector = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(1, self.numSpikeTrains + 1))
        elif isinstance(selectorArray, str) or _is_string_sequence(selectorArray):
            resolved = self.getNSTIndicesFromName(selectorArray)
            if isinstance(resolved, list):
                selector = [int(item) if not isinstance(item, list) else int(item[0]) for item in resolved]
            else:
                selector = [int(resolved)]
        else:
            selector = [int(item) for item in selectorArray]

        if not selector:
            raise ValueError("selectorArray resolved to no spike trains")

        delta = 1.0 / max(float(self.sampleRate), 1e-12)
        spike_times: list[float] = []
        offset = 0.0
        selected_trains = [self.getNST(index) for index in selector]
        name = selected_trains[0].name

        if windowTimes is None or len(windowTimes) == 0:
            for idx, train in enumerate(selected_trains):
                if idx == 0:
                    spike_times.extend(np.asarray(train.spikeTimes, dtype=float).reshape(-1).tolist())
                else:
                    prev_train = selected_trains[idx - 1]
                    offset += float(prev_train.maxTime) + float(delta)
                    if np.asarray(train.spikeTimes).size:
                        spike_times.extend((np.asarray(train.spikeTimes, dtype=float).reshape(-1) + offset).tolist())
        else:
            window_arr = np.asarray(windowTimes, dtype=float).reshape(-1)
            if len(selector) != window_arr.size - 1:
                raise ValueError("Window Times must be 1 row longer than selectorArray")
            for idx, train in enumerate(selected_trains):
                local_min = float(window_arr[idx])
                delta_tw = float(window_arr[idx + 1] - local_min)
                if np.asarray(train.spikeTimes).size:
                    spike_times.extend((np.asarray(train.spikeTimes, dtype=float).reshape(-1) * delta_tw + local_min).tolist())

        collapsed = nspikeTrain(spike_times, name, delta, minTime, float(maxTime) * len(selector), "time", "s", "", "", -1)
        collapsed.setName(name)
        collapsed.setMinTime(float(minTime))
        collapsed.setMaxTime(float(maxTime) * len(selector))
        collapsed.resample(1.0 / max(delta, 1e-12))
        return collapsed

    def setMinTime(self, value: float | None = None) -> None:
        if value is None:
            value = self.minTime
        for train in self.nstrain:
            train.setMinTime(float(value))
        self.minTime = float(value)

    def setMaxTime(self, value: float | None = None) -> None:
        if value is None:
            value = self.maxTime
        for train in self.nstrain:
            train.setMaxTime(float(value))
        self.maxTime = float(value)

    def resample(self, sampleRate: float) -> None:
        self.sampleRate = float(sampleRate)
        for train in self.nstrain:
            train.resample(sampleRate)

    def findMaxSampleRate(self) -> float:
        if self.numSpikeTrains == 0:
            return float("-inf")
        return float(max(train.sampleRate for train in self.nstrain))

    def setMask(self, mask: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(mask, dtype=int).reshape(-1)
        if arr.size == self.numSpikeTrains and np.all(np.isin(arr, [0, 1])):
            self.setNeuronMask(arr)
            return
        self.setNeuronMaskFromInd(arr)

    def setNeuronMaskFromInd(self, mask: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(mask, dtype=int).reshape(-1)
        newMask = np.zeros(self.numSpikeTrains, dtype=int)
        if arr.size:
            if np.any(arr < 1) or np.any(arr > self.numSpikeTrains):
                raise IndexError("Neuron index out of bounds.")
            newMask[arr - 1] = 1
        self.setNeuronMask(newMask)

    def setNeuronMask(self, mask: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(mask, dtype=int).reshape(-1)
        if arr.size != self.numSpikeTrains:
            raise ValueError("neuronMask length must match number of spike trains.")
        self.neuronMask = arr.astype(int)

    def resetMask(self) -> None:
        self.neuronMask = np.ones(self.numSpikeTrains, dtype=int)

    def getIndFromMask(self) -> list[int]:
        return (np.flatnonzero(self.neuronMask == 1) + 1).astype(int).tolist()

    def getIndFromMaskMinusOne(self, neuron: int) -> list[int]:
        return [idx for idx in self.getIndFromMask() if idx != int(neuron)]

    def isNeuronMaskSet(self) -> bool:
        return bool(np.any(self.neuronMask == 0))

    def setNeighbors(self, neighborArray: Sequence[Sequence[int]] | np.ndarray | None = None) -> None:
        if neighborArray is None:
            if self.numSpikeTrains == 0:
                self.neighbors = []
                return
            matrix = np.zeros((self.numSpikeTrains, max(self.numSpikeTrains - 1, 0)), dtype=int)
            for i in range(self.numSpikeTrains):
                neighbors = [idx for idx in range(1, self.numSpikeTrains + 1) if idx != (i + 1)]
                if neighbors:
                    matrix[i, : len(neighbors)] = neighbors
            self.neighbors = matrix
            return
        arr = np.asarray(neighborArray, dtype=int)
        if arr.ndim != 2 or arr.shape[0] != self.numSpikeTrains:
            raise ValueError("Neighbor Array is not of appropriate dimensions")
        self.neighbors = arr

    def areNeighborsSet(self) -> bool:
        return np.size(self.neighbors) > 0

    def getNeighbors(self, neuronNum: int | Sequence[int]):
        if isinstance(neuronNum, Sequence) and not isinstance(neuronNum, (str, bytes, np.ndarray)):
            rows = [self.getNeighbors(int(item)) for item in neuronNum]
            if rows and all(len(row) == len(rows[0]) for row in rows):
                return np.asarray(rows, dtype=int)
            return rows
        neuron_idx = int(neuronNum)
        if not self.areNeighborsSet():
            self.setNeighbors()
        if isinstance(self.neighbors, list):
            row = list(self.neighbors[neuron_idx - 1])
        else:
            row = np.asarray(self.neighbors[neuron_idx - 1], dtype=int).reshape(-1).tolist()
        available = set(self.getIndFromMaskMinusOne(neuron_idx))
        return [value for value in row if value in available and value > 0]

    def getMaxBinSizeBinary(self) -> float:
        selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(1, self.numSpikeTrains + 1))
        if not selectorArray:
            return np.inf
        values = [self.getNST(index).getMaxBinSizeBinary() for index in selectorArray]
        return float(np.min(values))

    def dataToMatrix(
        self,
        selectorArray: Sequence[int] | Sequence[str] | str | None = None,
        binwidth: float | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> np.ndarray:
        if self.numSpikeTrains == 0:
            return np.zeros((0, 0), dtype=float)
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        if binwidth is None:
            binwidth = 1.0 / self.sampleRate
        if selectorArray is None:
            selector = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(1, self.numSpikeTrains + 1))
        elif isinstance(selectorArray, str) or _is_string_sequence(selectorArray):
            resolved = self.getNSTIndicesFromName(selectorArray)
            if isinstance(resolved, list):
                selector = [int(item) if not isinstance(item, list) else int(item[0]) for item in resolved]
            else:
                selector = [int(resolved)]
        else:
            selector = [int(item) for item in selectorArray]
        if not selector:
            testSig = self.getNST(1).getSigRep(binwidth, minTime, maxTime)
            return np.zeros((testSig.dataToMatrix().shape[0], 0), dtype=float)
        testSig = self.getNST(selector[0]).getSigRep(binwidth, minTime, maxTime)
        dataMat = np.zeros((testSig.dataToMatrix().shape[0], len(selector)), dtype=float)
        for idx, neuron in enumerate(selector):
            sig = self.getNST(neuron).getSigRep(binwidth, minTime, maxTime)
            dataMat[:, idx] = sig.dataToMatrix().reshape(-1)
        return dataMat

    def getEnsembleNeuronCovariates(self, neuronNum: int = 1, neighborIndex=None, windowTimes=None):
        if neighborIndex is None:
            allNeighbors = self.getNeighbors(neuronNum)
        else:
            allNeighbors = [int(item) for item in np.asarray(neighborIndex, dtype=int).reshape(-1)]
        if windowTimes is None:
            windowTimes = [0.0, 0.001]
        from .history import History

        histObj = windowTimes if isinstance(windowTimes, History) else History(windowTimes)
        ensembleCovariates = histObj.computeHistory(self.getNST(list(range(1, self.numSpikeTrains + 1))))
        ensembleCovariates.maskAwayAllExcept(allNeighbors)
        self.addNeuronNamesToEnsCovColl(ensembleCovariates)
        return ensembleCovariates

    def addNeuronNamesToEnsCovColl(self, ensembleCovariates: CovariateCollection) -> None:
        for i in range(1, ensembleCovariates.numCov + 1):
            tempCov = ensembleCovariates.covArray[i - 1]
            name = self.getNST(i).name
            if not name:
                name = str(i)
            dataLabels = [f"{name}:{label}" if label else str(name) for label in tempCov.dataLabels]
            tempCov.setDataLabels(dataLabels)

    def restoreToOriginal(self, rMask: int = 0) -> None:
        for train in self.nstrain:
            train.restoreToOriginal()
        self._refresh_summary()
        self.sampleRate = self.findMaxSampleRate()
        self.resample(self.sampleRate)
        if rMask == 1:
            self.resetMask()

    def plot(self, *_, handle=None, **__):
        selected = self.getIndFromMask()
        if not selected:
            selected = list(range(1, self.numSpikeTrains + 1))
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(8.0, max(2.5, 0.55 * max(len(selected), 1) + 1.0)))[1]
        ax.clear()
        for row, neuron_index in enumerate(selected, start=1):
            train = self.getNST(neuron_index)
            train.plot(dHeight=0.8, yOffset=float(row), currentHandle=ax)
        ax.set_ylim(0.25, len(selected) + 0.75)
        ax.set_yticks(range(1, len(selected) + 1), [str(item) for item in selected])
        ax.set_title("Spike Train Raster")
        return ax

    def psth(
        self,
        binwidth: float = 0.100,
        selectorArray: Sequence[int] | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> Covariate:
        if binwidth <= 0:
            raise ValueError("binwidth must be > 0")
        min_time = self.minTime if minTime is None else float(minTime)
        max_time = self.maxTime if maxTime is None else float(maxTime)
        if selectorArray is None:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(1, self.numSpikeTrains + 1))

        span = max_time - min_time
        n_full = int(np.floor((span / binwidth) + 1e-12))
        window_times = min_time + np.arange(n_full + 1, dtype=float) * float(binwidth)
        if window_times.size == 0:
            window_times = np.array([min_time, max_time], dtype=float)
        if window_times[-1] < max_time - 1e-12:
            window_times = np.append(window_times, max_time)
        elif window_times[-1] > max_time + 1e-12:
            window_times[-1] = max_time
        if window_times.size < 2:
            window_times = np.array([min_time, max_time], dtype=float)
            if window_times[1] <= window_times[0]:
                window_times[1] = window_times[0] + float(binwidth)

        psth_hist = np.zeros(window_times.size, dtype=float)
        for neuron in selectorArray:
            spikes = np.asarray(self.getNST(int(neuron)).getSpikeTimes(), dtype=float).reshape(-1)
            if spikes.size == 0:
                continue
            valid = np.isfinite(spikes) & (spikes >= window_times[0]) & (spikes <= window_times[-1])
            if not np.any(valid):
                continue
            idx = np.searchsorted(window_times, spikes[valid], side="right") - 1
            idx = np.clip(idx, 0, window_times.size - 1)
            psth_hist += np.bincount(idx, minlength=window_times.size).astype(float)

        psth_data = psth_hist[:-1] / binwidth / float(len(selectorArray))
        time = (window_times[1:] + window_times[:-1]) * 0.5
        return Covariate(time, psth_data, "PSTH", "time", "s", "Hz", ["psth"])

    def psthGLM(self, binwidth: float):
        psth_signal = self.psth(binwidth)
        return psth_signal, None, None


class TrialConfig:
    """MATLAB-style TrialConfig with configuration-application semantics."""

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
        self.covMask = [] if covMask is None else covMask
        self.sampleRate = [] if sampleRate is None else sampleRate
        self.history = [] if history is None else history
        self.ensCovHist = [] if ensCovHist is None else ensCovHist
        self.ensCovMask = [] if ensCovMask is None else ensCovMask
        self.covLag = [] if covLag is None else covLag
        self.name = str(name)

    @property
    def covariate_names(self) -> list[str]:
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
        return self.name

    def setName(self, name: str) -> None:
        self.name = str(name)

    def setConfig(self, trial: "Trial") -> None:
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
    """MATLAB-style ConfigColl implementation."""

    def __init__(self, configs: Sequence[TrialConfig] | TrialConfig | str | None = None) -> None:
        self.numConfigs = 0
        self.configNames: list[str] = []
        self.configArray: list[TrialConfig | str | list[str]] = []
        if configs is not None:
            self.addConfig(configs)

    @property
    def configs(self) -> list[TrialConfig]:
        return [cfg for cfg in self.configArray if isinstance(cfg, TrialConfig)]

    def add_config(self, cfg: TrialConfig) -> None:
        self.addConfig(cfg)

    def addConfig(self, cfg: Sequence[TrialConfig] | TrialConfig | str | None) -> None:
        if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes, TrialConfig, np.ndarray)):
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
            self.numConfigs += 1
            self.configArray.append(cfg)
            self.setConfigNames(cfg, [self.numConfigs])
            return
        raise TypeError("ConfigColl can only add TrialConfig objects, strings, or sequences of them.")

    def get_config(self, idx: int) -> TrialConfig | str | list[str]:
        if idx < 0 or idx >= self.numConfigs:
            raise IndexError("ConfigCollection index out of bounds (0-based indexing).")
        return self.configArray[idx]

    def getConfig(self, idx: int):
        if idx < 1 or idx > self.numConfigs:
            raise IndexError("Index Out of Bounds")
        return self.configArray[idx - 1]

    def setConfig(self, trial: "Trial", index: int) -> None:
        config = self.getConfig(index)
        if isinstance(config, TrialConfig):
            config.setConfig(trial)
            return
        raise ValueError("Cannot Set Empty Configs")

    def getConfigNames(self, index: Sequence[int] | None = None) -> list[str]:
        if index is None:
            index = list(range(1, self.numConfigs + 1))
        out: list[str] = []
        for i in index:
            if i < 1 or i > self.numConfigs:
                raise IndexError("Index Out of Bounds")
            tempName = self.configNames[i - 1]
            out.append(tempName if tempName else f"Fit {i}")
        return out

    def setConfigNames(self, names, index: Sequence[int] | None = None) -> None:
        if index is None:
            index = list(range(1, self.numConfigs + 1))
        if isinstance(names, str):
            if len(index) != 1:
                raise ValueError("If specifying a single name, index must be length 1.")
            target = int(index[0]) - 1
            while len(self.configNames) < self.numConfigs:
                self.configNames.append("")
            self.configNames[target] = names if names else f"Fit {target + 1}"
            return
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            if len(index) != len(names):
                raise ValueError("If specifying multiple names, names and index must match in length.")
            for idx, name in zip(index, names):
                self.setConfigNames(str(name), [int(idx)])
            return
        raise TypeError("names must be a string or sequence of strings.")

    def getSubsetConfigs(self, subset: Sequence[int]) -> "ConfigCollection":
        tempconfigs = [self.getConfig(int(i)) for i in subset]
        return ConfigCollection(tempconfigs)

    def toStructure(self) -> dict[str, Any]:
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
        configs = []
        for row in structure.get("configArray", []):
            if isinstance(row, dict):
                configs.append(TrialConfig.fromStructure(row))
            else:
                configs.append(row)
        return ConfigCollection(configs)


class Trial:
    """MATLAB-style Trial object preserving collection-level workflow semantics."""

    def __init__(
        self,
        spike_collection: SpikeTrainCollection | None = None,
        covariate_collection: CovariateCollection | None = None,
        events: Events | None = None,
        hist: object | None = None,
        ensCovHist: object | None = None,
        ensCovMask: object | None = None,
        *,
        spikeColl: SpikeTrainCollection | None = None,
        covarColl: CovariateCollection | None = None,
        event: Events | None = None,
    ) -> None:
        self.nspikeColl = spike_collection if spike_collection is not None else spikeColl
        self.covarColl = covariate_collection if covariate_collection is not None else covarColl
        if not isinstance(self.nspikeColl, SpikeTrainCollection):
            raise ValueError("nstColl is a required argument")
        if not isinstance(self.covarColl, CovariateCollection):
            raise ValueError("CovColl is a required argument")

        self.ev: Events | None = None
        self.history: object | None = []
        self.ensCovHist: object | None = []
        self.ensCovColl: CovariateCollection | None = None
        self.sampleRate = float("nan")
        self.minTime = float("nan")
        self.maxTime = float("nan")
        self.covMask = self.covarColl.covMask
        self.ensCovMask = ensCovMask
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()
        self.trainingWindow: list[float] | np.ndarray | None = None
        self.validationWindow: list[float] | np.ndarray | None = None

        event_obj = events if events is not None else event
        self.setTrialEvents(event_obj)
        self.setHistory(hist)
        self.setEnsCovHist(ensCovHist)
        self.setEnsCovMask(ensCovMask)

        self.covMask = self.covarColl.covMask
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        else:
            self.sampleRate = float(self.covarColl.sampleRate)
        self.makeConsistentTime()
        self.setTrialPartition([])
        self.setTrialTimesFor("training")

    @property
    def spike_collection(self) -> SpikeTrainCollection:
        return self.nspikeColl

    @property
    def covariate_collection(self) -> CovariateCollection:
        return self.covarColl

    @property
    def spikeColl(self) -> SpikeTrainCollection:
        return self.nspikeColl

    def setTrialEvents(self, event: Events | None) -> None:
        self.ev = event if isinstance(event, Events) else None

    def getEvents(self) -> Events | None:
        return self.ev

    @property
    def covarColl(self) -> CovariateCollection:
        return self._covarColl

    @covarColl.setter
    def covarColl(self, value: CovariateCollection) -> None:
        self._covarColl = value

    def getTrialPartition(self) -> np.ndarray:
        training = [] if self.trainingWindow is None else list(self.trainingWindow)
        validation = [] if self.validationWindow is None else list(self.validationWindow)
        p = training + validation
        if not p:
            return np.asarray([self.minTime, self.maxTime, self.maxTime, self.maxTime], dtype=float)
        return np.asarray(p, dtype=float)

    def setTrialPartition(self, partitionTimes) -> None:
        if partitionTimes is None or len(partitionTimes) == 0:
            partitionTimes = self.getTrialPartition()
        values = np.asarray(partitionTimes, dtype=float).reshape(-1)
        if values.size == 4:
            trainingWindow = values[:2]
            validationWindow = values[2:]
        elif values.size == 3:
            trainingWindow = values[:2]
            validationWindow = values[1:]
        else:
            raise ValueError("partitionTimes must be length 3 or 4")
        self.trainingWindow = trainingWindow
        self.validationWindow = validationWindow
        self.setMinTime(trainingWindow[0])
        self.setMaxTime(trainingWindow[1])

    def setTrialTimesFor(self, partitionName: str = "training") -> None:
        p = self.getTrialPartition()
        if partitionName == "training":
            timeWindow = p[:2]
        elif partitionName == "validation":
            timeWindow = p[2:4]
        else:
            raise ValueError("partitionName must be either training or validation")
        self.setMinTime(float(timeWindow[0]))
        self.setMaxTime(float(timeWindow[1]))

    def setMinTime(self, minTime: float | None = None) -> None:
        if minTime is None:
            minTime = self.findMinTime()
        self.nspikeColl.setMinTime(float(minTime))
        self.covarColl.setMinTime(float(minTime))
        if self.ensCovColl is not None:
            self.ensCovColl.setMinTime(float(minTime))
        self.minTime = float(minTime)

    def setMaxTime(self, maxTime: float | None = None) -> None:
        if maxTime is None:
            maxTime = self.findMaxTime()
        self.nspikeColl.setMaxTime(float(maxTime))
        self.covarColl.setMaxTime(float(maxTime))
        if self.ensCovColl is not None:
            self.ensCovColl.setMaxTime(float(maxTime))
        self.maxTime = float(maxTime)

    def updateTimePartitions(self) -> None:
        if not (np.isfinite(self.minTime) and np.isfinite(self.maxTime)):
            return
        p = self.getTrialPartition()
        training = p[:2]
        validation = p[2:4]
        newTrainMin = max(self.minTime, training[0])
        newTrainMax = min(self.maxTime, training[1])
        newValMin = max(self.minTime, validation[0])
        newValMax = min(self.maxTime, validation[1])
        self.setTrialPartition([newTrainMin, newTrainMax, newValMin, newValMax])

    def plot(self, *_, handle=None, **__):
        cov_count = max(self.covarColl.numCov, 1)
        event_count = 1 if self.ev is not None and self.ev.eventTimes.size else 0
        panel_count = 1 + cov_count + event_count
        fig = handle if handle is not None else plt.figure(figsize=(9.0, max(4.0, 2.2 * panel_count)))
        fig.clear()
        axes = fig.subplots(panel_count, 1, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)

        cursor = 0
        self.nspikeColl.plot(handle=axes[cursor])
        axes[cursor].set_title("Trial Spike Raster")
        cursor += 1

        for cov_index in range(1, self.covarColl.numCov + 1):
            cov = self.covarColl.getCov(cov_index)
            cov.plot(handle=axes[cursor])
            axes[cursor].set_title(cov.name)
            cursor += 1

        if event_count:
            self.ev.plot(handle=axes[cursor])
            cursor += 1

        fig.tight_layout()
        return fig

    def setSampleRate(self, sampleRate: float) -> None:
        self.sampleRate = float(sampleRate)
        self.nspikeColl.resample(sampleRate)
        self.covarColl.resample(sampleRate)
        self.resampleEnsColl()

    def resample(self, sampleRate: float) -> None:
        self.setSampleRate(sampleRate)

    def setEnsCovMask(self, mask=None) -> None:
        if _is_empty_config_value(mask):
            nSpikes = self.nspikeColl.numSpikeTrains
            mask = np.ones((nSpikes, nSpikes), dtype=int) - np.eye(nSpikes, dtype=int)
        self.ensCovMask = np.asarray(mask, dtype=int)

    def setCovMask(self, mask) -> None:
        if isinstance(mask, str) and mask == "all":
            self.covarColl.resetMask()
        else:
            self.covarColl.setMask(mask)
        self.covMask = self.covarColl.covMask

    def resetCovMask(self) -> None:
        self.covarColl.resetMask()
        self.covMask = self.covarColl.covMask

    def setNeuronMask(self, mask) -> None:
        self.nspikeColl.setMask(mask)
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()

    def resetNeuronMask(self) -> None:
        self.nspikeColl.resetMask()
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()

    def setNeighbors(self, *args) -> None:
        self.nspikeColl.setNeighbors(*args)

    def setHistory(self, hist) -> None:
        if _is_empty_config_value(hist):
            self.history = []
            return
        from .history import History

        if isinstance(hist, History):
            self.history = hist
            return
        if isinstance(hist, np.ndarray):
            if hist.ndim > 2 or (hist.ndim == 2 and min(hist.shape) > 1):
                raise ValueError("Only one of the dimension of the windowTimes can be greater than 1.")
            arr = np.asarray(hist, dtype=float).reshape(-1)
            if arr.size <= 1:
                raise ValueError("At least two times points must be specified to determine a window")
            self.history = History(arr)
            return
        if isinstance(hist, Sequence) and not isinstance(hist, (str, bytes)):
            if hist and all(isinstance(item, History) for item in hist):
                self.history = list(hist)
                return
            arr = np.asarray(hist, dtype=float).reshape(-1)
            if arr.size <= 1:
                raise ValueError("At least two times points must be specified to determine a window")
            self.history = History(arr)
            return
        raise TypeError("Can only set trial history by using History objects or windowTimes")

    def resetHistory(self) -> None:
        self.history = []

    def setEnsCovHist(self, hist=None) -> None:
        if _is_empty_config_value(hist):
            self.ensCovHist = []
            self.ensCovColl = None
            return
        from .history import History

        if isinstance(hist, History):
            self.ensCovHist = hist
        elif isinstance(hist, np.ndarray):
            if hist.ndim > 2 or (hist.ndim == 2 and min(hist.shape) > 1):
                raise ValueError("Only one of the dimension of the windowTimes can be greater than 1.")
            arr = np.asarray(hist, dtype=float).reshape(-1)
            if arr.size <= 1:
                raise ValueError("At least two times points must be specified to determine a window")
            self.ensCovHist = History(arr)
        elif isinstance(hist, Sequence) and not isinstance(hist, (str, bytes)):
            arr = np.asarray(hist, dtype=float).reshape(-1)
            if arr.size <= 1:
                raise ValueError("At least two times points must be specified to determine a window")
            self.ensCovHist = History(arr)
        else:
            raise TypeError("Can only set trial ensCovHist by using History objects or windowTimes")
        self.ensCovColl = self.getEnsembleNeuronCovariates(1, [], self.ensCovHist)

    def isNeuronMaskSet(self) -> bool:
        return self.nspikeColl.isNeuronMaskSet()

    def isCovMaskSet(self) -> bool:
        return self.covarColl.isCovMaskSet()

    def isMaskSet(self) -> bool:
        return self.isNeuronMaskSet() or self.isCovMaskSet()

    def isHistSet(self) -> bool:
        if self.history in (None, []):
            return False
        from .history import History

        if isinstance(self.history, History):
            return True
        return isinstance(self.history, list) and bool(self.history) and all(isinstance(item, History) for item in self.history)

    def isEnsCovHistSet(self) -> bool:
        from .history import History

        return isinstance(self.ensCovHist, History)

    def addCov(self, cov: Covariate) -> None:
        self.covarColl.addToColl(cov)
        self.covMask = self.covarColl.covMask
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.makeConsistentTime()

    def removeCov(self, identifier: int | str) -> None:
        self.covarColl.removeCovariate(identifier)
        self.covMask = self.covarColl.covMask
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.makeConsistentTime()

    def getSpikeVector(self, *args, neuron_index: int = 1) -> np.ndarray:
        if not args:
            return self.nspikeColl.dataToMatrix()
        first = args[0]
        if isinstance(first, (int, np.integer)):
            selector = [int(first)]
            if len(args) == 1:
                return self.nspikeColl.dataToMatrix(selector)
            return self.nspikeColl.dataToMatrix(selector, *args[1:])
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, np.ndarray)):
            bin_edges = np.asarray(first, dtype=float).reshape(-1)
            return self.nspikeColl.getNST(neuron_index).to_binned_counts(bin_edges)
        return self.nspikeColl.dataToMatrix(*args)

    def get_covariate_matrix(self, selected_covariates: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return self.covarColl.matrixWithTime("standard", selected_covariates)

    def getDesignMatrix(self, neuronNum: int, dataSelector=None) -> np.ndarray:
        X = self.covarColl.dataToMatrix("standard", dataSelector)
        if self.isHistSet():
            H = self.getHistMatrices(neuronNum)
            X = H if X.size == 0 else np.column_stack([X, H])
        if self.isEnsCovHistSet():
            E = self.getEnsCovMatrix(neuronNum)
            X = E if X.size == 0 else np.column_stack([X, E])
        return X

    def getHistForNeurons(self, neuronIndex) -> CovariateCollection:
        if not self.isHistSet():
            raise ValueError("Set Trial history and retry")
        nst = self.nspikeColl.getNST(neuronIndex)
        target_time = np.asarray(self.covarColl.getCov(1).time, dtype=float).reshape(-1) if self.covarColl.numCov else None
        if isinstance(self.history, list):
            histCovColl: CovariateCollection | None = None
            for i, hist in enumerate(self.history, start=1):
                temp = hist.computeHistory(nst, i, time_grid=target_time)
                histCovColl = temp if histCovColl is None else CovariateCollection([*histCovColl.covArray, *temp.covArray])
            assert histCovColl is not None
            return histCovColl
        return self.history.computeHistory(nst, time_grid=target_time)

    def getHistMatrices(self, neuronIndex: int) -> np.ndarray:
        if not self.isHistSet():
            time = self.nspikeColl.getNST(neuronIndex).getSigRep().time
            return np.zeros((time.size, 0), dtype=float)
        histCovColl = self.getHistForNeurons(neuronIndex)
        return histCovColl.dataToMatrix("standard")

    def getEnsembleNeuronCovariates(self, *args):
        return self.nspikeColl.getEnsembleNeuronCovariates(*args)

    def getEnsCovMatrix(self, neuronNum: int, includedNeurons=None) -> np.ndarray:
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return np.zeros((self.nspikeColl.getNST(neuronNum).getSigRep().time.size, 0), dtype=float)
        if includedNeurons is None:
            includedNeurons = np.flatnonzero(self.ensCovMask[:, neuronNum - 1] == 1) + 1
        ensCovCollTemp = CovariateCollection(self.ensCovColl.covArray)
        ensCovCollTemp.covMask = [mask.copy() for mask in self.ensCovColl.covMask]
        ensCovCollTemp.maskAwayAllExcept(includedNeurons)
        return ensCovCollTemp.dataToMatrix("standard")

    def getNeuronIndFromMask(self) -> list[int]:
        return self.nspikeColl.getIndFromMask()

    def getNumUniqueNeurons(self) -> int:
        return len(self.nspikeColl.uniqueNeuronNames)

    def getNeuronNames(self) -> list[str]:
        return self.nspikeColl.getNSTnames()

    def getUniqueNeuronNames(self) -> list[str]:
        return self.nspikeColl.getUniqueNSTnames()

    def getNeuronIndFromName(self, neuronName: str):
        tempInd = self.nspikeColl.getNSTIndicesFromName(neuronName)
        currMask = set(self.neuronMask_indices())
        if isinstance(tempInd, list):
            return [idx for idx in tempInd if idx in currMask]
        return [tempInd] if tempInd in currMask else []

    def neuronMask_indices(self) -> list[int]:
        return self.nspikeColl.getIndFromMask()

    def getNeuronNeighbors(self, neuronNum=None):
        if neuronNum is None:
            neuronNum = self.getNeuronIndFromMask()
        return self.nspikeColl.getNeighbors(neuronNum)

    def getCovSelectorFromMask(self):
        return self.covarColl.getSelectorFromMasks()

    def getCov(self, identifier):
        return self.covarColl.getCov(identifier)

    def getNeuron(self, identifier):
        return self.nspikeColl.getNST(identifier)

    def getAllCovLabels(self) -> list[str]:
        return self.covarColl.getAllCovLabels()

    def getCovLabelsFromMask(self) -> list[str]:
        return self.covarColl.getCovLabelsFromMask()

    def getHistLabels(self) -> list[str]:
        if not self.isHistSet():
            return []
        return self.getHistForNeurons(1).getAllCovLabels()

    def getEnsCovLabels(self) -> list[str]:
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return []
        return self.ensCovColl.getAllCovLabels()

    def getEnsCovLabelsFromMask(self, neuronNum: int) -> list[str]:
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return []
        included = np.flatnonzero(self.ensCovMask[:, neuronNum - 1] == 1) + 1
        ensCovCollTemp = CovariateCollection(self.ensCovColl.covArray)
        ensCovCollTemp.covMask = [mask.copy() for mask in self.ensCovColl.covMask]
        ensCovCollTemp.maskAwayAllExcept(included)
        return ensCovCollTemp.getCovLabelsFromMask()

    def getLabelsFromMask(self, neuronNum: int) -> list[str]:
        labels = list(self.getCovLabelsFromMask())
        labels.extend(self.getHistLabels())
        labels.extend(self.getEnsCovLabelsFromMask(neuronNum))
        return labels

    def flattenCovMask(self) -> np.ndarray:
        return self.covarColl.flattenCovMask()

    def flattenMask(self) -> np.ndarray:
        flat = self.flattenCovMask()
        if self.isHistSet():
            flat = np.concatenate([flat, np.ones(len(self.getHistLabels()), dtype=int)])
        if self.isEnsCovHistSet():
            flat = np.concatenate([flat, np.ones(len(self.getEnsCovLabels()), dtype=int)])
        return flat

    def shiftCovariates(self, *args) -> None:
        self.covarColl.setCovShift(*args)
        self.makeConsistentTime()

    def resetEnsCovMask(self) -> None:
        self.setEnsCovMask()

    def resampleEnsColl(self) -> None:
        if self.ensCovColl is not None and self.ensCovHist not in (None, []):
            self.ensCovColl = self.getEnsembleNeuronCovariates(1, [], self.ensCovHist)
        else:
            self.setEnsCovHist()

    def restoreToOriginal(self) -> None:
        self.nspikeColl.restoreToOriginal()
        self.covarColl.restoreToOriginal()
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.resampleEnsColl()
        self.makeConsistentTime()

    def makeConsistentSampleRate(self) -> None:
        self.resample(self.findMaxSampleRate())

    def makeConsistentTime(self) -> None:
        self.setMinTime(self.findMinTime())
        self.setMaxTime(self.findMaxTime())

    def isSampleRateConsistent(self) -> bool:
        if self.nspikeColl.numSpikeTrains == 0 or self.covarColl.numCov == 0:
            return True
        target = round(float(self.findMaxSampleRate()), 3)
        values = [round(float(self.nspikeColl.sampleRate), 3), round(float(self.covarColl.sampleRate), 3)]
        return all(value == target for value in values)

    def findMaxSampleRate(self) -> float:
        values = [value for value in [self.nspikeColl.findMaxSampleRate(), self.covarColl.findMaxSampleRate()] if np.isfinite(value)]
        return float(max(values)) if values else float("nan")

    def findMinTime(self) -> float:
        return float(min(self.nspikeColl.minTime, self.covarColl.minTime))

    def findMaxTime(self) -> float:
        return float(max(self.nspikeColl.maxTime, self.covarColl.maxTime))


# Backward-compatible MATLAB-style aliases.
CovColl = CovariateCollection
nstColl = SpikeTrainCollection
ConfigColl = ConfigCollection


__all__ = [
    "CovariateCollection",
    "SpikeTrainCollection",
    "TrialConfig",
    "ConfigCollection",
    "Trial",
    "CovColl",
    "nstColl",
    "ConfigColl",
]
