"""Trial-level containers binding spikes, covariates, and events.

This module hosts the per-trial data structures used by every GLM and
decoding workflow in nSTAT:

- :class:`CovariateCollection` — ordered collection of
  :class:`~nstat.core.Covariate` objects (mirrors MATLAB ``CovColl.m``).
- :class:`SpikeTrainCollection` — ordered collection of
  :class:`~nstat.core.nspikeTrain` objects (mirrors MATLAB ``nstColl.m``).
- :class:`Trial` — joins a spike collection, covariate collection, and
  optional :class:`~nstat.events.Events` stream into a single, time- and
  rate-aligned analysis unit (mirrors MATLAB ``Trial.m``).

The model-specification helpers :class:`TrialConfig` (Matlab
``TrialConfig.m``) and :class:`ConfigCollection` (Matlab ``ConfigColl.m``)
were extracted to :mod:`nstat._trial_config_impl` for readability, but are
re-exported here so that ``from nstat.trial import TrialConfig`` continues
to work.  All time vectors are in **seconds** and sample rates in **Hz**.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .core import Covariate, SignalObj, nspikeTrain
from .events import Events


def _is_string_sequence(values: object) -> bool:
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Sequence):
        return False
    return all(isinstance(item, str) for item in values)


from ._trial_config_impl import _is_empty_config_value, TrialConfig, ConfigCollection  # noqa: E402, F401



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


def _copy_covariate_for_collection_view(cov: Covariate) -> Covariate:
    """Mirror MATLAB CovColl.getCov copy semantics.

    MATLAB reconstructs a fresh Covariate from the visible time/data payload
    rather than preserving internal sample-rate/original-data bookkeeping.
    That matters for degenerate one-sample covariates, where the constructor
    falls back to the 1 kHz default used throughout the toolbox.
    """

    copied = Covariate(
        np.asarray(cov.time, dtype=float).copy(),
        np.asarray(cov.data, dtype=float).copy(),
        cov.name,
        cov.xlabelval,
        cov.xunits,
        cov.yunits,
        list(cov.dataLabels),
        list(cov.plotProps),
    )
    copied.dataMask = np.asarray(cov.dataMask, dtype=int).copy()
    if cov.ci:
        copied.ci = list(cov.ci)
    if cov.conf_interval is not None:
        copied.conf_interval = (
            np.asarray(cov.conf_interval[0], dtype=float).copy(),
            np.asarray(cov.conf_interval[1], dtype=float).copy(),
        )
    return copied


class CovariateCollection:
    """Ordered collection of :class:`~nstat.core.Covariate` objects (Matlab ``CovColl``).

    Provides collection-level masking, time alignment, sample-rate
    enforcement, and covariate shifting.  Individual covariates are
    accessed via 0-based indexing (``getCov(1)``) to match Matlab.

    Parameters
    ----------
    covariates : Covariate, sequence of Covariate, or None
        Initial covariate(s) to add.

    See Also
    --------
    Covariate : Scalar or multi-dimensional signal with CIs.
    Trial : Combines a ``CovariateCollection`` with spike data.
    """

    def __init__(self, covariates: Sequence[Covariate] | Covariate | None = None, *more_covariates: Covariate) -> None:
        """Construct an ordered covariate collection (Matlab ``CovColl``).

        Parameters
        ----------
        covariates : Covariate, sequence of Covariate, CovariateCollection, or None
            Initial covariate(s) to add.  May be a single
            :class:`~nstat.core.Covariate`, an iterable of them, another
            :class:`CovariateCollection` (whose contents are merged in),
            or ``None`` (empty collection).
        *more_covariates : Covariate
            Additional covariates appended after *covariates* — supports
            the MATLAB-style ``CovColl(c1, c2, c3, ...)`` call signature.

        Notes
        -----
        On construction the collection captures the maximal sample rate
        across all input covariates (in **Hz**), the union of their time
        windows (in **seconds**), and a per-dimension mask of ones for
        each covariate (no dimensions masked out).  Time bounds and
        sample rate are recomputed lazily via ``_refresh_summary``.

        Indexing is **0-based** to match MATLAB (``coll.getCov(0)`` is
        the first covariate).

        See Also
        --------
        Covariate : Scalar or multi-dimensional signal with CIs.
        Trial : Combines a ``CovariateCollection`` with spike data.
        """
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
        """List of all covariates (copies with collection state applied)."""
        return [self.getCov(i) for i in range(self.numCov)]

    @property
    def names(self) -> list[str]:
        """List of covariate names in insertion order."""
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
        if index < 0 or index >= self.numCov:
            raise IndexError("Covariate index out of bounds.")
        return index

    def _apply_collection_state(self, cov: Covariate, index: int) -> Covariate:
        out = _copy_covariate_for_collection_view(cov)
        if self.covShift != 0:
            out.time = out.time + float(self.covShift)
            out.minTime = float(np.min(out.time))
            out.maxTime = float(np.max(out.time))
        if out.time.size > 1 and np.isfinite(self.sampleRate) and self.sampleRate > 0 and round(out.sampleRate, 3) != round(self.sampleRate, 3):
            out = out.resample(self.sampleRate)
        if np.isfinite(self.minTime) and np.isfinite(self.maxTime) and out.time.size > 0:
            out = out.getSigInTimeWindow(self.minTime, self.maxTime, holdVals=1)
        out.setMask(self.covMask[index])
        return out

    def add(self, covariate: Covariate) -> None:
        """Alias for :meth:`addToColl`."""
        self.addToColl(covariate)

    def addCovariate(self, covariate: Covariate) -> None:
        """Alias for :meth:`addToColl`."""
        self.addToColl(covariate)

    def addCovCollection(self, covariates: "CovariateCollection") -> None:
        """Merge all covariates from another collection into this one."""
        self.addToColl(covariates)

    def addToColl(self, covariates: Sequence[Covariate] | Covariate | "CovariateCollection" | None) -> None:
        if covariates is None:
            return
        if isinstance(covariates, CovariateCollection):
            for cov in covariates.covArray:
                self.addToColl(cov)
            return
        if isinstance(covariates, Covariate):
            new_cov = _copy_covariate(covariates)
            self.covArray.append(new_cov)
            self.covMask.append(np.ones(covariates.dimension, dtype=int))
            # Reconcile sample rates per MATLAB ``CovColl.m:799-810``
            # (audit finding H3): if the new covariate's rate is HIGHER
            # than the collection's, the collection rate is bumped up
            # and ``enforceSampleRate`` upsamples every previously-
            # stored covariate.  If the collection rate is higher than
            # the new covariate's, the new covariate is upsampled
            # in-place to match.
            cov_rate = float(new_cov.sampleRate) if np.isfinite(new_cov.sampleRate) else None
            coll_rate = float(self.sampleRate) if np.isfinite(self.sampleRate) else None
            # Use ``len(self.covArray)`` not ``self.numCov`` because
            # ``_refresh_summary`` (which updates ``numCov``) runs at the
            # very end of ``addToColl``; ``self.numCov`` is stale here.
            n_after_append = len(self.covArray)
            if cov_rate is not None:
                if coll_rate is None or n_after_append == 1:
                    # First element — adopt the new cov's rate.
                    self.sampleRate = cov_rate
                elif coll_rate < cov_rate:
                    # New cov has higher rate — upsample the collection.
                    self.sampleRate = cov_rate
                    self.enforceSampleRate()
                elif coll_rate > cov_rate:
                    # New cov has lower rate — upsample only the new cov.
                    new_cov.resampleMe(coll_rate)
            self._refresh_summary()
            return
        if isinstance(covariates, Sequence) and not isinstance(covariates, (str, bytes, np.ndarray)):
            for cov in covariates:
                self.addToColl(cov)
            return
        raise TypeError("CovColl can only add Covariate instances or sequences of Covariates.")

    def removeCovariate(self, identifier: int | str) -> None:
        """Remove a covariate by 0-based index or name."""
        index = self._covariate_from_identifier(identifier)
        del self.covArray[index]
        del self.covMask[index]
        self._refresh_summary()

    def copy(self) -> "CovariateCollection":
        """Return a deep copy of this collection."""
        cov = [self.getCov(i).copySignal() for i in range(self.numCov)]
        return CovariateCollection(cov)

    def get(self, name: str) -> Covariate:
        """Retrieve a covariate by name (convenience alias for :meth:`getCov`)."""
        return self.getCov(name)

    def getCov(self, identifier: int | str | Sequence[int] | Sequence[str]):
        """Return a covariate copy with collection state (shift, mask, rate) applied.

        Parameters
        ----------
        identifier : int, str, or sequence
            0-based index, covariate name, or sequence of either.
        """
        if isinstance(identifier, str):
            return self._apply_collection_state(self.covArray[self.getCovIndFromName(identifier)], self.getCovIndFromName(identifier))
        if isinstance(identifier, Sequence) and not isinstance(identifier, (str, bytes, np.ndarray)):
            if _is_string_sequence(identifier):
                return [self.getCov(item) for item in identifier]
            return [self.getCov(int(item)) for item in identifier]
        if isinstance(identifier, np.ndarray) and identifier.ndim > 0:
            return [self.getCov(int(item)) for item in identifier.reshape(-1)]
        index = self._covariate_from_identifier(identifier)
        return self._apply_collection_state(self.covArray[index], index)

    def getCovIndFromName(self, name: str) -> int:
        """Return the 0-based index of a covariate by *name*."""
        for idx, cov in enumerate(self.covArray):
            if cov.name == name:
                return idx
        raise KeyError(f"Covariate '{name}' not found")

    def getCovIndicesFromNames(self, name: Sequence[str] | str):
        """Return 0-based index(es) for one or more covariate names."""
        if isinstance(name, str):
            return self.getCovIndFromName(name)
        return [self.getCovIndFromName(item) for item in name]

    def isCovPresent(self, cov) -> int:
        """Return ``1`` if a covariate is in this collection, ``0`` otherwise."""
        if isinstance(cov, Covariate):
            if not cov.name:
                return 0
            try:
                self.getCovIndFromName(cov.name)
            except KeyError:
                return 0
            return 1
        if isinstance(cov, str):
            try:
                self.getCovIndFromName(cov)
            except KeyError:
                return 0
            return 1
        if isinstance(cov, (int, np.integer, float, np.floating)):
            index = int(cov)
            return int(index >= 0 and index < self.numCov)
        raise TypeError("Need either covariate class or name of covariate or index of covariate")

    def findMinTime(self) -> float:
        """Return the earliest ``minTime`` across all stored covariates."""
        if self.numCov == 0:
            return float("inf")
        return float(min(cov.minTime for cov in self.covArray))

    def findMaxTime(self) -> float:
        """Return the latest ``maxTime`` across all stored covariates.

        Note: MATLAB ``CovColl.m:371-380`` applies ``covShift`` twice
        here (once inside the per-covariate ``max(...)`` and once after
        the loop) — see AUDIT_REPORT M7 / nSTAT issue #18.  Python adds
        the shift exactly once, in ``_refresh_summary`` and
        ``setMaxTime``.  Do not "fix" this to match MATLAB's behavior.
        """
        if self.numCov == 0:
            return float("-inf")
        return float(max(cov.maxTime for cov in self.covArray))

    def findMaxSampleRate(self) -> float:
        """Return the highest sample rate across all stored covariates."""
        if self.numCov == 0:
            return float("nan")
        return float(max(cov.sampleRate for cov in self.covArray if np.isfinite(cov.sampleRate)))

    def setMinTime(self, minTime: float | None = None) -> None:
        """Set the collection-level minimum time (applies shift if set)."""
        if minTime is None:
            minTime = self.findMinTime() + float(self.covShift)
        self.minTime = float(minTime)

    def setMaxTime(self, maxTime: float | None = None) -> None:
        """Set the collection-level maximum time (applies shift if set)."""
        if maxTime is None:
            maxTime = self.findMaxTime() + float(self.covShift)
        self.maxTime = float(maxTime)

    def restrictToTimeWindow(self, wMin: float, wMax: float) -> None:
        """Set both min and max time to restrict the visible window."""
        self.setMinTime(wMin)
        self.setMaxTime(wMax)

    def setSampleRate(self, sampleRate: float) -> None:
        """Set the collection sample rate and enforce it on all covariates."""
        if self.originalSampleRate is None and np.isfinite(self.sampleRate):
            self.originalSampleRate = float(self.sampleRate)
        self.sampleRate = float(sampleRate)
        self.enforceSampleRate()

    def resample(self, sampleRate: float) -> None:
        """Alias for :meth:`setSampleRate`."""
        self.setSampleRate(sampleRate)

    def enforceSampleRate(self) -> None:
        """Resample every stored covariate to match ``self.sampleRate``.

        Matches MATLAB ``CovColl.m:491-502`` which iterates each
        covariate and calls ``resampleMe`` whenever its rate differs
        from the collection's rate.  The previous Python implementation
        only validated that ``self.sampleRate`` was finite; it did NOT
        actually resample stored covariates.  Net effect: a collection
        built by sequentially adding covariates at increasing rates
        would report ``self.sampleRate`` at the highest rate but leave
        earlier covariates at their original (lower) rates.  Closes
        audit finding H3.
        """
        if not np.isfinite(self.sampleRate) or self.sampleRate <= 0:
            self.sampleRate = self.findMaxSampleRate()
        if not np.isfinite(self.sampleRate) or self.sampleRate <= 0:
            return
        target_rate = float(self.sampleRate)
        for cov in self.covArray:
            if not np.isfinite(cov.sampleRate):
                continue
            if round(cov.sampleRate, 6) != round(target_rate, 6):
                cov.resampleMe(target_rate)

    def resetMask(self) -> None:
        """Enable all covariate dimensions (clear any masking)."""
        self.covMask = [np.ones(cov.dimension, dtype=int) for cov in self.covArray]

    def getCovDataMask(self, identifier: int | str) -> np.ndarray:
        """Return the binary dimension mask for a single covariate."""
        index = self._covariate_from_identifier(identifier)
        return np.asarray(self.covMask[index], dtype=int).copy()

    def isCovMaskSet(self) -> bool:
        """Return ``True`` if any covariate dimension is currently masked out."""
        return any(np.any(mask == 0) for mask in self.covMask)

    def flattenCovMask(self) -> np.ndarray:
        """Concatenate all per-covariate masks into a single 1-D binary array."""
        if not self.covMask:
            return np.array([], dtype=int)
        return np.concatenate([np.asarray(mask, dtype=int).reshape(-1) for mask in self.covMask])

    def getSelectorFromMasks(self, covMask: list[np.ndarray] | None = None) -> list[list[int]]:
        """Convert per-covariate binary masks to lists of active 0-based indices."""
        current = self.covMask if covMask is None else covMask
        selector: list[list[int]] = []
        for mask in current:
            active = np.flatnonzero(np.asarray(mask, dtype=int) == 1)
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
                selectorCell[covIndex] = currCov.getIndicesFromLabels([])
            else:
                selectorCell[covIndex] = currCov.getIndicesFromLabels([str(v) for v in dataSelector[1:]])
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
                selectorCell[covIndex] = currCov.getIndicesFromLabels([])
            else:
                selectorCell[covIndex] = currCov.getIndicesFromLabels([str(v) for v in parsed[1:]])
        return selectorCell

    def generateSelectorCell(self, dataSelector) -> list[list[int]]:
        """Parse a heterogeneous *dataSelector* into per-covariate index lists.

        Accepts name-based (``[['covName', 'label1', ...], ...]``) or
        numeric (``[[1,2], [3], ...]``) selectors.
        """
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
                if np.any(arr < 0) or np.any(arr >= cov.dimension):
                    raise IndexError("Covariate selector index out of bounds.")
                mask[arr] = 1
            masks.append(mask)
        return masks

    def setMasksFromSelector(self, selectorCell: list[list[int]]) -> None:
        """Set covariate masks from a list of 0-based index lists."""
        self.covMask = self._selector_to_cov_mask(selectorCell)

    def setMask(self, cellInput) -> None:
        """Set the covariate mask from a selector or ``'all'`` to reset.

        Accepts the same formats as :meth:`generateSelectorCell`.
        """
        if isinstance(cellInput, str) and cellInput == "all":
            self.resetMask()
            return
        selectorCell = self.generateSelectorCell(cellInput)
        self.setMasksFromSelector(selectorCell)

    def nActCovar(self) -> int:
        """Return the number of covariates with at least one active dimension."""
        return int(sum(1 for selector in self.getSelectorFromMasks() if selector))

    def maskAwayCov(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        """Zero-out the mask for the specified covariate(s)."""
        identifiers = identifier
        if isinstance(identifier, (int, str)):
            identifiers = [identifier]
        for item in identifiers:
            index = self._covariate_from_identifier(item)
            self.covMask[index] = np.zeros(self.covArray[index].dimension, dtype=int)

    def maskAwayOnlyCov(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        """Reset all masks then mask away only the specified covariate(s)."""
        self.resetMask()
        self.maskAwayCov(identifier)

    def maskAwayAllExcept(self, identifier: int | str | Sequence[int] | Sequence[str]) -> None:
        """Mask away every covariate *except* the ones specified.

        Matches MATLAB ``CovColl.m:202-208`` (``maskAwayOnlyCov`` →
        ``resetMask`` first, then ``maskAwayCov``).  Before the reset
        the kept covariates' masks were preserved as-is, which silently
        dropped any covariate whose stored mask was already all-zero —
        e.g. the per-neighbor masks inside ``Trial.ensCovColl``, which
        are initialised for neuron 1's view and contain zeros for the
        other neuron's row.  Without resetting, ``getEnsCovMatrix(n=2)``
        returned an empty matrix and the NetworkTutorial fit silently
        dropped the inter-neuron coupling term.
        """
        if isinstance(identifier, (int, str)):
            keep = {self._covariate_from_identifier(identifier)}
        else:
            keep = {self._covariate_from_identifier(item) for item in identifier}
        for idx, cov in enumerate(self.covArray):
            if idx in keep:
                self.covMask[idx] = np.ones(cov.dimension, dtype=int)
            else:
                self.covMask[idx] = np.zeros(cov.dimension, dtype=int)

    def setCovShift(self, deltaT: float, identifier=None) -> "CovariateCollection":
        """Apply a temporal shift *deltaT* to the collection's time axis.

        Matches MATLAB ``CovColl.m:504-520`` which calls
        ``resetCovShift`` *first* (zeroing any previously-applied shift
        and recomputing base bounds) and then sets the new shift.  The
        previous Python implementation accumulated shifts: calling
        ``setCovShift(0.2)`` twice produced ``base + 0.4`` instead of
        the intended ``base + 0.2``.  Closes audit finding L1.
        """
        del identifier  # MATLAB-compat positional; not used (see CovColl.m:506).
        self.resetCovShift()
        self.covShift = float(deltaT)
        if np.isfinite(self.minTime):
            self.minTime = float(self.minTime + self.covShift)
        if np.isfinite(self.maxTime):
            self.maxTime = float(self.maxTime + self.covShift)
        return self

    def resetCovShift(self) -> None:
        """Remove the temporal shift and recompute time bounds."""
        self.covShift = 0.0
        self.setMinTime()
        self.setMaxTime()

    def restoreToOriginal(self) -> None:
        """Restore original sample rate, time bounds, shift, and masks."""
        self.covShift = 0.0
        if self.originalSampleRate is not None:
            self.sampleRate = float(self.originalSampleRate)
        else:
            self.sampleRate = self.findMaxSampleRate()
        self.setMinTime(self.findMinTime())
        self.setMaxTime(self.findMaxTime())
        self.resetMask()

    def plot(self, *_, handle=None, **__):
        """Plot each covariate in a vertically stacked panel layout.

        Parameters
        ----------
        handle : matplotlib Figure, Axes, list of Axes, or None
            If a Figure, new subplots are created.
            If a single Axes or a list of Axes, plot into those directly.
            If None, a new figure is created.
        """
        selected = [idx for idx in range(self.numCov)]

        # Accept Figure, Axes, list-of-Axes, or None
        if handle is None:
            fig = plt.figure(figsize=(8.5, max(2.5, 2.2 * max(len(selected), 1))))
            fig.clear()
            axes = fig.subplots(len(selected), 1, sharex=True)
        elif isinstance(handle, plt.Figure):
            fig = handle
            fig.clear()
            axes = fig.subplots(len(selected), 1, sharex=True)
        elif isinstance(handle, (list, np.ndarray)):
            axes = handle
            fig = handle[0].get_figure() if len(handle) else plt.gcf()
        else:
            # Single Axes
            axes = [handle]
            fig = handle.get_figure()

        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes] if not isinstance(axes, list) else axes, dtype=object)
        for ax, cov_index in zip(axes.reshape(-1), selected, strict=False):
            cov = self.getCov(cov_index)
            cov.plot(handle=ax)
            ax.set_title(cov.name)
        fig.tight_layout()
        return fig

    def getAllCovLabels(self) -> list[str]:
        """Return the data-labels of every covariate dimension (no mask filtering)."""
        labels: list[str] = []
        for index in range(self.numCov):
            labels.extend(self.getCov(index).dataLabels)
        return labels

    def getCovLabelsFromMask(self) -> list[str]:
        """Return data-labels only for dimensions that are currently unmasked."""
        labels: list[str] = []
        for index in range(self.numCov):
            cov = self.getCov(index)
            mask = self.covMask[index]
            labels.extend([label for keep, label in zip(mask, cov.dataLabels) if keep == 1])
        return labels

    def getCovDimension(self, identifier=None) -> np.ndarray:
        """Return the dimension of each covariate selected by *identifier*.

        Matlab signature: ``dim = getCovDimension(ccObj, identifier)``

        Returns a 1-D int array whose *i*-th element is ``covs{i}.dimension``.
        """
        if identifier is None:
            covs = [self.getCov(i) for i in range(self.numCov)]
        elif isinstance(identifier, (int, np.integer)):
            covs = [self.getCov(int(identifier))]
        elif isinstance(identifier, (list, np.ndarray)):
            covs = [self.getCov(int(idx)) for idx in identifier]
        else:
            covs = [self.getCov(identifier)]
        return np.array([int(c.dimension) for c in covs], dtype=int)

    def matrixWithTime(self, repType: str = "standard", dataSelector=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return ``(time, data_matrix, labels)`` for active covariate dimensions.

        Parameters
        ----------
        repType : {'standard', 'zero-mean'}
            Signal representation type.
        dataSelector : optional
            Name-based or numeric selector; ``None`` uses the current mask.
        """
        if self.numCov == 0:
            raise ValueError("CovariateCollection is empty")
        if dataSelector is None:
            selectorCell = self.getSelectorFromMasks() if self.isCovMaskSet() else [
                list(range(self.getCov(i).dimension)) for i in range(self.numCov)
            ]
        else:
            selectorCell = self.generateSelectorCell(dataSelector)

        active_cov = [i for i, selector in enumerate(selectorCell) if selector]
        if not active_cov:
            time = self.getCov(0).time
            return time.copy(), np.zeros((time.size, 0), dtype=float), []

        time = self.getCov(active_cov[0]).getSigRep(repType).time
        parts: list[np.ndarray] = []
        labels: list[str] = []
        for covIndex in active_cov:
            cov = self.getCov(covIndex).getSigRep(repType)
            selector = selectorCell[covIndex]
            data = cov.dataToMatrix(selector)
            endInd = min(time.size, data.shape[0])
            block = np.zeros((time.size, data.shape[1]), dtype=float)
            block[:endInd, :] = data[:endInd, :]
            parts.append(block)
            labels.extend([cov.dataLabels[idx] for idx in selector])
        return time.copy(), np.hstack(parts) if parts else np.zeros((time.size, 0), dtype=float), labels

    def dataToMatrix(self, repType: str | Sequence[str] | None = "standard", dataSelector=None, *_) -> np.ndarray:
        """Return the covariate data matrix (no time column) for active dimensions."""
        if repType not in {"standard", "zero-mean"}:
            dataSelector = repType
            repType = "standard"
        _, matrix, _ = self.matrixWithTime(str(repType), dataSelector)
        return matrix

    def dataToStructure(
        self,
        selectorCell=None,
        binwidth: float | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> dict[str, Any]:
        """Serialize active covariate data to a ``{'time': ..., 'signals': ...}`` dict."""
        del binwidth, minTime, maxTime
        if selectorCell is None:
            if self.isCovMaskSet():
                selectorCell = self.getSelectorFromMasks()
            else:
                selectorCell = [list(range(self.getCov(i).dimension)) for i in range(self.numCov)]
        dataMatrix = self.dataToMatrix("standard", selectorCell)
        return {
            "time": self.getCov(0).time.copy() if self.numCov else np.array([], dtype=float),
            "signals": {"values": dataMatrix},
        }

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``CovColl.toStructure``)."""
        self.resetMask()
        structure: dict[str, Any] = {
            "numCov": int(self.numCov),
            "minTime": float(self.minTime),
            "maxTime": float(self.maxTime),
            "covDimensions": [int(value) for value in self.covDimensions],
            "covMask": [np.asarray(mask, dtype=int).reshape(-1).tolist() for mask in self.covMask],
            "covShift": float(self.covShift),
            "sampleRate": float(self.sampleRate) if np.isfinite(self.sampleRate) else self.sampleRate,
            "originalSampleRate": self.originalSampleRate,
            "originalMinTime": self.originalMinTime,
            "originalMaxTime": self.originalMaxTime,
            "covArray": [cov.toStructure() for cov in self.covArray],
        }
        return structure

    @staticmethod
    def fromStructure(structure) -> "CovariateCollection" | list["CovariateCollection"]:
        """Reconstruct from a dict produced by :meth:`toStructure`."""
        if isinstance(structure, list):
            return [CovariateCollection.fromStructure(item) for item in structure]
        if not isinstance(structure, dict):
            raise TypeError("CovColl.fromStructure expects a dictionary or list of dictionaries.")
        cov = [
            Covariate(
                row["time"],
                row["data"],
                row.get("name", ""),
                row.get("xlabelval", "time"),
                row.get("xunits", "s"),
                row.get("yunits", ""),
                row.get("dataLabels"),
                row.get("plotProps"),
            )
            for row in structure.get("covArray", [])
        ]
        ccObj = CovariateCollection(cov)
        if "minTime" in structure:
            ccObj.setMinTime(float(structure["minTime"]))
        if "maxTime" in structure:
            ccObj.setMaxTime(float(structure["maxTime"]))
        return ccObj


class SpikeTrainCollection:
    """Ordered collection of :class:`~nstat.core.nspikeTrain` objects (Matlab ``nstColl``).

    Provides a neuron mask, neighbour graph, and methods for PSTH,
    GLM-PSTH, state-space GLM, raster plots, and data-matrix export.
    Spike trains are accessed via 0-based indexing (``getNST(1)``) to
    match Matlab conventions.

    Parameters
    ----------
    trains : nspikeTrain, sequence of nspikeTrain, or None
        Initial spike train(s) to add.

    See Also
    --------
    nspikeTrain : Single-neuron point-process representation.
    Trial : Combines a ``SpikeTrainCollection`` with covariates.
    """

    def __init__(self, trains: Sequence[nspikeTrain] | nspikeTrain | None = None) -> None:
        """Construct an ordered spike-train collection (Matlab ``nstColl``).

        Parameters
        ----------
        trains : nspikeTrain, sequence of nspikeTrain, or None
            Initial spike train(s) to add.  ``None`` creates an empty
            collection.

        Notes
        -----
        On construction the collection captures the maximal sample rate
        across all input trains (in **Hz**) and the union of their
        observation windows (in **seconds**), and initialises the neuron
        mask to all ones (no neurons masked out).

        Indexing is **0-based** to match MATLAB (``coll.getNST(0)`` is
        the first spike train).  Trains added after construction are
        deep-copied (via :meth:`nspikeTrain.nstCopy`) to prevent shared
        mutable state.

        See Also
        --------
        nspikeTrain : Single-neuron point-process representation.
        Trial : Combines a ``SpikeTrainCollection`` with covariates.
        """
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
        """Number of spike trains in this collection."""
        return self.numSpikeTrains

    @property
    def neuronNames(self) -> list[str]:
        """Neuron name for each spike train in the collection.

        Mirrors the MATLAB ``neuronNames`` stored property.  In Python
        this is derived dynamically from each train's ``.name`` attribute
        so it is always consistent with the underlying data.
        """
        return self.getNSTnames()

    @property
    def uniqueNeuronNames(self) -> list[str]:
        """Unique, insertion-ordered neuron names in the collection."""
        return self.getUniqueNSTnames()

    def __iter__(self):
        for tr in self.nstrain:
            yield tr

    def __len__(self) -> int:
        return int(self.numSpikeTrains)

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
        """Append a single spike train (deep-copied) to the collection."""
        train = nst.nstCopy()
        if not getattr(train, "name", ""):
            train.setName(str(self.numSpikeTrains))
        if self.numSpikeTrains == 0:
            self.minTime = float(train.minTime)
            self.maxTime = float(train.maxTime)
            self.sampleRate = float(train.sampleRate)
        else:
            self.updateTimes(train)
            old_rate = float(self.sampleRate)
            new_rate = float(max(old_rate, float(train.sampleRate)))
            self.sampleRate = new_rate
            # Only resample existing trains when the collection rate just
            # increased — otherwise existing trains are already aligned.
            # This makes append amortized O(1) instead of O(n) per add.
            if round(new_rate, 9) != round(old_rate, 9):
                for stored in self.nstrain:
                    if round(float(stored.sampleRate), 9) != round(new_rate, 9):
                        stored.resample(new_rate)
            # The new train itself may need to resample up to the collection
            # rate (it is being mutated as part of the join contract).
            if round(float(train.sampleRate), 9) != round(new_rate, 9):
                train.resample(new_rate)
        self.nstrain.append(train)
        self.numSpikeTrains = len(self.nstrain)
        self.neuronMask = np.append(self.neuronMask, 1).astype(int)
        if self.numSpikeTrains == 1:
            self.neighbors = []

    def addToColl(self, nst: Sequence[nspikeTrain] | nspikeTrain | "SpikeTrainCollection") -> None:
        """Add one or more spike trains (or another collection) to this collection."""
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
        """Merge another collection into this one (in-place)."""
        self.addToColl(nstColl2)
        return self

    def length(self) -> int:
        """Return the number of spike trains (Matlab ``nstColl.length``)."""
        return int(self.numSpikeTrains)

    def getFirstSpikeTime(self) -> float:
        """Return the earliest time boundary across all trains."""
        return float(self.minTime)

    def getLastSpikeTime(self) -> float:
        """Return the latest time boundary across all trains."""
        return float(self.maxTime)

    def get_nst(self, idx: int) -> nspikeTrain:
        """Return a spike train by 0-based index (Pythonic API)."""
        if idx < 0 or idx >= self.numSpikeTrains:
            raise IndexError("SpikeTrainCollection index out of bounds (0-based indexing).")
        return self.nstrain[idx]

    def getNST(self, idx) -> nspikeTrain | list[nspikeTrain]:
        """Return spike train(s) by 0-based index (Matlab ``nstColl.getNST``).

        Always returns a deep copy of the stored train (never the stored
        reference itself).  This is intentional — the PR #80 ``non-destructive``
        contract for ``getNST`` was previously implemented as
        ``copy-only-when-rates-differ``, which meant callers in the common
        path got a reference and could silently mutate the stored train via
        attribute assignment.  All in-place collection-wide operations
        (e.g. ``enforceSampleRate``) access ``self.nstrain`` directly and
        bypass this method.
        """
        import copy as _copy
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes, np.ndarray)):
            return [self.getNST(int(item)) for item in idx]
        index = int(idx)
        if index < 0 or index >= self.numSpikeTrains:
            raise IndexError("nstColl index out of bounds.")
        nst = _copy.deepcopy(self.nstrain[index])
        # Matlab resamples to collection sampleRate on retrieval.
        if nst.sampleRate != self.sampleRate:
            nst.resample(self.sampleRate)
        return nst

    def getNSTnames(self, selectorArray=None) -> list[str]:
        """Return neuron names, optionally filtered by *selectorArray* (0-based indices)."""
        all_names = [train.name for train in self.nstrain]
        if selectorArray is None:
            # Default: return names for all neurons in the mask
            indices = [i for i, m in enumerate(self.neuronMask) if m]
        else:
            indices = [int(idx) for idx in np.asarray(selectorArray, dtype=int).reshape(-1)]
        return [all_names[i] for i in indices if 0 <= i < len(all_names)]

    def getUniqueNSTnames(self, selectorArray=None) -> list[str]:
        """Return unique, insertion-ordered neuron names."""
        names = [name for name in self.getNSTnames(selectorArray) if name]
        return list(dict.fromkeys(names))

    def getNSTIndicesFromName(self, name: Sequence[str] | str):
        """Return 0-based index(es) for a neuron name (or list of names)."""
        if isinstance(name, str):
            matches = [i for i, value in enumerate(self.getNSTnames()) if value == name]
            if not matches:
                raise KeyError(f"Neuron '{name}' not found")
            return matches if len(matches) > 1 else matches[0]
        return [self.getNSTIndicesFromName(item) for item in name]

    def getNSTnameFromInd(self, ind: int) -> str:
        """Return the neuron name for 0-based index *ind*."""
        index = int(ind)
        if index < 0 or index >= self.numSpikeTrains:
            raise IndexError("Index is out of bounds!")
        return str(self.nstrain[index].name)

    def getNSTFromName(self, neuronName=None):
        """Return spike train(s) matching the given neuron name(s)."""
        if neuronName is None:
            neuronName = self.getUniqueNSTnames()
        indices = self.getNSTIndicesFromName(neuronName)
        return self.getNST(indices)

    def getFieldVal(self, fieldName: str):
        """Collect a named field from every spike train (Matlab ``nstColl.getFieldVal``)."""
        fieldVal: list[float] = []
        neuronNumbers: list[int] = []
        cnt = 0
        for index in range(self.numSpikeTrains):
            currVal = self.getNST(index).getFieldVal(fieldName)
            if currVal is None:
                continue
            if isinstance(currVal, np.ndarray) and currVal.size == 0:
                continue
            if len(fieldVal) <= cnt:
                fieldVal.extend([0.0] * (cnt + 1 - len(fieldVal)))
            fieldVal[cnt] = float(currVal)
            if len(neuronNumbers) <= cnt:
                neuronNumbers.extend([0] * (cnt + 1 - len(neuronNumbers)))
            neuronNumbers[cnt] = index
            cnt += 1
        return np.asarray(fieldVal, dtype=float), np.asarray(neuronNumbers, dtype=int)

    def shiftTime(self, timeShift: float | None = None) -> "SpikeTrainCollection":
        """Return a new collection with spike times shifted by *timeShift*."""
        if timeShift is None:
            timeShift = -float(self.minTime)
        shifted = [nspikeTrain(np.asarray(train.spikeTimes, dtype=float) + float(timeShift)) for train in self.nstrain]
        return SpikeTrainCollection(shifted)

    def toSpikeTrain(
        self,
        selectorArray: Sequence[int] | Sequence[str] | str | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
        windowTimes: Sequence[float] | None = None,
    ) -> nspikeTrain:
        """Collapse selected spike trains into a single :class:`nspikeTrain`.

        Concatenates spike times end-to-end, optionally rescaling
        each trial into windows defined by *windowTimes*.
        """
        if self.numSpikeTrains == 0:
            raise ValueError("nstColl.toSpikeTrain requires at least one spike train")

        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime

        if selectorArray is None:
            selector = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
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
            # The end of the collapsed window equals the sum of each train's
            # duration.  When the collection is homogeneous (the common case
            # after ``nstColl`` construction normalizes ``maxTime``) this
            # collapses to ``N * maxTime``, matching legacy behavior.
            # When trains have heterogeneous ``maxTime`` (bypassing
            # normalization) this still bounds every offset spike correctly.
            collapsed_max = sum(float(t.maxTime) for t in selected_trains)
        else:
            window_arr = np.asarray(windowTimes, dtype=float).reshape(-1)
            if len(selector) != window_arr.size - 1:
                raise ValueError("Window Times must be 1 row longer than selectorArray")
            for idx, train in enumerate(selected_trains):
                local_min = float(window_arr[idx])
                delta_tw = float(window_arr[idx + 1] - local_min)
                if np.asarray(train.spikeTimes).size:
                    spike_times.extend((np.asarray(train.spikeTimes, dtype=float).reshape(-1) * delta_tw + local_min).tolist())
            collapsed_max = float(window_arr[-1])

        collapsed = nspikeTrain(spike_times, name, 1.0 / delta, minTime, collapsed_max, "time", "s", "", "", -1)
        collapsed.setName(name)
        collapsed.setMinTime(float(minTime))
        collapsed.setMaxTime(collapsed_max)
        collapsed.resample(1.0 / max(delta, 1e-12))
        return collapsed

    def setMinTime(self, value: float | None = None) -> None:
        """Set the minimum time for every train in the collection."""
        if value is None:
            value = self.minTime
        for train in self.nstrain:
            train.setMinTime(float(value))
        self.minTime = float(value)

    def setMaxTime(self, value: float | None = None) -> None:
        """Set the maximum time for every train in the collection."""
        if value is None:
            value = self.maxTime
        for train in self.nstrain:
            train.setMaxTime(float(value))
        self.maxTime = float(value)

    def resample(self, sampleRate: float) -> None:
        """Resample all trains to *sampleRate* and align time bounds."""
        self.sampleRate = float(sampleRate)
        for train in self.nstrain:
            train.resample(sampleRate)
            train.setMinTime(float(self.minTime))
            train.setMaxTime(float(self.maxTime))

    def enforceSampleRate(self) -> None:
        """Resample any train whose rate differs from the collection rate.

        Accesses ``self.nstrain`` directly (not via ``getNST``) because the
        intent is in-place mutation of the stored trains.
        """
        target = float(self.sampleRate)
        for train in self.nstrain:
            if round(float(train.sampleRate), 9) != round(target, 9):
                train.resample(target)

    def findMaxSampleRate(self) -> float:
        """Return the highest sample rate among all trains."""
        if self.numSpikeTrains == 0:
            return float("-inf")
        return float(max(train.sampleRate for train in self.nstrain))

    def setMask(self, mask: Sequence[int] | np.ndarray) -> None:
        """Set the neuron mask from a binary array or 0-based indices."""
        arr = np.asarray(mask, dtype=int).reshape(-1)
        if arr.size == self.numSpikeTrains and np.all(np.isin(arr, [0, 1])):
            self.setNeuronMask(arr)
            return
        self.setNeuronMaskFromInd(arr)

    def setNeuronMaskFromInd(self, mask: Sequence[int] | np.ndarray) -> None:
        """Set the neuron mask from 0-based neuron indices."""
        arr = np.asarray(mask, dtype=int).reshape(-1)
        newMask = np.zeros(self.numSpikeTrains, dtype=int)
        if arr.size:
            if np.any(arr < 0) or np.any(arr >= self.numSpikeTrains):
                raise IndexError("Neuron index out of bounds.")
            newMask[arr] = 1
        self.setNeuronMask(newMask)

    def setNeuronMask(self, mask: Sequence[int] | np.ndarray) -> None:
        """Set the binary neuron mask directly (length must match ``numSpikeTrains``)."""
        arr = np.asarray(mask, dtype=int).reshape(-1)
        if arr.size != self.numSpikeTrains:
            raise ValueError("neuronMask length must match number of spike trains.")
        self.neuronMask = arr.astype(int)

    def resetMask(self) -> None:
        """Enable all neurons (ones-mask)."""
        self.neuronMask = np.ones(self.numSpikeTrains, dtype=int)

    def getIndFromMask(self) -> list[int]:
        """Return 0-based indices of neurons currently enabled by the mask."""
        return np.flatnonzero(self.neuronMask == 1).astype(int).tolist()

    def getIndFromMaskMinusOne(self, neuron: int) -> list[int]:
        """Return active indices excluding *neuron* (0-based)."""
        return [idx for idx in self.getIndFromMask() if idx != int(neuron)]

    def isNeuronMaskSet(self) -> bool:
        """Return ``True`` if any neuron is currently masked out."""
        return bool(np.any(self.neuronMask == 0))

    def setNeighbors(self, neighborArray: Sequence[Sequence[int]] | np.ndarray | None = None) -> None:
        """Set or auto-generate the neuron neighbour matrix.

        If *neighborArray* is ``None``, every neuron is a neighbour of
        every other neuron (all-to-all minus self).
        """
        if neighborArray is None:
            if self.numSpikeTrains == 0:
                self.neighbors = []
                return
            matrix = np.zeros((self.numSpikeTrains, max(self.numSpikeTrains - 1, 0)), dtype=int)
            for i in range(self.numSpikeTrains):
                neighbors = [idx for idx in range(self.numSpikeTrains) if idx != i]
                if neighbors:
                    matrix[i, : len(neighbors)] = neighbors
            self.neighbors = matrix
            return
        arr = np.asarray(neighborArray, dtype=int)
        if arr.ndim != 2 or arr.shape[0] != self.numSpikeTrains:
            raise ValueError("Neighbor Array is not of appropriate dimensions")
        self.neighbors = arr

    def areNeighborsSet(self) -> bool:
        """Return ``True`` if the neighbour matrix has been initialized."""
        return np.size(self.neighbors) > 0

    def getNeighbors(self, neuronNum: int | Sequence[int]):
        """Return the 0-based neighbour indices for one or more neurons."""
        if isinstance(neuronNum, Sequence) and not isinstance(neuronNum, (str, bytes, np.ndarray)):
            rows = [self.getNeighbors(int(item)) for item in neuronNum]
            if rows and all(len(row) == len(rows[0]) for row in rows):
                return np.asarray(rows, dtype=int)
            return rows
        neuron_idx = int(neuronNum)
        if not self.areNeighborsSet():
            self.setNeighbors()
        if isinstance(self.neighbors, list):
            row = list(self.neighbors[neuron_idx])
        else:
            row = np.asarray(self.neighbors[neuron_idx], dtype=int).reshape(-1).tolist()
        available = set(self.getIndFromMaskMinusOne(neuron_idx))
        return [value for value in row if value in available and value >= 0]

    def getMaxBinSizeBinary(self) -> float:
        """Return the largest bin-width that keeps all active trains binary."""
        selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        if not selectorArray:
            return np.inf
        values = [self.getNST(index).getMaxBinSizeBinary() for index in selectorArray]
        return float(np.min(values))

    def BinarySigRep(self) -> bool:
        """Return ``True`` if every train's signal representation is binary."""
        return bool(all(self.getNST(index).isSigRepBinary() for index in range(self.numSpikeTrains)))

    def isSigRepBinary(self) -> bool:
        """Alias for :meth:`BinarySigRep`."""
        return self.BinarySigRep()

    def dataToMatrix(
        self,
        selectorArray: Sequence[int] | Sequence[str] | str | None = None,
        binwidth: float | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> np.ndarray:
        """Return an ``(nTimeBins, nNeurons)`` binary spike-count matrix."""
        if self.numSpikeTrains == 0:
            return np.zeros((0, 0), dtype=float)
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        if binwidth is None:
            binwidth = 1.0 / self.sampleRate
        if selectorArray is None:
            selector = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        elif isinstance(selectorArray, str) or _is_string_sequence(selectorArray):
            resolved = self.getNSTIndicesFromName(selectorArray)
            if isinstance(resolved, list):
                selector = [int(item) if not isinstance(item, list) else int(item[0]) for item in resolved]
            else:
                selector = [int(resolved)]
        else:
            selector = [int(item) for item in selectorArray]
        if not selector:
            testSig = self.getNST(0).getSigRep(binwidth, minTime, maxTime)
            return np.zeros((testSig.dataToMatrix().shape[0], 0), dtype=float)
        testSig = self.getNST(selector[0]).getSigRep(binwidth, minTime, maxTime)
        dataMat = np.zeros((testSig.dataToMatrix().shape[0], len(selector)), dtype=float)
        for idx, neuron in enumerate(selector):
            sig = self.getNST(neuron).getSigRep(binwidth, minTime, maxTime)
            dataMat[:, idx] = sig.dataToMatrix().reshape(-1)
        return dataMat

    def getEnsembleNeuronCovariates(self, neuronNum: int = 1, neighborIndex=None, windowTimes=None):
        """Build ensemble-history covariates for *neuronNum* from its neighbours."""
        if neighborIndex is None or (
            isinstance(neighborIndex, (list, tuple, np.ndarray)) and np.asarray(neighborIndex).size == 0
        ):
            allNeighbors = self.getNeighbors(neuronNum)
        else:
            allNeighbors = [int(item) for item in np.asarray(neighborIndex, dtype=int).reshape(-1)]
        if windowTimes is None:
            windowTimes = [0.0, 0.001]
        from .history import History

        histObj = windowTimes if isinstance(windowTimes, History) else History(windowTimes)
        ensembleCovariates = histObj.computeHistory(self.getNST(list(range(self.numSpikeTrains))))
        ensembleCovariates.maskAwayAllExcept(allNeighbors)
        self.addNeuronNamesToEnsCovColl(ensembleCovariates)
        return ensembleCovariates

    def addNeuronNamesToEnsCovColl(self, ensembleCovariates: CovariateCollection) -> None:
        """Prefix ensemble-covariate labels with their neuron name."""
        for i in range(ensembleCovariates.numCov):
            tempCov = ensembleCovariates.covArray[i]
            name = self.getNST(i).name
            if not name:
                name = str(i + 1)
            dataLabels = [f"{name}:{label}" if label else str(name) for label in tempCov.dataLabels]
            tempCov.setDataLabels(dataLabels)

    def restoreToOriginal(self, rMask: int = 0) -> None:
        """Restore all trains to their original state; optionally reset the mask."""
        for train in self.nstrain:
            train.restoreToOriginal()
        self._refresh_summary()
        self.sampleRate = self.findMaxSampleRate()
        self.resample(self.sampleRate)
        if rMask == 1:
            self.resetMask()

    def ensureConsistancy(self) -> None:
        """Enforce consistent sample rate and time bounds across all trains."""
        self.enforceSampleRate()
        self.setMinTime()
        self.setMaxTime()

    def updateTimes(self, nst: nspikeTrain) -> None:
        """Expand collection time bounds to include *nst*, or clamp *nst*."""
        if float(nst.minTime) <= float(self.minTime):
            self.setMinTime(float(nst.minTime))
        else:
            nst.setMinTime(float(self.minTime))
        if float(nst.maxTime) >= float(self.maxTime):
            self.setMaxTime(float(nst.maxTime))
        else:
            nst.setMaxTime(float(self.maxTime))

    def plot(self, selectorArray: Sequence[int] | None = None,
             minTime: float | None = None, maxTime: float | None = None,
             handle=None, reverseOrder: bool = False, **__):
        """Plot a spike-train raster.

        Parameters
        ----------
        selectorArray : sequence of int, optional
            0-based indices of neurons to plot.  Defaults to the neuron mask
            (or all neurons if no mask is set).  Matches Matlab positional arg.
        minTime, maxTime : float, optional
            Time window to display.  Defaults to the collection's time span.
        handle : matplotlib Axes, optional
            Axes to plot into.
        reverseOrder : bool
            If ``True``, reverse the display order so the last neuron is at
            the top.  Matches Matlab ``reverseOrderPlot`` parameter.
        """
        if selectorArray is not None and len(selectorArray) > 0:
            selected = [int(x) for x in selectorArray]
        else:
            selected = self.getIndFromMask()
            if not selected:
                selected = list(range(self.numSpikeTrains))
        if reverseOrder:
            selected = list(reversed(selected))
        ax = handle if handle is not None else plt.subplots(1, 1, figsize=(8.0, max(2.5, 0.55 * max(len(selected), 1) + 1.0)))[1]
        ax.clear()
        for row, neuron_index in enumerate(selected, start=1):
            train = self.getNST(neuron_index)
            train.plot(dHeight=0.8, yOffset=float(row), currentHandle=ax)
        ax.set_ylim(0.25, len(selected) + 0.75)
        ax.set_yticks(range(1, len(selected) + 1), [str(item) for item in selected])
        if minTime is not None or maxTime is not None:
            lo = float(minTime) if minTime is not None else float(self.minTime)
            hi = float(maxTime) if maxTime is not None else float(self.maxTime)
            ax.set_xlim(lo, hi)
        ax.set_title("Spike Train Raster")
        return ax

    def getMinISIs(self, selectorArray: Sequence[int] | None = None, minTime: float | None = None, maxTime: float | None = None) -> np.ndarray:
        """Return the minimum ISI for each selected neuron."""
        isis = self.getISIs(selectorArray, minTime, maxTime)
        return np.asarray([float(np.min(values)) if values.size else 0.0 for values in isis], dtype=float)

    def getISIs(self, selectorArray: Sequence[int] | None = None, minTime: float | None = None, maxTime: float | None = None) -> list[np.ndarray]:
        """Return a list of ISI arrays, one per selected neuron."""
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        if selectorArray is None or len(selectorArray) == 0:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        return [self.getNST(int(neuron)).getISIs(minTime, maxTime) for neuron in selectorArray]

    def plotISIHistogram(self, selectorArray: Sequence[int] | None = None, minTime: float | None = None, maxTime: float | None = None, handle=None):
        """Plot ISI histograms for each selected neuron in stacked subplots."""
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        if selectorArray is None or len(selectorArray) == 0:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        fig = handle if handle is not None else plt.figure(figsize=(7.0, max(2.5, 2.2 * len(selectorArray))))
        fig.clear()
        axes = fig.subplots(len(selectorArray), 1)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for ax, neuron in zip(axes.reshape(-1), selectorArray, strict=False):
            self.getNST(int(neuron)).plotISIHistogram(minTime, maxTime, handle=ax)
        fig.tight_layout()
        return fig

    def plotExponentialFit(
        self,
        selectorArray: Sequence[int] | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
        numBins: int | None = None,
        handle=None,
    ):
        """Plot exponential-distribution fits of ISIs for selected neurons."""
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        if selectorArray is None or len(selectorArray) == 0:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        fig = handle if handle is not None else plt.figure(figsize=(7.0, max(2.5, 2.2 * len(selectorArray))))
        fig.clear()
        axes = fig.subplots(len(selectorArray), 1)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for ax, neuron in zip(axes.reshape(-1), selectorArray, strict=False):
            self.getNST(int(neuron)).plotExponentialFit(minTime, maxTime, numBins, handle=ax)
        fig.tight_layout()
        return fig

    def getSpikeTimes(self, minTime: float | None = None, maxTime: float | None = None) -> list[np.ndarray]:
        """Return a list of spike-time arrays, one per active neuron."""
        del minTime, maxTime
        selector = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        return [self.getNST(int(index)).getSpikeTimes() for index in selector]

    def psth(
        self,
        binwidth: float = 0.100,
        selectorArray: Sequence[int] | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> Covariate:
        """Compute the peri-stimulus time histogram (standard binned PSTH).

        Returns a :class:`Covariate` with firing rate in Hz.
        """
        if binwidth <= 0:
            raise ValueError("binwidth must be > 0")
        min_time = self.minTime if minTime is None else float(minTime)
        max_time = self.maxTime if maxTime is None else float(maxTime)
        if selectorArray is None:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))

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
            spike_times = spikes[valid]
            # Mirror MATLAB histc edge semantics on a numerically noisy uniform grid:
            # interior samples land in [edge_k, edge_{k+1}), but samples that match
            # an edge belong to that edge's bin, with the final edge kept in the
            # extra histc bin that is discarded below.
            left = np.searchsorted(window_times, spike_times, side="left")
            right = np.searchsorted(window_times, spike_times, side="right") - 1
            edge_tol = max(1e-12, abs(float(binwidth)) * 1e-9)
            exact_edge = (left < window_times.size) & np.isclose(
                spike_times,
                window_times[np.clip(left, 0, window_times.size - 1)],
                rtol=0.0,
                atol=edge_tol,
            )
            idx = np.where(exact_edge, left, right)
            idx = np.clip(idx, 0, window_times.size - 1)
            psth_hist += np.bincount(idx, minlength=window_times.size).astype(float)

        psth_data = psth_hist[:-1] / binwidth / float(len(selectorArray))
        time = (window_times[1:] + window_times[:-1]) * 0.5
        return Covariate(time, psth_data, "PSTH", "time", "s", "Hz", ["psth"])

    def psthGLM(self, basisWidth: float | None = None, history=None, fitType: str = "poisson",
                selectorArray=None, minTime=None, maxTime=None, sampleRate=None,
                *, binwidth: float | None = None, windowTimes=None,
                alphaVal: float = 0.05, Mc: int = 1000):
        """GLM-based PSTH estimation (Matlab ``nstColl.psthGLM``).

        Parameters match MATLAB:
        ``psthGLM(basisWidth, history, fitType, selectorArray, minTime, maxTime, sampleRate)``

        The Python keyword-only ``binwidth`` and ``windowTimes`` are accepted
        as aliases for ``basisWidth`` and ``history`` respectively.

        Returns ``(psth_covariate, histSignal, psthFitResult)`` matching the
        Matlab signature.
        """
        # Resolve aliases: binwidth ↔ basisWidth, windowTimes ↔ history
        if basisWidth is None and binwidth is not None:
            basisWidth = binwidth
        elif basisWidth is None and binwidth is None:
            basisWidth = 0.100  # MATLAB default
        if windowTimes is not None and history is None:
            history = windowTimes
        windowTimes = history  # use unified name internally
        from .analysis import Analysis
        from .confidence_interval import ConfidenceInterval
        from .glm import fit_poisson_glm

        # Use MATLAB-compatible param names (selectorArray/minTime/maxTime/sampleRate unused in current impl)
        _sr = float(sampleRate) if sampleRate is not None else float(self.sampleRate)
        _minT = float(minTime) if minTime is not None else float(self.minTime)
        _maxT = float(maxTime) if maxTime is not None else float(self.maxTime)
        basis = self.generateUnitImpulseBasis(
            float(basisWidth), _minT, _maxT, _sr
        )
        trial = Trial(
            SpikeTrainCollection([train.nstCopy() for train in self.nstrain]),
            CovariateCollection([basis]),
        )
        hist = [] if windowTimes is None else np.asarray(windowTimes, dtype=float).reshape(-1)
        label_select = [[basis.name, *list(basis.dataLabels)]]
        cfg = TrialConfig(label_select, float(self.sampleRate), hist, [])
        cfg.setName("GLM-PSTH+Hist" if np.asarray(hist).size else "GLM-PSTH")
        cfgColl = ConfigCollection([cfg])
        algorithm = "GLM" if str(fitType or "poisson").lower() == "poisson" else "BNLRCG"

        # ---- Matlab batchMode=1: concatenate Y and X across ALL trials ----
        # Matlab nstColl.psthGLM (line 1003-1004) calls
        #   RunAnalysisForAllNeurons(trial, cfgColl, 0, Algorithm, [], 1)
        # with batchMode=1, which pools all trials of the same neuron into
        # a single GLM fit.  Python's RunAnalysisForAllNeurons previously
        # ignored batchMode, fitting each trial separately — producing
        # single-trial coefficients instead of across-trial pooled ones.
        cfgColl.setConfig(trial, 0)
        stacked_x: list[np.ndarray] = []
        stacked_y: list[np.ndarray] = []
        for idx in range(trial.nspikeColl.num_spike_trains):
            x_i = np.asarray(trial.getDesignMatrix(idx), dtype=float)
            y_i = np.asarray(trial.getSpikeVector(idx), dtype=float).reshape(-1)
            n_obs = min(x_i.shape[0], y_i.shape[0])
            stacked_x.append(x_i[:n_obs])
            stacked_y.append(y_i[:n_obs])
        X = np.vstack(stacked_x)
        y = np.concatenate(stacked_y)

        if algorithm == "GLM":
            glm_res = fit_poisson_glm(X, y, include_intercept=False)
            raw_coeffs = np.asarray(glm_res.coefficients, dtype=float).reshape(-1)
            lambda_hat = glm_res.predict_rate(X)
            W = np.maximum(lambda_hat, 1e-12)
        else:
            from .glm import fit_binomial_glm
            glm_res = fit_binomial_glm(X, y, include_intercept=False)
            raw_coeffs = np.asarray(glm_res.coefficients, dtype=float).reshape(-1)
            lambda_hat = np.clip(glm_res.predict_probability(X), 1e-12, 1.0 - 1e-9)
            W = lambda_hat * (1.0 - lambda_hat)
            W = np.maximum(W, 1e-12)

        # Standard errors from Fisher information (Hessian inverse)
        try:
            XtWX = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
            covb = np.linalg.inv(XtWX)
            se_vec = np.sqrt(np.maximum(np.diag(covb), 0.0))
        except np.linalg.LinAlgError:
            se_vec = np.full(raw_coeffs.size, np.nan, dtype=float)

        # Build a proper FitResult for the third return value by fitting just
        # the first spike train (fast), then override its coefficients with
        # the batch-fit values.
        fit = Analysis.RunAnalysisForNeuron(trial, 0, cfgColl, 0, algorithm)
        if isinstance(fit, list):
            fit = fit[0]
        # Override with batch-fit coefficients and standard errors
        fit.b[0] = raw_coeffs.copy()
        if fit.stats and isinstance(fit.stats[0], dict):
            fit.stats[0]["se"] = se_vec.copy()

        numBasis = basis.dimension

        if raw_coeffs.size < numBasis:
            padded = np.zeros(numBasis, dtype=float)
            padded[: raw_coeffs.size] = raw_coeffs
            bVals = padded
            se_padded = np.full(numBasis, np.nan, dtype=float)
            se_padded[: se_vec.size] = se_vec[:numBasis] if se_vec.size >= numBasis else se_vec
            se_basis = se_padded
        else:
            bVals = raw_coeffs[:numBasis]
            se_basis = se_vec[:numBasis]

        is_poisson = str(fitType or "poisson").lower() == "poisson"
        sr = float(self.sampleRate)

        # basis.data is (nTimeBins x numBasis): multiply to get GLM rate
        bdata = np.asarray(basis.data, dtype=float)
        lambda_glm = np.exp(bdata @ bVals) * sr
        psth_cov = Covariate(
            basis.time.copy(),
            lambda_glm.reshape(-1, 1),
            "GLM-PSTH",
            basis.xlabelval,
            basis.xunits,
            "Hz",
            ["\\lambda_{GLM}"],
        )

        # ---- Monte Carlo confidence intervals for PSTH (Matlab parity) ----
        se_clean = np.where(np.isnan(se_basis), 0.0, se_basis)
        if np.any(se_clean > 0):
            rng = np.random.default_rng()
            z = rng.standard_normal((se_clean.size, Mc))
            xKDraw = bVals[:, None] + se_clean[:, None] * z  # (numBasis, Mc)
            if is_poisson:
                lambdaDraw = np.exp(np.clip(xKDraw, -30, 30)) * sr
            else:
                xc = np.clip(xKDraw, -30, 30)
                lambdaDraw = (np.exp(xc) / (1.0 + np.exp(xc))) * sr
            lambdaDraw = np.where(np.isinf(lambdaDraw), 0.0, lambdaDraw)

            # Per-coefficient empirical quantiles
            CIs = np.column_stack([
                np.quantile(lambdaDraw, alphaVal / 2.0, axis=1),
                np.quantile(lambdaDraw, 1.0 - alphaVal / 2.0, axis=1),
            ])  # (numBasis, 2)
            lower = bdata @ CIs[:, 0]
            upper = bdata @ CIs[:, 1]

            ciPSTHGLM = ConfidenceInterval(
                basis.time, np.column_stack([lower, upper]),
                "CI_{psth_GLM}", psth_cov.xlabelval, psth_cov.xunits, "Hz",
            )
            psth_cov.setConfInterval(ciPSTHGLM)

        # ---- History signal (only present when windowTimes is specified) ----
        histSignal = None
        if np.asarray(hist).size and raw_coeffs.size > numBasis:
            histVals = raw_coeffs[numBasis:]
            se_hist = se_vec[numBasis:] if se_vec.size > numBasis else np.zeros_like(histVals)

            # Build piecewise-constant basis for history time axis (Matlab style)
            selfHist = np.asarray(hist, dtype=float).reshape(-1)
            histTime = np.arange(0.0, float(np.max(selfHist)) + 0.001, 0.001)
            nHistBins = len(selfHist) - 1
            if len(histTime) > 0 and nHistBins > 0:
                basisMat = np.zeros((len(histTime), nHistBins), dtype=float)
                for i in range(nHistBins):
                    if i == nHistBins - 1:
                        col = (histTime >= selfHist[i]) & (histTime <= selfHist[i + 1])
                    else:
                        col = (histTime >= selfHist[i]) & (histTime < selfHist[i + 1])
                    basisMat[:, i] = col.astype(float)

                expHistVals = np.exp(histVals[:nHistBins])
                histSignal = Covariate(
                    histTime, (basisMat @ expHistVals).reshape(-1, 1),
                    "PSTH_{glm}", "time", "s", "Hz",
                )

                # Monte Carlo CIs for history signal
                se_h_clean = np.where(np.isnan(se_hist[:nHistBins]), 0.0, se_hist[:nHistBins])
                if np.any(se_h_clean > 0):
                    rng2 = np.random.default_rng()
                    z2 = rng2.standard_normal((se_h_clean.size, Mc))
                    # Matlab centers on zero for history CIs (variability around null)
                    xKDrawH = se_h_clean[:, None] * z2
                    if is_poisson:
                        histDraw = np.exp(np.clip(xKDrawH, -30, 30)) * sr
                    else:
                        xc2 = np.clip(xKDrawH, -30, 30)
                        histDraw = (np.exp(xc2) / (1.0 + np.exp(xc2))) * sr
                    CIsH = np.column_stack([
                        np.quantile(histDraw, alphaVal / 2.0, axis=1),
                        np.quantile(histDraw, 1.0 - alphaVal / 2.0, axis=1),
                    ])
                    lowerH = basisMat @ CIsH[:, 0]
                    upperH = basisMat @ CIsH[:, 1]
                    ciHist = ConfidenceInterval(
                        histTime, np.column_stack([lowerH, upperH]),
                        "CI_{psth_GLMHIST}", psth_cov.xlabelval, psth_cov.xunits, "Hz",
                    )
                    histSignal.setConfInterval(ciHist)
            else:
                histSignal = Covariate(
                    np.arange(len(histVals), dtype=float),
                    histVals.reshape(-1, 1),
                    "History", "lag", "bins", "", ["h"],
                )

        return psth_cov, histSignal, fit

    def psthBars(
        self,
        binwidth: float = 0.100,
        selectorArray: Sequence[int] | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> SignalObj:
        """Deterministic pure-Python fallback for MATLAB nstColl.psthBars.

        MATLAB delegates this method to an external BARS package that is not
        bundled with the source tree. The Python port preserves the public
        surface and return structure with a smoothed PSTH approximation.
        """
        if binwidth <= 0:
            raise ValueError("binwidth must be > 0")
        min_time = self.minTime if minTime is None else float(minTime)
        max_time = self.maxTime if maxTime is None else float(maxTime)
        if selectorArray is None or len(selectorArray) == 0:
            selectorArray = self.getIndFromMask() if self.isNeuronMaskSet() else list(range(self.numSpikeTrains))
        selector = [int(item) for item in selectorArray]
        if not selector:
            raise ValueError("selectorArray must contain at least one neuron")

        time = np.arange(min_time, max_time + float(binwidth), float(binwidth), dtype=float)
        if time.size == 0:
            time = np.array([min_time, max_time], dtype=float)
        if not np.isclose(time[-1], max_time):
            if time[-1] < max_time:
                time = np.append(time, max_time)
            else:
                time[-1] = max_time

        psthData = np.zeros(time.size, dtype=float)
        for neuron in selector:
            spikeTimes = np.asarray(self.getNST(neuron).getSpikeTimes(), dtype=float).reshape(-1)
            if spikeTimes.size == 0:
                continue
            valid = np.isfinite(spikeTimes) & (spikeTimes >= time[0]) & (spikeTimes <= time[-1])
            if not np.any(valid):
                continue
            spikeTimes = spikeTimes[valid]
            left = np.searchsorted(time, spikeTimes, side="left")
            right = np.searchsorted(time, spikeTimes, side="right") - 1
            edge_tol = max(1e-12, abs(float(binwidth)) * 1e-9)
            exact_edge = (left < time.size) & np.isclose(
                spikeTimes,
                time[np.clip(left, 0, time.size - 1)],
                rtol=0.0,
                atol=edge_tol,
            )
            idx = np.where(exact_edge, left, right)
            idx = np.clip(idx, 0, time.size - 1)
            psthData += np.bincount(idx, minlength=time.size).astype(float)

        psthData = psthData / float(binwidth) / float(len(selector))

        # MATLAB uses an external BARS fitter here; preserve the public output
        # structure with a deterministic smoothed-rate fallback.
        if psthData.size >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=float)
            mean_curve = np.convolve(psthData, kernel, mode="same")
        else:
            mean_curve = psthData.copy()
        mode_curve = mean_curve.copy()
        counts_per_bin = np.maximum(mean_curve * float(binwidth) * float(len(selector)), 0.0)
        stderr = np.sqrt(counts_per_bin) / max(float(binwidth) * float(len(selector)), 1e-12)
        ciLower = np.maximum(mean_curve - 1.96 * stderr, 0.0)
        ciUpper = mean_curve + 1.96 * stderr
        data = np.column_stack([mode_curve, mean_curve, ciLower, ciUpper])
        return SignalObj(
            time,
            data,
            "PSTH_{bars}",
            "time",
            "s",
            "Hz",
            ["mode", "mean", "ciLower", "ciUpper"],
        )

    def _psth_glm_coeffs(
        self,
        basisWidth: float,
        windowTimes=None,
        fitType: str = "poisson",
    ) -> np.ndarray:
        from .analysis import Analysis

        basis = self.generateUnitImpulseBasis(float(basisWidth), float(self.minTime), float(self.maxTime), float(self.sampleRate))
        trial = Trial(SpikeTrainCollection([train.nstCopy() for train in self.nstrain]), CovariateCollection([basis]))
        hist = [] if windowTimes is None else np.asarray(windowTimes, dtype=float).reshape(-1)
        label_select = [[basis.name, *list(basis.dataLabels)]]
        cfg = TrialConfig(label_select, float(self.sampleRate), hist, [])
        cfg.setName("GLM-PSTH+Hist" if np.asarray(hist).size else "GLM-PSTH")
        cfgColl = ConfigCollection([cfg])
        algorithm = "GLM" if str(fitType or "poisson").lower() == "poisson" else "BNLRCG"
        psth_result = Analysis.RunAnalysisForAllNeurons(trial, cfgColl, 0, algorithm, [], 1)
        fit = psth_result[0] if isinstance(psth_result, list) else psth_result
        coeffs = fit._rawCoeffs(1)
        numBasis = basis.dimension
        if coeffs.size < numBasis:
            padded = np.zeros(numBasis, dtype=float)
            padded[: coeffs.size] = coeffs
            return padded
        return coeffs[:numBasis]

    def estimateVarianceAcrossTrials(
        self,
        numBasis: int | None = None,
        windowTimes=None,
        numIter: int | None = None,
        fitType: str | None = None,
    ) -> np.ndarray:
        """Estimate the state-noise covariance ``Q`` from bootstrap GLM fits.

        Used internally by :meth:`ssglm` / :meth:`ssglmFB` to initialise
        the EM algorithm's state-noise prior.
        """
        if fitType is None or fitType == "":
            fitType = "poisson"
        if numIter is None:
            numIter = 20
        if windowTimes is None:
            windowTimes = []
        if numBasis is None:
            numBasis = 20

        numBasis = int(numBasis)
        numIter = int(numIter)
        coeffs = np.zeros((numBasis, numIter), dtype=float)
        numRealizations = int(self.numSpikeTrains)
        if numRealizations == 0 or numBasis <= 0 or numIter <= 0:
            return np.zeros((max(numBasis, 0), max(numBasis, 0)), dtype=float)

        basisWidth = (float(self.maxTime) - float(self.minTime)) / float(numBasis)
        sumNumber = max(int(np.floor(numRealizations / 2.0 - 1.0)), 0)
        delta = 1.0 / float(self.sampleRate)
        minTime = float(self.minTime)
        maxTime = float(self.maxTime)
        halfIters = min(int(np.floor(numIter / 2.0)), sumNumber)

        for i in range(halfIters):
            subset = SpikeTrainCollection(self.getNST(list(range(i, i + sumNumber + 1))))
            subset.resample(1.0 / delta)
            subset.setMaxTime(maxTime)
            subset.setMinTime(minTime)
            coeffs[:, i] = subset._psth_glm_coeffs(basisWidth, windowTimes, fitType)

        for i in range(numRealizations - 1, numRealizations - halfIters - 1, -1):
            subset = SpikeTrainCollection(self.getNST(list(range(i, i - sumNumber - 1, -1))))
            subset.resample(1.0 / delta)
            subset.setMaxTime(maxTime)
            subset.setMinTime(minTime)
            coeffs[:, i] = subset._psth_glm_coeffs(basisWidth, windowTimes, fitType)

        coeff_rows = [row[row != 0] for row in coeffs]
        max_width = max((row.size for row in coeff_rows), default=0)
        if max_width == 0:
            return np.zeros((numBasis, numBasis), dtype=float)
        coeffsTemp = np.full((numBasis, max_width), np.nan, dtype=float)
        for idx, row in enumerate(coeff_rows):
            coeffsTemp[idx, : row.size] = row

        nTerms = 4
        filt_num = np.ones(nTerms, dtype=float) / float(nTerms)
        coeffsTemp[np.isnan(coeffsTemp)] = 0.0
        if coeffsTemp.T.shape[0] > 3 * nTerms:
            from scipy.signal import filtfilt  # lazy: avoid scipy on import
            fcoeffs = filtfilt(filt_num, [1.0], coeffsTemp.T, axis=0).T
        else:
            fcoeffs = coeffsTemp

        diffs = np.diff(fcoeffs, axis=1)
        if diffs.shape[1] <= 1:
            varEst = np.full(numBasis, np.nan, dtype=float)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                varEst = np.nanvar(diffs, axis=1, ddof=1)
        return np.diag(varEst)

    @staticmethod
    def generateUnitImpulseBasis(basisWidth: float, minTime: float, maxTime: float, sampleRate: float = 1000.0) -> Covariate:
        """Create a piecewise-constant (unit impulse) basis :class:`Covariate`.

        Each column is a rectangular pulse spanning one *basisWidth*
        interval, used as the design matrix for GLM-PSTH estimation.
        """
        windowTimes = np.arange(float(minTime), float(maxTime), float(basisWidth))
        if windowTimes.size == 0 or not np.isclose(windowTimes[-1], maxTime):
            windowTimes = np.append(windowTimes, float(maxTime))
        else:
            windowTimes[-1] = float(maxTime)
        if windowTimes.size < 2:
            windowTimes = np.array([float(minTime), float(maxTime)], dtype=float)
        timeVec = np.arange(float(minTime), float(maxTime) + (1.0 / float(sampleRate)), 1.0 / float(sampleRate))
        dataMat = np.zeros((timeVec.size, windowTimes.size - 1), dtype=float)
        dataLabels: list[str] = []
        for i in range(windowTimes.size - 1):
            start = float(windowTimes[i])
            stop = float(windowTimes[i + 1])
            if i == windowTimes.size - 2:
                dataMat[:, i] = ((timeVec >= start) & (timeVec <= stop)).astype(float)
            else:
                dataMat[:, i] = ((timeVec >= start) & (timeVec < stop)).astype(float)
            dataLabels.append(f"b{i + 1:02d}" if i + 1 < 10 else f"b{i + 1}")
        return Covariate(timeVec, dataMat, "UnitPulseBasis", "time", "s", "", dataLabels)

    def ssglm(
        self,
        windowTimes=None,
        numBasis: int | None = None,
        numVarEstIter: int | None = None,
        fitType: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        """State-space GLM via EM algorithm (forward only).

        Matches Matlab nstColl.ssglm(). Estimates time-varying firing rate
        using a state-space model with EM parameter estimation.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator used to jitter zero entries in the
            initial variance estimate.  Defaults to
            ``np.random.default_rng()`` (non-reproducible across runs).
            Pass an explicit seeded generator for reproducible fits.

        Parameters
        ----------
        windowTimes : array-like or None
            History window boundaries. None for no history.
        numBasis : int or None
            Number of basis functions. Defaults to duration/0.02.
        numVarEstIter : int or None
            Iterations for variance estimation. Default 10.
        fitType : 'poisson' or 'binomial'

        Returns
        -------
        xK : (R, K) estimated state trajectories
        WK : (R, R, K) estimated state covariances
        Qhat : (R,) estimated state noise variance
        gammahat : (J,) estimated history coefficients
        logll : float, log-likelihood
        fitResults : FitResult
        """
        from .decoding_algorithms import DecodingAlgorithms

        if fitType is None or fitType == "":
            fitType = "poisson"
        if numVarEstIter is None:
            numVarEstIter = 10
        if numBasis is None:
            basisWidth = 0.02
            numBasis = max(1, int((self.maxTime - self.minTime) / basisWidth))

        # Convert spike trains to binary observation matrix (K x N)
        dN = self.dataToMatrix().T  # dataToMatrix returns (N, K), transpose to (K, N)
        dN = np.clip(dN, 0, 1)  # binarize
        K, N = dN.shape

        delta = 1.0 / float(self.sampleRate)
        basisWidth = (float(self.maxTime) - float(self.minTime)) / float(numBasis)

        # Get initial coefficients from GLM PSTH
        x0 = self._psth_glm_coeffs(basisWidth, windowTimes, fitType)
        if x0.size < numBasis:
            x0 = np.concatenate([x0, np.zeros(numBasis - x0.size)])
        elif x0.size > numBasis:
            x0 = x0[:numBasis]

        # Get initial history coefficients
        if windowTimes is not None and len(windowTimes) > 1:
            try:
                from .analysis import Analysis
                basis = self.generateUnitImpulseBasis(basisWidth, float(self.minTime), float(self.maxTime), float(self.sampleRate))
                trial = Trial(SpikeTrainCollection([t.nstCopy() for t in self.nstrain]), CovariateCollection([basis]))
                hist_arr = np.asarray(windowTimes, dtype=float).reshape(-1)
                label_sel = [[basis.name, *list(basis.dataLabels)]]
                cfg = TrialConfig(label_sel, float(self.sampleRate), hist_arr, [])
                cfg.setName("GLM-PSTH+Hist")
                cfgColl = ConfigCollection([cfg])
                psthResult = Analysis.RunAnalysisForAllNeurons(trial, cfgColl, 0, "GLM", [], 1)
                fit = psthResult[0] if isinstance(psthResult, list) else psthResult
                gamma0 = np.asarray(fit.getHistCoeffs(0)[0], dtype=float).reshape(-1)
                gamma0 = np.where(np.isnan(gamma0), -5.0, gamma0)
            except Exception:
                numHist = len(windowTimes) - 1
                gamma0 = np.full(numHist, -5.0, dtype=float)
        else:
            gamma0 = np.array([], dtype=float)

        # Estimate initial Q0
        Q0 = self.estimateVarianceAcrossTrials(numBasis, windowTimes, numVarEstIter, fitType)
        Q0_diag = np.diag(Q0)
        zero_mask = Q0_diag == 0
        if np.any(zero_mask):
            # Jitter only the zero entries, leaving well-estimated variances
            # untouched.  Uses default_rng (modern NumPy RNG, reproducible
            # when caller passes a seeded generator).
            _rng = rng if rng is not None else np.random.default_rng()
            jitter = 0.001 * _rng.random(int(zero_mask.sum()))
            Q0_diag = Q0_diag.astype(float, copy=True)
            Q0_diag[zero_mask] = jitter

        A = np.eye(numBasis)

        # Build history matrices
        HkAll = DecodingAlgorithms._ssglm_build_history(dN, windowTimes, delta)

        # Run EM
        xK, WK, Wku, Qhat, gammahat, logll, _, _, nIter, _ = DecodingAlgorithms.PPSS_EM(
            A, Q0_diag, x0, dN, fitType, delta, gamma0, windowTimes, numBasis, HkAll
        )

        # Package results
        fitResults = DecodingAlgorithms.prepareEMResults(
            fitType, self.name if hasattr(self, 'name') else 'N01',
            dN, HkAll, xK, WK, Qhat, gammahat, windowTimes, delta,
            np.eye(Qhat.size + gammahat.size), logll
        )

        return xK, WK, Qhat, gammahat, logll, fitResults

    def ssglmFB(
        self,
        windowTimes=None,
        numBasis: int | None = None,
        numVarEstIter: int | None = None,
        fitType: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        """State-space GLM via EM Forward-Backward algorithm.

        Enhanced version of ssglm() that uses forward-backward-forward
        iterations for improved convergence. Calls PPSS_EMFB.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for Q0 jitter; see :meth:`ssglm`.

        Parameters
        ----------
        windowTimes : array-like or None
            History window boundaries.
        numBasis : int or None
            Number of basis functions.
        numVarEstIter : int or None
            Iterations for variance estimation.
        fitType : 'poisson' or 'binomial'

        Returns
        -------
        xK, WK, Wku, Qhat, gammahat, fitResults, stimulus, stimCIs, logll,
        QhatAll, gammahatAll, nIter
        """
        from .decoding_algorithms import DecodingAlgorithms

        if fitType is None or fitType == "":
            fitType = "poisson"
        if numVarEstIter is None:
            numVarEstIter = 10
        if numBasis is None:
            basisWidth = 0.02
            numBasis = max(1, int((self.maxTime - self.minTime) / basisWidth))

        dN = self.dataToMatrix().T
        dN = np.clip(dN, 0, 1)

        delta = 1.0 / float(self.sampleRate)
        basisWidth = (float(self.maxTime) - float(self.minTime)) / float(numBasis)

        x0 = self._psth_glm_coeffs(basisWidth, windowTimes, fitType)
        if x0.size < numBasis:
            x0 = np.concatenate([x0, np.zeros(numBasis - x0.size)])
        elif x0.size > numBasis:
            x0 = x0[:numBasis]

        if windowTimes is not None and len(windowTimes) > 1:
            try:
                from .analysis import Analysis
                basis = self.generateUnitImpulseBasis(basisWidth, float(self.minTime), float(self.maxTime), float(self.sampleRate))
                trial = Trial(SpikeTrainCollection([t.nstCopy() for t in self.nstrain]), CovariateCollection([basis]))
                hist_arr = np.asarray(windowTimes, dtype=float).reshape(-1)
                label_sel = [[basis.name, *list(basis.dataLabels)]]
                cfg = TrialConfig(label_sel, float(self.sampleRate), hist_arr, [])
                cfg.setName("GLM-PSTH+Hist")
                cfgColl = ConfigCollection([cfg])
                psthResult = Analysis.RunAnalysisForAllNeurons(trial, cfgColl, 0, "GLM", [], 1)
                fit = psthResult[0] if isinstance(psthResult, list) else psthResult
                gamma0 = np.asarray(fit.getHistCoeffs(0)[0], dtype=float).reshape(-1)
                gamma0 = np.where(np.isnan(gamma0), -5.0, gamma0)
            except Exception:
                numHist = len(windowTimes) - 1
                gamma0 = np.full(numHist, -5.0, dtype=float)
        else:
            gamma0 = np.array([], dtype=float)

        Q0 = self.estimateVarianceAcrossTrials(numBasis, windowTimes, numVarEstIter, fitType)
        Q0_diag = np.diag(Q0)
        zero_mask = Q0_diag == 0
        if np.any(zero_mask):
            _rng = rng if rng is not None else np.random.default_rng()
            jitter = 0.001 * _rng.random(int(zero_mask.sum()))
            Q0_diag = Q0_diag.astype(float, copy=True)
            Q0_diag[zero_mask] = jitter

        A = np.eye(numBasis)
        neuronName = self.name if hasattr(self, 'name') else 'N01'

        return DecodingAlgorithms.PPSS_EMFB(
            A, Q0_diag, x0, dN, fitType, delta, gamma0, windowTimes, numBasis, neuronName
        )

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``nstColl.toStructure``)."""
        self.resetMask()
        return {
            "nstrain": [train.toStructure() for train in self.nstrain],
            "numSpikeTrains": int(self.numSpikeTrains),
            "minTime": float(self.minTime),
            "maxTime": float(self.maxTime),
            "sampleRate": float(self.sampleRate),
            "neuronMask": self.neuronMask.tolist(),
            "neuronNames": self.neuronNames,
            "neighbors": np.asarray(self.neighbors, dtype=int).tolist() if np.size(self.neighbors) else [],
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "SpikeTrainCollection":
        """Reconstruct from a dict produced by :meth:`toStructure` (Matlab ``nstColl.fromStructure``)."""
        nst_list = [nspikeTrain.fromStructure(item) for item in structure.get("nstrain", [])]
        coll = SpikeTrainCollection(nst_list)
        if "minTime" in structure:
            coll.setMinTime(float(structure["minTime"]))
        if "maxTime" in structure:
            coll.setMaxTime(float(structure["maxTime"]))
        neighbors = structure.get("neighbors", [])
        if neighbors and np.size(neighbors):
            coll.setNeighbors(np.asarray(neighbors, dtype=int))
        return coll



class Trial:
    """Single-trial data container binding spikes, covariates, and events (Matlab ``Trial``).

    A ``Trial`` enforces consistent time bounds and sample rate across
    its spike collection, covariate collection, and optional event stream.
    It provides the design-matrix construction used by :class:`Analysis`
    to fit point-process GLMs.

    Parameters
    ----------
    spike_collection : SpikeTrainCollection
        Neural spike data.
    covariate_collection : CovariateCollection
        Stimulus or task covariates.
    events : Events, optional
        Discrete event markers.
    hist : History or array_like, optional
        Self-history specification.
    ensCovHist : History or array_like, optional
        Ensemble-history specification.
    ensCovMask : array_like, optional
        Binary mask for ensemble neighbours.

    See Also
    --------
    SpikeTrainCollection, CovariateCollection, Analysis
    """

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
        """Construct a Trial bundling spikes, covariates, and events (Matlab ``Trial``).

        Parameters
        ----------
        spike_collection : SpikeTrainCollection
            Neural spike data.  **Required.**  Also accepted as the
            MATLAB-style keyword ``spikeColl=``.
        covariate_collection : CovariateCollection
            Stimulus or task covariates.  **Required.**  Also accepted
            as the MATLAB-style keyword ``covarColl=``.
        events : Events, optional
            Discrete event markers (e.g. trial onsets).  Also accepted
            as the MATLAB-style keyword ``event=``.
        hist : History or array_like, optional
            Self-history specification.  May be a
            :class:`~nstat.history.History` object or a vector of window
            boundary times (seconds).  An empty list / ``None`` is the
            "unset" sentinel for MATLAB parity.
        ensCovHist : History or array_like, optional
            Ensemble-history specification (history terms aggregated
            across neighbour neurons).
        ensCovMask : array_like, optional
            Binary mask selecting which neighbours contribute ensemble
            history.

        Notes
        -----
        The constructor enforces consistent time bounds (in **seconds**)
        and sample rate (in **Hz**) across the spike collection and
        covariate collection.  Resampling is performed automatically when
        the two disagree.

        Raises
        ------
        ValueError
            If *spike_collection* or *covariate_collection* is missing
            or of the wrong type.

        See Also
        --------
        SpikeTrainCollection, CovariateCollection, Analysis
        """
        self.nspikeColl = spike_collection if spike_collection is not None else spikeColl
        self.covarColl = covariate_collection if covariate_collection is not None else covarColl
        if not isinstance(self.nspikeColl, SpikeTrainCollection):
            raise ValueError("nstColl is a required argument")
        if not isinstance(self.covarColl, CovariateCollection):
            raise ValueError("CovColl is a required argument")

        self.ev: Events | None = None
        # Both ``history`` and ``ensCovHist`` accept either a History object
        # or a sequence of window-time floats; empty list is the "unset"
        # sentinel for MATLAB parity.  ``_is_empty_config_value`` handles
        # both None and [] downstream.
        self.history: Any = []
        self.ensCovHist: Any = []
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
        """The trial's spike-train collection."""
        return self.nspikeColl

    @property
    def covariate_collection(self) -> CovariateCollection:
        """The trial's covariate collection."""
        return self.covarColl

    @property
    def spikeColl(self) -> SpikeTrainCollection:
        """Alias for :attr:`spike_collection` (Matlab compat)."""
        return self.nspikeColl

    def setTrialEvents(self, event: Events | None) -> None:
        """Attach an :class:`Events` object (or ``None`` to clear)."""
        self.ev = event if isinstance(event, Events) else None

    def getEvents(self) -> Events | None:
        """Return the attached Events, or ``None``."""
        return self.ev

    @property
    def covarColl(self) -> CovariateCollection:
        return self._covarColl

    @covarColl.setter
    def covarColl(self, value: CovariateCollection) -> None:
        self._covarColl = value

    def getTrialPartition(self) -> np.ndarray:
        """Return ``[trainMin, trainMax, valMin, valMax]`` partition times."""
        training = [] if self.trainingWindow is None else list(self.trainingWindow)
        validation = [] if self.validationWindow is None else list(self.validationWindow)
        p = training + validation
        if not p:
            return np.asarray([self.minTime, self.maxTime, self.maxTime, self.maxTime], dtype=float)
        return np.asarray(p, dtype=float)

    def setTrialPartition(self, partitionTimes) -> None:
        """Set training and validation time windows from a 3- or 4-element array."""
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
        """Set trial time bounds to either the ``'training'`` or ``'validation'`` window."""
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
        """Set minimum time across spikes, covariates, and ensemble covariates."""
        if minTime is None:
            minTime = self.findMinTime()
        self.nspikeColl.setMinTime(float(minTime))
        self.covarColl.setMinTime(float(minTime))
        if self.ensCovColl is not None:
            self.ensCovColl.setMinTime(float(minTime))
        self.minTime = float(minTime)

    def setMaxTime(self, maxTime: float | None = None) -> None:
        """Set maximum time across spikes, covariates, and ensemble covariates."""
        if maxTime is None:
            maxTime = self.findMaxTime()
        self.nspikeColl.setMaxTime(float(maxTime))
        self.covarColl.setMaxTime(float(maxTime))
        if self.ensCovColl is not None:
            self.ensCovColl.setMaxTime(float(maxTime))
        self.maxTime = float(maxTime)

    def updateTimePartitions(self) -> None:
        """Clamp training/validation windows to current min/max time."""
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

    def plotRaster(self, handle=None):
        """Plot only the spike raster for this trial.

        Parameters
        ----------
        handle : matplotlib Figure or Axes, optional
            If an ``Axes`` is provided the raster is drawn there.
            If a ``Figure`` is provided a new axes is added.
            If *None* a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if handle is None:
            fig, ax = plt.subplots(figsize=(9.0, 3.0))
        elif isinstance(handle, plt.Axes):
            ax = handle
            fig = ax.figure
        else:
            fig = handle
            fig.clear()
            ax = fig.add_subplot(111)
        self.nspikeColl.plot(handle=ax)
        ax.set_title("Trial Spike Raster")
        fig.tight_layout()
        return fig

    def plotCovariates(self, handle=None):
        """Plot covariates (and events, if set) for this trial.

        Layout adapts to the number of active covariates, following the
        Matlab ``Trial.plotCovariates`` behaviour.

        Parameters
        ----------
        handle : matplotlib Figure, optional
            Figure to draw on.  If *None* a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        numCovars = self.covarColl.nActCovar()
        if handle is None:
            fig = plt.figure(figsize=(9.0, max(4.0, 2.2 * max(numCovars, 1))))
        else:
            fig = handle
            fig.clear()

        if numCovars <= 1:
            ax = fig.add_subplot(111)
            self.covarColl.plot(handle=ax)
            if self.ev is not None and self.ev.eventTimes.size:
                self.ev.plot(handle=ax)
        elif numCovars == 2:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            self.covarColl.plot(handle=[ax1, ax2])
            if self.ev is not None and self.ev.eventTimes.size:
                self.ev.plot(handle=[ax1, ax2])
        else:
            axes = [fig.add_subplot(numCovars, 1, i + 1)
                    for i in range(numCovars)]
            self.covarColl.plot(handle=axes)
            if self.ev is not None and self.ev.eventTimes.size:
                self.ev.plot(handle=axes)

        fig.tight_layout()
        return fig

    def plot(self, *_, handle=None, **__):
        """Plot spike raster, covariates, and events in a multi-panel figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
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

        for cov_index in range(self.covarColl.numCov):
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
        """Resample spikes, covariates, and ensemble covariates to *sampleRate*."""
        self.sampleRate = float(sampleRate)
        self.nspikeColl.resample(sampleRate)
        self.covarColl.resample(sampleRate)
        self.resampleEnsColl()

    def resample(self, sampleRate: float) -> None:
        """Alias for :meth:`setSampleRate`."""
        self.setSampleRate(sampleRate)

    def setEnsCovMask(self, mask=None) -> None:
        """Set the ensemble-covariate neighbour mask (default: all-to-all minus self)."""
        if _is_empty_config_value(mask):
            nSpikes = self.nspikeColl.numSpikeTrains
            mask = np.ones((nSpikes, nSpikes), dtype=int) - np.eye(nSpikes, dtype=int)
        self.ensCovMask = np.asarray(mask, dtype=int)

    def setCovMask(self, mask) -> None:
        """Set the covariate mask; ``'all'`` resets to full visibility."""
        if isinstance(mask, str) and mask == "all":
            self.covarColl.resetMask()
        else:
            self.covarColl.setMask(mask)
        self.covMask = self.covarColl.covMask

    def resetCovMask(self) -> None:
        """Reset the covariate mask to all-visible."""
        self.covarColl.resetMask()
        self.covMask = self.covarColl.covMask

    def setNeuronMask(self, mask) -> None:
        """Set the neuron (spike-train) mask and sync to ``self.neuronMask``."""
        self.nspikeColl.setMask(mask)
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()

    def resetNeuronMask(self) -> None:
        """Reset the neuron mask to all-visible."""
        self.nspikeColl.resetMask()
        self.neuronMask = np.asarray(self.nspikeColl.neuronMask, dtype=int).copy()

    def setNeighbors(self, *args) -> None:
        """Set the neighbour structure for ensemble-history covariates."""
        self.nspikeColl.setNeighbors(*args)

    def setHistory(self, hist) -> None:
        """Set the spike-history configuration.

        Parameters
        ----------
        hist : History, array-like, or list[History]
            A ``History`` object, an array of window-edge times (seconds), or
            a list of ``History`` objects for per-neuron history orders.
        """
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
        """Clear the spike-history configuration."""
        self.history = []

    def setEnsCovHist(self, hist=None) -> None:
        """Set the ensemble-covariate history and rebuild the ensemble collection.

        Parameters
        ----------
        hist : History or array-like, optional
            A ``History`` object or window-edge array.  Passing ``None``
            clears the ensemble history and removes the ``ensCovColl``.
        """
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
        """Return ``True`` if any neuron is currently masked out."""
        return self.nspikeColl.isNeuronMaskSet()

    def isCovMaskSet(self) -> bool:
        """Return ``True`` if any covariate dimension is currently masked out."""
        return self.covarColl.isCovMaskSet()

    def isMaskSet(self) -> bool:
        """Return ``True`` if either the neuron or covariate mask is active."""
        return self.isNeuronMaskSet() or self.isCovMaskSet()

    def isHistSet(self) -> bool:
        """Return ``True`` if a spike-history configuration has been set."""
        if self.history in (None, []):
            return False
        from .history import History

        if isinstance(self.history, History):
            return True
        return isinstance(self.history, list) and bool(self.history) and all(isinstance(item, History) for item in self.history)

    def isEnsCovHistSet(self) -> bool:
        """Return ``True`` if an ensemble-covariate history has been set."""
        from .history import History

        return isinstance(self.ensCovHist, History)

    def getNumHist(self) -> int | list[int]:
        """Return the number of history coefficients.

        If a single ``History`` object is set, returns the number of
        history window coefficients (``len(windowTimes) - 1``).
        If a list of ``History`` objects is set, returns a list with
        the count for each.  Returns ``0`` when no history is set.

        Matches Matlab ``Trial.getNumHist()``.
        """
        from .history import History

        if not self.isHistSet():
            return 0
        if isinstance(self.history, History):
            wt = np.asarray(self.history.windowTimes, dtype=float).ravel()
            return max(int(wt.size - 1), 0)
        if isinstance(self.history, list):
            counts: list[int] = []
            for h in self.history:
                if isinstance(h, History):
                    wt = np.asarray(h.windowTimes, dtype=float).ravel()
                    counts.append(max(int(wt.size - 1), 0))
                else:
                    counts.append(0)
            return counts
        return 0

    def addCov(self, cov: Covariate) -> None:
        """Add a covariate and enforce consistent sample rate / time bounds."""
        self.covarColl.addToColl(cov)
        self.covMask = self.covarColl.covMask
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.makeConsistentTime()

    def removeCov(self, identifier: int | str) -> None:
        """Remove a covariate by 0-based index or name."""
        self.covarColl.removeCovariate(identifier)
        self.covMask = self.covarColl.covMask
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.makeConsistentTime()

    def getSpikeVector(self, *args, neuron_index: int = 1) -> np.ndarray:
        """Return the spike data as a column matrix.

        Parameters
        ----------
        *args
            When empty, returns all neurons via ``dataToMatrix()``.  An int
            selects a single neuron (0-based).  A sequence of bin edges
            returns binned counts for the neuron given by *neuron_index*.
        neuron_index : int, default 1
            Neuron to bin when *args* provides bin edges (0-based).
        """
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
        """Return ``(time, data, names)`` for the covariate collection."""
        return self.covarColl.matrixWithTime("standard", selected_covariates)

    def getDesignMatrix(self, neuronNum: int, dataSelector=None) -> np.ndarray:
        """Build the full design matrix for neuron *neuronNum* (0-based).

        Horizontally concatenates covariates, spike-history columns, and
        ensemble-history columns — the complete regressor matrix used by
        the GLM fitter.
        """
        X = self.covarColl.dataToMatrix("standard", dataSelector)
        if self.isHistSet():
            H = self.getHistMatrices(neuronNum)
            if X.size == 0:
                X = H
            else:
                # Align row counts — covariates and history may differ by
                # one sample due to boundary effects in time-grid construction.
                n = min(X.shape[0], H.shape[0])
                X = np.column_stack([X[:n, :], H[:n, :]])
        if self.isEnsCovHistSet():
            E = self.getEnsCovMatrix(neuronNum)
            if X.size == 0:
                X = E
            else:
                n = min(X.shape[0], E.shape[0])
                X = np.column_stack([X[:n, :], E[:n, :]])
        return X

    def getHistForNeurons(self, neuronIndex) -> CovariateCollection:
        """Compute the spike-history covariates for one neuron.

        Parameters
        ----------
        neuronIndex : int
            0-based neuron index whose spike train supplies the history.

        Returns
        -------
        CovariateCollection
            Collection of history-basis covariates aligned to the trial
            time grid.
        """
        if not self.isHistSet():
            raise ValueError("Set Trial history and retry")
        nst = self.nspikeColl.getNST(neuronIndex)
        target_time = np.asarray(self.covarColl.getCov(0).time, dtype=float).reshape(-1) if self.covarColl.numCov else None
        if isinstance(self.history, list):
            histCovColl: CovariateCollection | None = None
            for i, hist in enumerate(self.history, start=1):
                temp = hist.computeHistory(nst, i, time_grid=target_time)
                histCovColl = temp if histCovColl is None else CovariateCollection([*histCovColl.covArray, *temp.covArray])
            assert histCovColl is not None
            return histCovColl
        return self.history.computeHistory(nst, time_grid=target_time)

    def getHistMatrices(self, neuronIndex: int) -> np.ndarray:
        """Return the spike-history columns as a 2-D array for *neuronIndex* (0-based)."""
        if not self.isHistSet():
            time = self.nspikeColl.getNST(neuronIndex).getSigRep().time
            return np.zeros((time.size, 0), dtype=float)
        histCovColl = self.getHistForNeurons(neuronIndex)
        return histCovColl.dataToMatrix("standard")

    def getEnsembleNeuronCovariates(self, *args):
        """Delegate to ``SpikeTrainCollection.getEnsembleNeuronCovariates``."""
        return self.nspikeColl.getEnsembleNeuronCovariates(*args)

    def getEnsCovMatrix(self, neuronNum: int, includedNeurons=None) -> np.ndarray:
        """Return the ensemble-covariate design-matrix columns for *neuronNum*.

        Uses ``ensCovMask`` to exclude self-history and applies neighbour
        filtering when *includedNeurons* is not specified.
        """
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return np.zeros((self.nspikeColl.getNST(neuronNum).getSigRep().time.size, 0), dtype=float)
        if includedNeurons is None:
            includedNeurons = np.flatnonzero(self.ensCovMask[:, neuronNum] == 1)
        ensCovCollTemp = CovariateCollection(self.ensCovColl.covArray)
        ensCovCollTemp.covMask = [mask.copy() for mask in self.ensCovColl.covMask]
        ensCovCollTemp.maskAwayAllExcept(includedNeurons)
        return ensCovCollTemp.dataToMatrix("standard")

    def getNeuronIndFromMask(self) -> list[int]:
        """Return 0-based indices of currently unmasked neurons."""
        return self.nspikeColl.getIndFromMask()

    def getNumUniqueNeurons(self) -> int:
        """Return the number of distinct neuron names in the collection."""
        return len(self.nspikeColl.uniqueNeuronNames)

    def getNeuronNames(self) -> list[str]:
        """Return all neuron names (may contain duplicates for repeated trials)."""
        return self.nspikeColl.getNSTnames()

    def getUniqueNeuronNames(self) -> list[str]:
        """Return deduplicated neuron names."""
        return self.nspikeColl.getUniqueNSTnames()

    def getNeuronIndFromName(self, neuronName: str):
        """Return 0-based indices matching *neuronName*, filtered by the neuron mask."""
        tempInd = self.nspikeColl.getNSTIndicesFromName(neuronName)
        currMask = set(self.neuronMask_indices())
        if isinstance(tempInd, list):
            return [idx for idx in tempInd if idx in currMask]
        return [tempInd] if tempInd in currMask else []

    def neuronMask_indices(self) -> list[int]:
        """Return 0-based indices of unmasked neurons (alias for ``getNeuronIndFromMask``)."""
        return self.nspikeColl.getIndFromMask()

    def getNeuronNeighbors(self, neuronNum=None):
        """Return the neighbour list for *neuronNum* (defaults to all unmasked neurons)."""
        if neuronNum is None:
            neuronNum = self.getNeuronIndFromMask()
        return self.nspikeColl.getNeighbors(neuronNum)

    def getCovSelectorFromMask(self):
        """Return the per-covariate selector list derived from the current mask."""
        return self.covarColl.getSelectorFromMasks()

    def getCov(self, identifier):
        """Return a ``Covariate`` by 0-based index or name."""
        return self.covarColl.getCov(identifier)

    def getNeuron(self, identifier):
        """Return an ``nspikeTrain`` by 0-based index or name."""
        return self.nspikeColl.getNST(identifier)

    def getAllCovLabels(self) -> list[str]:
        """Return labels for all covariate dimensions (ignoring mask)."""
        return self.covarColl.getAllCovLabels()

    def getCovLabelsFromMask(self) -> list[str]:
        """Return labels for only the currently unmasked covariate dimensions."""
        return self.covarColl.getCovLabelsFromMask()

    def getHistLabels(self) -> list[str]:
        """Return string labels for all spike-history basis columns."""
        if not self.isHistSet():
            return []
        return self.getHistForNeurons(0).getAllCovLabels()

    def getEnsCovLabels(self) -> list[str]:
        """Return string labels for all ensemble-covariate columns."""
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return []
        return self.ensCovColl.getAllCovLabels()

    def getEnsCovLabelsFromMask(self, neuronNum: int) -> list[str]:
        """Return ensemble-covariate labels for *neuronNum*, filtered by ``ensCovMask``."""
        if not self.isEnsCovHistSet() or self.ensCovColl is None:
            return []
        included = np.flatnonzero(self.ensCovMask[:, neuronNum] == 1)
        ensCovCollTemp = CovariateCollection(self.ensCovColl.covArray)
        ensCovCollTemp.covMask = [mask.copy() for mask in self.ensCovColl.covMask]
        ensCovCollTemp.maskAwayAllExcept(included)
        return ensCovCollTemp.getCovLabelsFromMask()

    def getAllLabels(self) -> list[str]:
        """Return all covariate + history + ensemble labels (no mask filtering).

        Matlab equivalent: ``Trial.getAllLabels``.
        """
        labels = list(self.getAllCovLabels())
        labels.extend(self.getHistLabels())
        labels.extend(self.getEnsCovLabels())
        return labels

    def getLabelsFromMask(self, neuronNum: int) -> list[str]:
        """Return all design-matrix labels for *neuronNum*, respecting masks."""
        labels = list(self.getCovLabelsFromMask())
        labels.extend(self.getHistLabels())
        labels.extend(self.getEnsCovLabelsFromMask(neuronNum))
        return labels

    def flattenCovMask(self) -> np.ndarray:
        """Flatten the per-covariate mask list into a single 1-D int array."""
        return self.covarColl.flattenCovMask()

    def flattenMask(self) -> np.ndarray:
        """Flatten the full mask (covariates + history + ensemble) into 1-D."""
        flat = self.flattenCovMask()
        if self.isHistSet():
            flat = np.concatenate([flat, np.ones(len(self.getHistLabels()), dtype=int)])
        if self.isEnsCovHistSet():
            flat = np.concatenate([flat, np.ones(len(self.getEnsCovLabels()), dtype=int)])
        return flat

    def shiftCovariates(self, *args) -> None:
        """Apply a time shift to covariates and re-synchronize time bounds."""
        self.covarColl.setCovShift(*args)
        self.makeConsistentTime()

    def resetEnsCovMask(self) -> None:
        """Reset the ensemble-covariate mask to the default (all-to-all minus self)."""
        self.setEnsCovMask()

    def resampleEnsColl(self) -> None:
        """Rebuild the ensemble-covariate collection at the current sample rate."""
        if self.ensCovColl is not None and self.ensCovHist not in (None, []):
            self.ensCovColl = self.getEnsembleNeuronCovariates(1, [], self.ensCovHist)
        else:
            self.setEnsCovHist()

    def restoreToOriginal(self) -> None:
        """Reset all collections to their original state and re-synchronize."""
        self.nspikeColl.restoreToOriginal()
        self.covarColl.restoreToOriginal()
        if not self.isSampleRateConsistent():
            self.makeConsistentSampleRate()
        self.resampleEnsColl()
        self.makeConsistentTime()

    # ------------------------------------------------------------------
    # Serialization (Matlab Trial.toStructure / Trial.fromStructure)
    # ------------------------------------------------------------------
    def toStructure(self) -> dict[str, Any]:
        """Serialize a Trial to a plain dict (Matlab ``Trial.toStructure``)."""
        from .history import History

        structure: dict[str, Any] = {}
        structure["nspikeColl"] = self.nspikeColl.toStructure()
        structure["covarColl"] = self.covarColl.toStructure()
        structure["ev"] = self.ev.toStructure() if self.ev is not None else None
        structure["history"] = self.history.toStructure() if isinstance(self.history, History) else None
        structure["ensCovHist"] = self.ensCovHist.toStructure() if isinstance(self.ensCovHist, History) else None
        structure["sampleRate"] = float(self.sampleRate) if np.isfinite(self.sampleRate) else self.sampleRate
        structure["minTime"] = float(self.minTime)
        structure["maxTime"] = float(self.maxTime)
        structure["covMask"] = [np.asarray(m, dtype=int).tolist() for m in self.covMask] if self.covMask is not None else []
        structure["neuronMask"] = np.asarray(self.neuronMask, dtype=int).tolist()
        structure["trainingWindow"] = np.asarray(self.trainingWindow, dtype=float).tolist() if self.trainingWindow is not None else []
        structure["validationWindow"] = np.asarray(self.validationWindow, dtype=float).tolist() if self.validationWindow is not None else []
        return structure

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "Trial":
        """Reconstruct a Trial from a dict produced by :meth:`toStructure` (Matlab ``Trial.fromStructure``)."""
        from .events import Events
        from .history import History

        nspikeColl = SpikeTrainCollection.fromStructure(structure["nspikeColl"])
        covarColl = CovariateCollection.fromStructure(structure["covarColl"])
        ev = Events.fromStructure(structure.get("ev"))
        h = History.fromStructure(structure.get("history"))
        ensHist = History.fromStructure(structure.get("ensCovHist"))
        trial = Trial(nspikeColl, covarColl, ev, h, ensHist)

        if "minTime" in structure:
            trial.setMinTime(float(structure["minTime"]))
        if "maxTime" in structure:
            trial.setMaxTime(float(structure["maxTime"]))

        trainingW = structure.get("trainingWindow", [])
        validationW = structure.get("validationWindow", [])
        if trainingW and validationW:
            partition = list(trainingW) + list(validationW)
            trial.setTrialPartition(partition)

        return trial

    def makeConsistentSampleRate(self) -> None:
        """Resample all collections to the maximum sample rate found."""
        self.resample(self.findMaxSampleRate())

    def makeConsistentTime(self) -> None:
        """Set all collections to the union of min/max time across sub-collections."""
        self.setMinTime(self.findMinTime())
        self.setMaxTime(self.findMaxTime())

    def isSampleRateConsistent(self) -> bool:
        """Return ``True`` if spike and covariate collections share the same sample rate."""
        if self.nspikeColl.numSpikeTrains == 0 or self.covarColl.numCov == 0:
            return True
        target = round(float(self.findMaxSampleRate()), 3)
        values = [round(float(self.nspikeColl.sampleRate), 3), round(float(self.covarColl.sampleRate), 3)]
        return all(value == target for value in values)

    def findMaxSampleRate(self) -> float:
        """Return the maximum sample rate across spike and covariate collections."""
        values = [value for value in [self.nspikeColl.findMaxSampleRate(), self.covarColl.findMaxSampleRate()] if np.isfinite(value)]
        return float(max(values)) if values else float("nan")

    def findMinSampleRate(self) -> float:
        """Return the minimum sample rate across spike collection, covariate collection, and trial.

        Matches Matlab ``Trial.findMinSampleRate()``.
        """
        candidates: list[float] = []
        if hasattr(self, "sampleRate") and np.isfinite(self.sampleRate):
            candidates.append(float(self.sampleRate))
        try:
            sr = self.nspikeColl.sampleRate
            if np.isfinite(sr):
                candidates.append(float(sr))
        except Exception:
            pass
        try:
            sr = self.covarColl.sampleRate
            if np.isfinite(sr):
                candidates.append(float(sr))
        except Exception:
            pass
        return float(min(candidates)) if candidates else float("nan")

    def findMinTime(self) -> float:
        """Return the earliest start time across sub-collections."""
        return float(min(self.nspikeColl.minTime, self.covarColl.minTime))

    def findMaxTime(self) -> float:
        """Return the latest end time across sub-collections."""
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
