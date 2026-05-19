"""Canonical implementation of ``nspikeTrain``.

Previously a section of :mod:`nstat.core` (~640 lines); split out for
readability.  ``nstat.core`` continues to re-export ``nspikeTrain`` for
back-compat — all existing import paths
(``from nstat.core import nspikeTrain``, ``from nstat.nspikeTrain import
nspikeTrain``, ``from nstat import nspikeTrain``) still work.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import SignalObj, _matlab_mode_1d, _roundn


class nspikeTrain:
    """Point-process (spike train) object (Matlab ``nspikeTrain``).

    Stores an array of event times (spikes) and converts them on demand
    into a binned ``SignalObj`` signal representation (``sigRep``).  Burst
    statistics, ISI analysis, and raster plotting are built in.

    Parameters
    ----------
    spikeTimes : array_like
        Spike times in seconds.
    name : str, optional
        Neuron / channel label.
    binwidth : float, optional
        Bin width in seconds for the signal representation (default 1 ms).
    minTime, maxTime : float, optional
        Observation window.  Defaults to ``min/max(spikeTimes)``.
    xlabelval, xunits, yunits : str, optional
        Axis label and unit strings.
    dataLabels : str or sequence of str, optional
        Label(s) for the spike-train dimension.
    makePlots : int, optional
        ``0`` — compute statistics silently (default);
        ``1`` — compute and plot;
        ``< 0`` — skip statistics entirely (fast construction).

    See Also
    --------
    SignalObj : Continuous time-series container returned by ``getSigRep``.
    SpikeTrainCollection : Multi-neuron collection.
    """

    def __init__(
        self,
        spikeTimes,
        name: str = "",
        sampleRate: float = 1000.0,
        minTime: float | None = None,
        maxTime: float | None = None,
        xlabelval: str = "time",
        xunits: str = "s",
        yunits: str = "",
        dataLabels: str | Sequence[str] | None = "",
        makePlots: int = 0,
    ) -> None:
        if spikeTimes is None:
            raise ValueError("nspikeTrain requires a spikeTimes array as input to create an object")
        spikes = np.asarray(spikeTimes, dtype=float).reshape(-1)
        self.spikeTimes = np.sort(spikes)
        self.originalSpikeTimes = self.spikeTimes.copy()
        self.name = str(name)
        self.sampleRate = float(sampleRate)
        self.originalSampleRate = float(self.sampleRate)
        if minTime is None:
            minTime = float(np.min(self.spikeTimes)) if self.spikeTimes.size else 0.0
        if maxTime is None:
            maxTime = float(np.max(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.minTime = float(minTime)
        self.maxTime = float(maxTime)
        self.originalMinTime = float(self.minTime)
        self.originalMaxTime = float(self.maxTime)
        self.xlabelval = str(xlabelval)
        self.xunits = str(xunits)
        self.yunits = str(yunits)
        self.dataLabels = dataLabels if dataLabels is not None else ""
        self.sigRep: SignalObj | None = None
        self.isSigRepBin: bool | None = None
        self._sigrep_cache_key: tuple[float, float, float] | None = None
        self.MER = None
        if makePlots >= 0:
            self.computeStatistics(makePlots)
        else:
            self.avgFiringRate = None
            self.B = None
            self.An = None
            self.burstTimes = None
            self.burstRate = None
            self.burstDuration = None
            self.burstSig = None
            self.burstIndex = None
            self.numBursts = None
            self.numSpikesPerBurst = None
            self.avgSpikesPerBurst = None
            self.stdSpikesPerBurst = None
            self.Lstatistic = None

    @property
    def times(self) -> np.ndarray:
        """Alias for ``spikeTimes``."""
        return self.spikeTimes

    @property
    def n_spikes(self) -> int:
        """Number of spikes in the train."""
        return int(self.spikeTimes.size)

    @property
    def duration(self) -> float:
        """Observation window duration ``maxTime - minTime`` in seconds."""
        return float(self.maxTime - self.minTime)

    @property
    def firing_rate_hz(self) -> float:
        """Average firing rate (spikes / duration) in Hz."""
        if self.duration <= 0:
            return 0.0
        return float(self.n_spikes / self.duration)

    def setMER(self, MERSig: SignalObj) -> None:
        """Attach a micro-electrode recording signal to this spike train."""
        if isinstance(MERSig, SignalObj):
            self.MER = MERSig

    def setName(self, name: str) -> None:
        """Set the neuron / channel name."""
        self.name = str(name)

    def computeStatistics(self, makePlots: int = 0) -> None:
        """Compute ISI, burst, and regularity statistics (Matlab ``computeStatistics``)."""
        self.avgFiringRate = self.firing_rate_hz
        isi = self.getISIs()
        # Filter spike times to [minTime, maxTime] so burst statistics
        # remain valid after setMinTime / setMaxTime (Matlab parity).
        spike_times = self.getSpikeTimes(self.minTime, self.maxTime)
        mode_isi = _matlab_mode_1d(isi)
        self.burstIndex = float(1.0 / mode_isi / self.avgFiringRate) if np.isfinite(mode_isi) and self.avgFiringRate > 0 else np.nan
        self.B = np.nan
        self.An = np.nan
        self.burstTimes = np.array([], dtype=float)
        self.burstRate = np.array([], dtype=float)
        self.burstDuration = np.array([], dtype=float)
        self.burstSig = None
        self.numBursts = 0
        self.numSpikesPerBurst = np.array([], dtype=float)
        self.avgSpikesPerBurst = np.nan
        self.stdSpikesPerBurst = np.nan
        self.Lstatistic = np.nan

        if isi.size:
            sigma = float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0
            mu = float(np.mean(isi))
            if np.isfinite(mu) and mu > 0:
                r = sigma / mu
                self.B = float((r - 1.0) / (r + 1.0))
                n = float(spike_times.size)
                self.An = float((np.sqrt(n + 2.0) * r - np.sqrt(n)) / (((np.sqrt(n + 2.0) - 2.0) * r) + np.sqrt(n)))

                ln = isi[isi < mu]
                ml = float(np.mean(ln)) if ln.size else np.nan
                if np.isfinite(ml):
                    burst_isi = (isi < ml).astype(float)
                    shifted = np.concatenate([burst_isi[1:], [0.0]]) if burst_isi.size else np.array([], dtype=float)
                    y = (burst_isi + shifted) > 1.0
                    diff_sig = np.concatenate([[0.0], np.diff(y.astype(float))]) if y.size else np.array([], dtype=float)
                    burst_start = np.flatnonzero(diff_sig == 1.0)
                    burst_end = np.flatnonzero(diff_sig == -1.0) + 1
                    if burst_start.size == 0:
                        burst_end = np.array([], dtype=int)
                    if burst_end.size > burst_start.size and burst_end.size:
                        first = np.flatnonzero(y[: burst_end[0]] == 1)
                        if first.size:
                            burst_start = np.concatenate([[int(first[0])], burst_start])
                    if burst_start.size > burst_end.size and burst_start.size:
                        last = np.flatnonzero(y[burst_start[-1] :] == 1)
                        if last.size:
                            burst_end = np.concatenate([burst_end, [int(last[-1])]])
                    if burst_start.size and burst_end.size:
                        burst_data = np.zeros(spike_times.size, dtype=float)
                        for start, end in zip(burst_start, burst_end, strict=False):
                            burst_data[int(start) : int(end) + 1] = 1.0
                        self.burstDuration = spike_times[burst_end] - spike_times[burst_start]
                        self.burstSig = SignalObj(spike_times, burst_data, "Burst Signal")
                        self.burstTimes = spike_times[burst_start]
                        self.numBursts = int(burst_start.size)
                        duration = self.maxTime - self.minTime
                        self.burstRate = float(self.numBursts / duration) if duration > 0 else np.nan
                        # MATLAB convention: ``numSpikesPerBurst`` stores
                        # the inter-burst-interval count (== ``end - start + 1``),
                        # while ``avgSpikesPerBurst``/``stdSpikesPerBurst`` add
                        # an extra +1 (counting the starting spike).  This is
                        # an idiosyncrasy of the upstream toolbox; we replicate
                        # it for gold-fixture parity (see
                        # ``tests/parity/fixtures/matlab_gold/nspiketrain_exactness.mat``).
                        self.numSpikesPerBurst = (burst_end - burst_start + 1).astype(float)
                        self.avgSpikesPerBurst = float(np.mean(self.numSpikesPerBurst + 1.0))
                        if self.numSpikesPerBurst.size > 1:
                            self.stdSpikesPerBurst = float(np.std(self.numSpikesPerBurst + 1.0, ddof=1))
                        elif self.numSpikesPerBurst.size == 1:
                            self.stdSpikesPerBurst = 0.0

        self.Lstatistic = self.getLStatistic()
        if makePlots == 1:
            self.plot()

    def getLStatistic(self) -> float:
        """Return the L-statistic (number of unique bin counts in ``sigRep``)."""
        isi = self.getISIs()
        if isi.size == 0:
            return np.nan
        mean_isi = float(np.mean(isi))
        if not np.isfinite(mean_isi) or mean_isi <= 0:
            return np.nan
        duration = self.maxTime - self.minTime
        if not np.isfinite(duration) or duration <= 0:
            return np.nan
        approx = self.getSigRep(mean_isi)
        return float(np.unique(approx.data[:, 0]).size)

    def _cache_key(self, binwidth: float, minTime: float, maxTime: float) -> tuple[float, float, float]:
        return (round(float(binwidth), 12), round(float(minTime), 12), round(float(maxTime), 12))

    def _build_sigrep(self, binwidth: float, minTime: float, maxTime: float) -> SignalObj:
        if binwidth <= 0:
            raise ValueError("binwidth must be > 0")
        if maxTime < minTime:
            raise ValueError("maxTime must be >= minTime")

        max_bins = int(1e6)
        precision = max(0, int(2 * np.ceil(np.log10(1.0 / binwidth))))
        bw = float(_roundn([binwidth], precision)[0])
        duration = float(maxTime - minTime)
        if np.isfinite(duration) and duration > 0 and np.isfinite(bw) and bw > 0:
            est_bins = duration / bw + 1.0
            if not np.isfinite(est_bins) or est_bins > max_bins:
                bw = duration / float(max_bins - 1)
                precision = max(0, int(2 * np.ceil(np.log10(1.0 / bw))))
                bw = float(_roundn([bw], precision)[0])
        if not np.isfinite(bw) or bw <= 0:
            bw = duration / float(max_bins - 1) if np.isfinite(duration) and duration > 0 else 1.0 / max(self.sampleRate, 1.0)

        numBins = int(np.floor(duration / bw + 1.0)) if np.isfinite(duration) else 2
        if numBins < 2:
            numBins = 2
        if numBins > max_bins:
            numBins = max_bins
        timeVec = np.linspace(minTime, maxTime, numBins, dtype=float)
        if timeVec.size > 1:
            bw = float(np.mean(np.diff(timeVec)))
        windowTimes = np.concatenate([[minTime - bw / 2.0], timeVec + bw / 2.0])

        spikeTimes = _roundn(self.spikeTimes, precision)
        rounded_windows = _roundn(windowTimes, precision + 1)
        counts = np.zeros(timeVec.size, dtype=float)
        split_index = int(np.floor(rounded_windows.size / 2.0))
        for idx in range(timeVec.size):
            left = rounded_windows[idx]
            right = rounded_windows[idx + 1]
            if idx == rounded_windows.size - 2:
                temp = spikeTimes[spikeTimes >= left]
                counts[idx] = float(np.sum(temp <= right))
            elif idx + 1 > split_index:
                temp = spikeTimes[spikeTimes >= left]
                counts[idx] = float(np.sum(temp < right))
            else:
                temp = spikeTimes[spikeTimes < right]
                counts[idx] = float(np.sum(temp >= left))

        label = self.dataLabels if isinstance(self.dataLabels, str) else ""
        sig = SignalObj(timeVec, counts.astype(float), self.name, self.xlabelval, self.xunits, self.yunits, label)
        self.isSigRepBin = bool(np.all(counts <= 1))
        return sig

    def setSigRep(self, binwidth: float | None = None, minTime: float | None = None, maxTime: float | None = None) -> SignalObj:
        """Build the binned signal representation and store it in-place."""
        sig = self.getSigRep(binwidth, minTime, maxTime)
        self.sigRep = sig.copySignal()
        self.sampleRate = float(sig.sampleRate)
        self.isSigRepBin = bool(np.max(np.asarray(sig.data, dtype=float)) <= 1.0)
        # Keep the freshly-built cached representation alive instead of
        # clearing it through the public min/max setters.
        self.minTime = float(sig.minTime)
        self.maxTime = float(sig.maxTime)
        self.computeStatistics(0)
        return self.sigRep

    def clearSigRep(self) -> None:
        """Invalidate the cached signal representation."""
        self.sigRep = None
        self._sigrep_cache_key = None
        self.isSigRepBin = None

    def setMinTime(self, minTime: float) -> None:
        """Set the observation-window start and recompute statistics."""
        self.minTime = float(minTime)
        self.clearSigRep()
        self.computeStatistics(0)

    def setMaxTime(self, maxTime: float) -> None:
        """Set the observation-window end and recompute statistics."""
        self.maxTime = float(maxTime)
        self.clearSigRep()
        self.computeStatistics(0)

    def resample(self, sampleRate: float) -> "nspikeTrain":
        """Rebuild the signal representation at *sampleRate* Hz."""
        self.setSigRep(1.0 / float(sampleRate), self.minTime, self.maxTime)
        self.sampleRate = float(sampleRate)
        return self

    def getSpikeTimes(self, minTime: float | None = None, maxTime: float | None = None) -> np.ndarray:
        """Return spike times within ``[minTime, maxTime]``."""
        start = self.minTime if minTime is None else float(minTime)
        stop = self.maxTime if maxTime is None else float(maxTime)
        spikes = self.spikeTimes[(self.spikeTimes >= start) & (self.spikeTimes <= stop)]
        return spikes.copy()

    def getISIs(self, minTime: float | None = None, maxTime: float | None = None) -> np.ndarray:
        """Return inter-spike intervals within the given time window."""
        spikes = self.getSpikeTimes(minTime, maxTime)
        if spikes.size < 2:
            return np.array([], dtype=float)
        return np.diff(spikes)

    def getMinISI(self, minTime: float | None = None, maxTime: float | None = None) -> float:
        """Return the minimum ISI (refractory period estimate)."""
        isi = self.getISIs(minTime, maxTime)
        if isi.size == 0:
            return float("nan")
        return float(np.min(isi))

    def getSigRep(
        self,
        binwidth: float | None = None,
        minTime: float | None = None,
        maxTime: float | None = None,
    ) -> SignalObj:
        """Return the binned signal representation, using cache when possible.

        The result is a ``SignalObj`` of spike counts on a regular grid
        with bin width *binwidth* (default ``1/sampleRate``).
        """
        bw = (1.0 / self.sampleRate) if binwidth is None else float(binwidth)
        start = self.minTime if minTime is None else float(minTime)
        stop = self.maxTime if maxTime is None else float(maxTime)
        key = self._cache_key(bw, start, stop)
        if self.sigRep is not None and self._sigrep_cache_key == key:
            return self.sigRep.copySignal()
        sig = self._build_sigrep(bw, start, stop)
        self.sigRep = sig.copySignal()
        self._sigrep_cache_key = key
        return sig

    def getMaxBinSizeBinary(self) -> float:
        """Return the largest bin width that keeps the ``sigRep`` binary."""
        isi = self.getISIs()
        if isi.size == 0:
            return np.inf
        return float(np.min(isi))

    def isSigRepBinary(self) -> bool:
        """Return ``True`` if every bin in the default ``sigRep`` has <= 1 spike."""
        default_key = self._cache_key(1.0 / float(self.sampleRate), float(self.minTime), float(self.maxTime))
        if self._sigrep_cache_key != default_key or self.isSigRepBin is None:
            self.getSigRep(1.0 / float(self.sampleRate), float(self.minTime), float(self.maxTime))
        return bool(self.isSigRepBin)

    def computeRate(self) -> SignalObj:
        """Return firing rate ``sigRep * sampleRate`` in spikes/sec."""
        sig = self.getSigRep()
        if self.sampleRate <= 0:
            return sig
        rate = np.asarray(sig.data[:, 0], dtype=float) * float(self.sampleRate)
        return SignalObj(sig.time, rate, self.name, sig.xlabelval, sig.xunits, "spikes/sec", sig.dataLabels)

    def restoreToOriginal(self) -> None:
        """Reset spike times and time bounds to original values."""
        self.spikeTimes = self.originalSpikeTimes.copy()
        self.minTime = float(np.min(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.maxTime = float(np.max(self.spikeTimes)) if self.spikeTimes.size else 0.0
        self.clearSigRep()

    def partitionNST(
        self,
        windowTimes: Sequence[float],
        normalizeTime: int | bool | None = None,
        lbound: float | None = None,
        ubound: float | None = None,
    ):
        """Partition into per-trial spike trains (Matlab ``partitionNST``).

        Parameters
        ----------
        windowTimes : sequence of float
            Edge times defining trial boundaries (N edges -> N-1 trials).
        normalizeTime : bool, optional
            If ``True``, rescale each trial's spikes to [0, 1].
        lbound, ubound : float, optional
            Accept only windows whose duration falls in ``[lbound, ubound]``.

        Returns
        -------
        nstColl
        """
        from .nstColl import nstColl

        windows = np.asarray(windowTimes, dtype=float).reshape(-1)
        if windows.size <= 1:
            return nstColl([])
        if ubound is None:
            ubound = lbound

        normalize = bool(normalizeTime) if normalizeTime is not None else False
        partitions: list[nspikeTrain] = []
        for index, (window_start, window_stop) in enumerate(zip(windows[:-1], windows[1:]), start=1):
            window_start = round(float(window_start) * self.sampleRate) / self.sampleRate
            window_stop = round(float(window_stop) * self.sampleRate) / self.sampleRate
            duration = float(window_stop - window_start)
            if lbound is not None and ubound is not None and not (float(lbound) <= abs(duration) <= float(ubound)):
                continue
            if index == windows.size - 1:
                subset = self.spikeTimes[(self.spikeTimes >= window_start) & (self.spikeTimes <= window_stop)]
            else:
                subset = self.spikeTimes[(self.spikeTimes >= window_start) & (self.spikeTimes < window_stop)]
            subset = subset - float(window_start)
            if normalize and duration != 0:
                subset = subset / duration
            partitions.append(nspikeTrain(subset, self.name, makePlots=-1))

        coll = nstColl(partitions)
        if normalize:
            coll.setMinTime(0.0)
            coll.setMaxTime(1.0)
        return coll

    def getFieldVal(self, fieldName: str):
        """Return the value of attribute *fieldName* (Matlab ``getFieldVal``)."""
        return getattr(self, fieldName, [])

    def plotISISpectrumFunction(self):
        """Plot ISI vs. time (Matlab ``plotISISpectrumFunction``)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
        isi = self.getISIs()
        if isi.size:
            (line,) = ax.plot(self.spikeTimes[1:], isi, ".")
        else:
            (line,) = ax.plot([], [], ".")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ISI [s]")
        return line

    def plotJointISIHistogram(self):
        """Joint ISI scatter plot: ISI(t) vs ISI(t+1) on log-log axes."""
        import matplotlib.pyplot as plt

        ax = plt.subplots(1, 1, figsize=(4.5, 4.0))[1]
        isi = self.getISIs()
        if isi.size >= 2:
            xvals = np.asarray(isi[:-1], dtype=float).reshape(-1)
            yvals = np.asarray(isi[1:], dtype=float).reshape(-1)
            ax.loglog(xvals, yvals, ".")
            mean_isi = float(np.mean(isi))
            ln = isi[isi < mean_isi]
            ml = float(np.mean(ln)) if ln.size else np.nan
            if np.isfinite(ml) and ml > 0:
                ymin = float(np.min(yvals))
                ymax = float(np.max(yvals))
                xmin = float(np.min(xvals))
                xmax = float(np.max(xvals))
                ax.loglog([ml, ml], [ymin, ymax], "k--")
                ax.loglog([xmin, xmax], [ml, ml], "k--")
        ax.set_xlabel("ISI(t) [s]")
        ax.set_ylabel("ISI(t+1) [s]")
        return ax

    def plotISIHistogram(self, minTime: float | None = None, maxTime: float | None = None, numBins: int | None = None, handle=None):
        """Plot ISI histogram (Matlab ``plotISIHistogram``).

        Parameters
        ----------
        minTime, maxTime : float, optional
            Time window for ISIs.  Defaults to the spike train bounds.
        numBins : int, optional
            Number of histogram bins.  When *None* the bin width defaults to
            1 ms (Matlab default behaviour).
        handle : matplotlib Axes, optional
            Axes to plot into.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if handle is None else handle
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        isi = self.getISIs(minTime, maxTime)
        counts = np.array([], dtype=float)
        bins = np.array([], dtype=float)
        if isi.size:
            isi_max = float(np.max(isi))
            if numBins is not None and int(numBins) > 0:
                # Linearly-spaced bins when numBins is specified (Matlab parity).
                n = int(numBins)
                bin_width = max(isi_max / n, 1e-12)
                bins = np.linspace(0.0, isi_max, n + 1, dtype=float)
            else:
                # Default: 1 ms bin width.
                bin_width = 0.001
                bins = np.arange(0.0, isi_max + bin_width, bin_width, dtype=float)
            if bins.size < 2:
                bins = np.array([0.0, bin_width], dtype=float)
            idx = np.searchsorted(bins, isi, side="right") - 1
            idx = np.where(
                np.isclose(isi, bins[-1], rtol=0.0, atol=max(1e-12, bin_width * 1e-9)),
                bins.size - 1,
                idx,
            )
            idx = np.clip(idx, 0, bins.size - 1)
            counts = np.bincount(idx, minlength=bins.size).astype(float)
            centers = bins[:counts.size] if bins.size > counts.size else bins
            ax.bar(
                centers,
                counts[:centers.size],
                width=bin_width,
                align="edge",
                edgecolor="none",
                linewidth=2.0,
                color=(0.831372559070587, 0.815686285495758, 0.7843137383461),
            )
        ax.set_xlabel("ISI [sec]")
        ax.set_ylabel("Spike Counts")
        ax.autoscale(enable=True, axis="x", tight=True)
        return counts

    def plotProbPlot(self, minTime: float | None = None, maxTime: float | None = None, handle=None):
        """Exponential probability plot of ISIs (Matlab ``plotProbPlot``)."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if handle is None else handle
        if maxTime is None:
            maxTime = self.maxTime
        if minTime is None:
            minTime = self.minTime
        isi = self.getISIs(minTime, maxTime)
        ax.clear()
        if isi.size:
            sorted_isi = np.sort(np.asarray(isi, dtype=float).reshape(-1))
            n = sorted_isi.size
            p = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
            exp_quantiles = -np.log(1.0 - p)
            ax.plot(sorted_isi, exp_quantiles, linestyle="none", marker=".")
        return ax

    def plotExponentialFit(self, minTime: float | None = None, maxTime: float | None = None, numBins: int | None = None, handle=None):
        """ISI histogram + exponential prob-plot side by side."""
        import matplotlib.pyplot as plt

        fig = handle if handle is not None else plt.figure(figsize=(10.0, 4.0))
        fig.clear()
        axes = fig.subplots(1, 2)
        self.plotISIHistogram(minTime, maxTime, numBins, axes[0])
        self.plotProbPlot(minTime, maxTime, axes[1])
        fig.tight_layout()
        return fig

    def plot(self, dHeight: float = 1.0, yOffset: float = 0.5, currentHandle=None, handle=None):
        """Raster plot: vertical tick per spike (Matlab ``plot``).

        Parameters
        ----------
        dHeight : float
            Tick height (default 1.0).
        yOffset : float
            Vertical centre of ticks (default 0.5).
        currentHandle, handle : matplotlib Axes, optional
            Axes to draw into.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if (currentHandle is None and handle is None) else (currentHandle or handle)
        lines = []
        for spike_time in self.spikeTimes:
            (line,) = ax.plot(
                [spike_time, spike_time],
                [yOffset - dHeight / 2.0, yOffset + dHeight / 2.0],
                "k",
            )
            lines.append(line)
        if currentHandle is None and handle is None:
            xunits = f" [{self.xunits}]" if self.xunits else ""
            yunits = f" [{self.yunits}]" if self.yunits else ""
            ax.set_xlabel(f"{self.xlabelval}{xunits}")
            ax.set_ylabel(f"{self.name}{yunits}")
            if self.minTime != self.maxTime:
                ax.set_xlim(self.minTime, self.maxTime)
        return lines

    def nstCopy(self) -> "nspikeTrain":
        """Return a deep copy (Matlab ``nstCopy``).

        Matlab's ``nstCopy`` builds the copy's sigRep and calls
        ``computeStatistics(0)`` so the copy has valid burst parameters.
        """
        return nspikeTrain(
            self.spikeTimes.copy(),
            self.name,
            self.sampleRate if self.sampleRate > 0 else 1000.0,
            self.minTime,
            self.maxTime,
            self.xlabelval,
            self.xunits,
            self.yunits,
            self.dataLabels,
            0,
        )

    def to_binned_counts(self, bin_edges: Sequence[float]) -> np.ndarray:
        """Histogram spike times into *bin_edges* and return count vector."""
        edges = np.asarray(bin_edges, dtype=float).reshape(-1)
        counts, _ = np.histogram(self.spikeTimes, bins=edges)
        return counts.astype(float)

    def toStructure(self) -> dict[str, Any]:
        """Serialize to a plain dict (Matlab ``toStructure``)."""
        return {
            "spikeTimes": self.spikeTimes.tolist(),
            "name": self.name,
            "sampleRate": self.sampleRate,
            "minTime": self.minTime,
            "maxTime": self.maxTime,
            "xlabelval": self.xlabelval,
            "xunits": self.xunits,
            "yunits": self.yunits,
            "dataLabels": self.dataLabels,
        }

    @staticmethod
    def fromStructure(structure: dict[str, Any]) -> "nspikeTrain":
        """Reconstruct an ``nspikeTrain`` from a dict."""
        sampleRate = float(structure.get("sampleRate", 1000.0))
        return nspikeTrain(
            structure.get("spikeTimes", []),
            structure.get("name", ""),
            sampleRate if sampleRate > 0 else 1000.0,
            structure.get("minTime"),
            structure.get("maxTime"),
            structure.get("xlabelval", "time"),
            structure.get("xunits", "s"),
            structure.get("yunits", ""),
            structure.get("dataLabels", ""),
            -1,
        )


__all__ = ["nspikeTrain"]
