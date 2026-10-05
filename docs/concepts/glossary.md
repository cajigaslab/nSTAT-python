# Glossary

Plain-language definitions of the terms used across the Concepts pages and the
nSTAT API. Where a term maps to a specific object or method, the name is given
in `code font`.

<!-- Each term is a `###` heading preceded by a slug marker comment.
     docs/conf.py reads these markers (myst_heading_slug_func) so the heading
     anchor is the short, stable slug the Concepts pages link to. -->

## Recording and signals

<!-- slug: action-potential -->
### Action potential (spike)

The brief (~1 ms) all-or-none electrical event a
neuron emits when it fires. nSTAT represents a neuron's spikes as a list of
times in an `nspikeTrain`.

<!-- slug: microelectrode -->
### Microelectrode

A fine metal/silicon probe that measures the extracellular
voltage produced by nearby neural currents. See
[Microelectrode recordings](microelectrode_recordings.md).

<!-- slug: broadband-signal -->
### Broadband signal

The raw wideband voltage from an electrode, before
filtering — it contains both spikes and the LFP.

<!-- slug: local-field-potential -->
### Local field potential (LFP)

The low-frequency (~1–300 Hz) part of the
extracellular signal, reflecting summed synaptic/subthreshold currents of a
local population. Represented as a `SignalObj`. See
[The LFP and spectral analysis](lfp_and_spectral.md).

<!-- slug: eeg-ecog -->
### EEG / ECoG

Field potentials recorded at the scalp (EEG) or cortical
surface (ECoG); analyzed with the same `SignalObj` spectral tools as the LFP.

<!-- slug: single-unit -->
### Single unit

Spikes attributed to one isolated neuron after spike sorting.

<!-- slug: multi-unit-activity -->
### Multi-unit activity (MUA)

Pooled spikes from several nearby neurons that
could not be separated into single units.

<!-- slug: spike-sorting -->
### Spike sorting

The pipeline (detect → extract features → cluster) that
turns a broadband trace into per-neuron spike trains. nSTAT assumes this is
already done; see [Lewicki 1998](https://pubmed.ncbi.nlm.nih.gov/10221571/).

<!-- slug: tetrode-multi-electrode-array -->
### Tetrode / multi-electrode array

A probe with several nearby contacts;
viewing each spike from multiple sites improves sorting accuracy.

## Clinical microelectrode recordings and rhythms

<!-- slug: rhythmic-oscillatory-cell -->
### Rhythmic / oscillatory cell

A neuron whose firing probability rises and
falls periodically, even without a changing stimulus. Modeled in nSTAT with a
periodic `Covariate` in the point-process GLM. See
[Rhythmic firing and the clinical microelectrode](rhythmic_firing_and_clinical_microelectrode.md).

<!-- slug: tremor-cell -->
### Tremor cell

A rhythmic cell whose firing is phase-locked to a few-hertz
(~3–8 Hz) limb tremor; characterized in the human subthalamic nucleus and
thalamus during DBS surgery. See
[Levy et al. 2000](https://pubmed.ncbi.nlm.nih.gov/11027240/).

<!-- slug: deep-brain-stimulation -->
### Deep brain stimulation (DBS)

A therapy that delivers electrical stimulation
through an electrode implanted in a deep brain nucleus (e.g. the subthalamic
nucleus, STN) to treat Parkinson's disease and other disorders.

<!-- slug: microelectrode-mapping-localization -->
### Microelectrode mapping / localization

Advancing a recording microelectrode
millimetre by millimetre to identify a deep target from the firing-rate,
burstiness, and spectral signatures of each nucleus it passes through. A
latent-state / change-point problem. See
[Hutchison et al. 1998](https://pubmed.ncbi.nlm.nih.gov/9778260/).

<!-- slug: beta-band -->
### Beta band (13–30 Hz)

A field-potential rhythm whose power in the STN tracks
Parkinsonian motor impairment; the feedback signal for **adaptive DBS**.
Estimated with `SignalObj.MTMspectrum`. See
[Little et al. 2013](https://pubmed.ncbi.nlm.nih.gov/23852650/),
[Tinkhauser et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28334851/).

<!-- slug: adaptive-dbs -->
### Adaptive (closed-loop) DBS

Stimulation gated by a measured biomarker (e.g.
beta power) rather than delivered continuously — a decode-then-actuate loop.

## Point processes and modeling

<!-- slug: point-process -->
### Point process

A probabilistic model for the timing of discrete events
(spikes). The right framework for spike trains.

<!-- slug: conditional-intensity-function -->
### Conditional intensity function (CIF), $\lambda(t \mid H_t)$

The instantaneous
firing rate at time $t$ given the history $H_t$; $\lambda \cdot \Delta$ is the spike
probability in a small interval. The complete description of a point process.
In nSTAT: `CIF`, `CIFModel`, `LinearCIF`.

<!-- slug: history -->
### History $H_t$

Everything observed up to time $t$ — the neuron's own past
spikes, the ensemble's spikes, and covariates — that the CIF may depend on.

<!-- slug: homogeneous-inhomogeneous-poisson-process -->
### Homogeneous / inhomogeneous Poisson process

Point process with constant
rate ($\lambda$) / time-varying rate ($\lambda(t)$) and *no* history dependence.

<!-- slug: generalized-linear-model -->
### Generalized linear model (GLM)

Here, a model of $\log \lambda(t \mid H_t)$ as a
linear sum of covariate, history, and ensemble terms. Fit by
`Analysis`/`fit_poisson_glm`; configured by `TrialConfig`. See
[Spike trains and point-process GLMs](spike_trains_and_glms.md).

<!-- slug: link-function -->
### Link function

The transform applied to the rate; nSTAT uses the **log**
link so $\lambda > 0$ and covariates act multiplicatively.

<!-- slug: covariate -->
### Covariate

An external (extrinsic) signal — stimulus, position, movement —
that may drive firing. In nSTAT: `Covariate`, grouped in a `CovColl`.

<!-- slug: basis -->
### Basis (e.g. spline)

A set of functions used to expand a covariate so its
effect on firing can be nonlinear.

<!-- slug: history-term-refractory-period -->
### History term / refractory period

History covariates capture the neuron's
dependence on its own recent spikes; the dip just after a spike (no immediate
re-firing) is the refractory period.

<!-- slug: ensemble-functional-coupling -->
### Ensemble / functional coupling

Dependence of one neuron's firing on other
neurons', beyond shared stimulus drive.

<!-- slug: aic-bic -->
### AIC / BIC

Penalized-likelihood scores for comparing models of differing
complexity (`fit.AIC`, `fit.BIC`). Lower is better — but confirm with
goodness-of-fit.

<!-- slug: state-space-glm -->
### State-space GLM (SSGLM)

A GLM whose coefficients evolve across trials (a
latent state), estimated by EM; captures learning. `nstColl.ssglm()`/`ssglmFB()`.
See [Smith & Brown 2003](https://pubmed.ncbi.nlm.nih.gov/12803953/).

## Goodness-of-fit and decoding

<!-- slug: time-rescaling-theorem -->
### Time-rescaling theorem

If the CIF is correct, integrating it between
spikes yields i.i.d. unit-rate exponential intervals — the basis of the KS
goodness-of-fit test. `FitResult.computeKSStats`. See
[Brown et al. 2002](https://pubmed.ncbi.nlm.nih.gov/11802915/).

<!-- slug: kolmogorov-smirnov-test-ks-plot -->
### Kolmogorov–Smirnov (KS) test / KS plot

A test of whether the rescaled
intervals match the expected distribution; the KS plot shows the empirical CDF
against the diagonal with confidence bands.

<!-- slug: population-time-rescaling -->
### Population (marked) time-rescaling

A joint goodness-of-fit test for a
*population* that catches inter-neuron coupling a per-neuron test misses.
`population_time_rescale`. See
[Tao et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30298220/).

<!-- slug: encoding-vs-decoding -->
### Encoding vs. decoding

Encoding models predict spikes from
stimulus/state (the GLM); decoding infers stimulus/state from spikes.

<!-- slug: point-process-adaptive-filter -->
### Point-process adaptive filter (PPAF)

Recursive Bayesian decoder — the
spiking analogue of the Kalman filter — that estimates a continuous state from
a population's spikes. `DecodingAlgorithms`. See
[Eden et al. 2004](https://pubmed.ncbi.nlm.nih.gov/15070506/).

<!-- slug: hybrid-point-process-filter -->
### Hybrid point-process filter (PPHF)

Jointly estimates a discrete mode and
a continuous state from spikes.

<!-- slug: kalman-filter-smoother -->
### Kalman filter / smoother

Optimal recursive estimator of a latent state
from *Gaussian* observations (e.g. LFP); the smoother uses the whole record.

<!-- slug: clusterless-decoding -->
### Clusterless decoding

Decoding directly from spike-waveform features
("marks") without spike sorting. `nstat.extras.decoding.clusterless_bridge`.
See [Denovellis et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34570699/).

## Spectral analysis

<!-- slug: power-spectral-density -->
### Power spectral density (PSD)

How a signal's power is distributed across
frequency.

<!-- slug: periodogram -->
### Periodogram

The naive squared-FFT spectrum estimate; high variance and
spectral leakage. `SignalObj.periodogram`.

<!-- slug: multitaper-method -->
### Multitaper method

A low-variance, leakage-controlled spectrum estimate
that averages over orthogonal Slepian (DPSS) tapers. `SignalObj.MTMspectrum`.
See [Thomson 1982](https://doi.org/10.1109/PROC.1982.12433),
[Mitra & Pesaran 1999](https://pubmed.ncbi.nlm.nih.gov/9929474/).

<!-- slug: time-bandwidth-product -->
### Time–bandwidth product $NW$

Sets the multitaper smoothing/resolution
trade-off; the number of tapers is $K \approx 2 \cdot NW - 1$.

<!-- slug: spectrogram -->
### Spectrogram

Power as a function of both time and frequency, from a
sliding-window multitaper estimate. `SignalObj.spectrogram`.

---

See the [Concepts overview](index.md) for the full learning path and the
[Bibliography](bibliography.md) for sources.
