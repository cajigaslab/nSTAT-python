# Concepts &amp; Background

This section teaches the neuroscience and statistics behind nSTAT, not just
the API. It is written for students and newcomers: each page builds intuition
first, then shows the matching nSTAT objects and a runnable snippet, and cites
the primary literature so you can go deeper. No prior neuroscience is assumed.

## Suggested learning path

1. **[Microelectrode recordings: spikes and the LFP](microelectrode_recordings.md)**
   — what an electrode actually measures, how the broadband signal splits into
   spikes and the LFP, single- vs multi-unit activity, and spike sorting.
   *The physical grounding for everything else.*
2. **[Spike trains and point-process GLMs](spike_trains_and_glms.md)** —
   spike trains as point processes, the conditional intensity function, and
   fitting it with point-process GLMs (stimulus + history + ensemble).
   *The core encoding model.*
3. **[The LFP and spectral analysis](lfp_and_spectral.md)** — the local field
   potential and the continuous-signal tools: multitaper spectra,
   spectrograms, and Kalman filtering (applies to LFP, EEG, ECoG).
4. **[Goodness-of-fit and decoding](goodness_of_fit_and_decoding.md)** —
   the time-rescaling KS test, population goodness-of-fit, and reading the
   stimulus/state back out with point-process and clusterless decoders.

Hands-on companion: the runnable tutorial
[`examples/tutorials/encoding_to_goodness_of_fit.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/encoding_to_goodness_of_fit.py)
walks the whole arc — simulate a stimulus-driven neuron, fit a point-process
GLM, and run the time-rescaling goodness-of-fit test (with a correct-vs-wrong
model contrast). It needs no data download.

Reference material:

- **[Glossary](glossary.md)** — plain-language definitions, each linked to the
  relevant nSTAT object.
- **[Annotated bibliography](bibliography.md)** — the primary sources, with a
  note on why each matters for nSTAT users.

## How the concepts map to nSTAT

| Concept | nSTAT objects | Example |
|---|---|---|
| Spike trains | `nspikeTrain`, `nstColl` | `nSpikeTrainExamples.ipynb` |
| Encoding GLM | `Analysis`, `TrialConfig`, `FitResult` | Paper Example 02 |
| Across-trial learning | `nstColl.ssglm()` (SSGLM) | Paper Example 03 |
| Goodness-of-fit | `FitResult.computeKSStats`, `population_time_rescale` | Paper Examples 01–03 |
| LFP / spectra | `SignalObj` (`MTMspectrum`, `spectrogram`) | `SignalObjExamples.ipynb` |
| Decoding | `DecodingAlgorithms` (PPAF/PPHF), `clusterless_bridge` | Paper Example 05 |

Once the concepts are clear, see the
[Paper-aligned toolbox map](../PaperOverview.md) for the full API crosswalk and
the [paper-example gallery](../paper_examples.md) for worked analyses with
figures.

```{toctree}
:maxdepth: 1
:hidden:

microelectrode_recordings
spike_trains_and_glms
lfp_and_spectral
goodness_of_fit_and_decoding
glossary
bibliography
```
