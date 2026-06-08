# Concepts &amp; Background

This section teaches the neuroscience and statistics behind nSTAT, not just
the API. It is written for students and newcomers: each page builds intuition
first, then shows the matching nSTAT objects and a runnable snippet, and cites
the primary literature so you can go deeper. No prior neuroscience is assumed.

**Jump in by goal:**
*I want to…* &nbsp;
[fit a GLM](spike_trains_and_glms.md) ·
[check a model's fit](goodness_of_fit_and_decoding.md) ·
[analyze LFP / spectra](lfp_and_spectral.md) ·
[decode a stimulus/state](goodness_of_fit_and_decoding.md) ·
[model learning across trials](state_space_and_em.md) ·
[measure coupling between neurons](network_connectivity.md) ·
[avoid common mistakes](pitfalls_and_faq.md).

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
5. **[State-space models: learning dynamics and EM](state_space_and_em.md)** —
   models whose parameters change over time: the across-trial state-space GLM
   (SSGLM) and EM-trained latent state-space models. *Where to go after the
   static GLM.*
6. **[Network connectivity and functional coupling](network_connectivity.md)** —
   how neurons influence each other (ensemble GLM terms, cross-correlograms,
   Granger), and why correlation is not connection.

Hands-on companions (both run on simulated data, no download):

- Notebook —
  [`examples/tutorials/Tutorial_MicroelectrodeToDecoding.ipynb`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/Tutorial_MicroelectrodeToDecoding.ipynb):
  a guided tour spanning every topic above (spikes vs. LFP, spike trains,
  multitaper spectra, GLM fitting, goodness-of-fit, decoding), with figures.
- Script —
  [`examples/tutorials/encoding_to_goodness_of_fit.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/encoding_to_goodness_of_fit.py):
  the encoding → GLM → goodness-of-fit arc as a four-act lesson, with a
  correct-vs-wrong model contrast.

Reference material:

- **[Common pitfalls & FAQ](pitfalls_and_faq.md)** — the mistakes that quietly
  invalidate an analysis, and how to avoid them. Skim before your first real
  analysis.
- **[Glossary](glossary.md)** — plain-language definitions, each linked to the
  relevant nSTAT object.
- **[Annotated bibliography](bibliography.md)** — the primary sources, with a
  note on why each matters for nSTAT users.

## How the concepts map to nSTAT

![The nSTAT analysis pipeline: raw data to nSTAT objects to Trial+TrialConfig to Analysis to FitResult, branching into goodness-of-fit, model comparison, and decoding](figures/workflow.png)

*The pipeline at a glance: raw data become nSTAT objects, a `Trial` plus a
`TrialConfig` specify the model, `Analysis` fits it, and the resulting
`FitResult` feeds goodness-of-fit, model comparison, and decoding.*

| Concept | nSTAT objects | Example |
|---|---|---|
| Spike trains | `nspikeTrain`, `nstColl` | `nSpikeTrainExamples.ipynb` |
| Encoding GLM | `Analysis`, `TrialConfig`, `FitResult` | Paper Example 02 |
| Across-trial learning (SSGLM) | `nstColl.ssglm()` / `ssglmFB()` | Paper Example 03 |
| EM-trained state-space models | `nstat.extras.em.dynamax_bridge` | `examples/extras/em_dynamax_demo.py` |
| Goodness-of-fit | `FitResult.computeKSStats`, `population_time_rescale` | Paper Examples 01–03 |
| LFP / spectra | `SignalObj` (`MTMspectrum`, `spectrogram`) | `SignalObjExamples.ipynb` |
| Decoding | `DecodingAlgorithms` (PPAF/PPHF), `clusterless_bridge` | Paper Example 05 |
| Functional coupling | `TrialConfig` ensemble terms, `Analysis` (Granger) | `network_coupling.py` |

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
state_space_and_em
network_connectivity
pitfalls_and_faq
glossary
bibliography
```
