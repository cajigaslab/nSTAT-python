# Further study

> **Goal of this page.** A short map of where to go beyond nSTAT — the topics
> this toolbox does not implement, with primary references.
>
> **See also:** the [glossary](glossary.md) defines every term used across
> the concepts track, with HTML anchors for direct linking.

## What nSTAT covers

The implemented topics are organized in the [concepts index](index.md):

- spikes and the LFP from a microelectrode;
- spike trains as point processes and the encoding GLM;
- the LFP and spectral analysis;
- goodness-of-fit (time-rescaling) and decoding (PPAF / PPHF / clusterless);
- the state-space GLM and EM-trained latent state-space models;
- functional connectivity (ensemble GLM, Granger);
- uncertainty and confidence intervals;
- rhythmic firing and the clinical microelectrode.

Each page pairs intuition with the matching nSTAT objects, runnable snippets,
and primary references.

## What nSTAT does not — and where to learn it

- **Population geometry and dimensionality reduction.** nSTAT ships only a
  PCA sketch ([population geometry](population_geometry.md)). For the standard
  tooling see Gaussian-Process Factor Analysis
  ([Yu et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19357332/)) and the
  dimensionality-reduction guide
  ([Cunningham & Yu 2014](https://pubmed.ncbi.nlm.nih.gov/25151264/)); for the
  dynamical-systems view see
  [Vyas et al. 2020](https://pubmed.ncbi.nlm.nih.gov/32640928/).
- **Deep-learning encoders and decoders.** The bridge page
  [From filters to deep learning](from_filters_to_deep_learning.md) draws the
  line from the PPAF to modern sequence decoders — what carries over
  (encoding underlies decoding; goodness-of-fit and uncertainty still matter)
  and what changes, with pointers into the literature.
- **Spike sorting.** nSTAT consumes already-sorted spike trains. For raw
  acquisition to sorted units, use a dedicated tool such as
  [SpikeInterface](https://github.com/SpikeInterface/spikeinterface), then
  bring the results in via the [interop bridges](../extras.rst).
- **Vendor-format I/O.** The [interop bridges](../extras.rst)
  (`nstat.extras.interop.{neo,nwb,pynapple}`) read Spike2 / Blackrock / Plexon
  / NEX / TDT and NWB into `nspikeTrain` / `SignalObj`.

## See also

- [Concepts index](index.md) — the learning path and the object-model crosswalk.
- [Annotated bibliography](bibliography.md) — primary sources behind every
  concepts page.
- [Self-check](self_check.md) — per-topic quizzes plus cross-cutting synthesis
  questions.
