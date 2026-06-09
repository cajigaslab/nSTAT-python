# nSTAT and further study

> **Goal of this page.** nSTAT is a *toolbox*, but the Concepts track is built to
> teach. This page steps back and shows the whole arc — from point processes to
> the modern frontier — and exactly **where nSTAT fits, where it stops, and
> where to go next**. Use it as a map: to place each Concepts page in a larger
> course, and to find the on-ramps to material nSTAT does not implement.

## The arc: from point processes to foundation models

Computational neural data analysis, taught end to end, runs through six stages.
The first three are nSTAT's home ground; the last three are where nSTAT hands
off to a fuller curriculum (with on-ramps already in this track):

| Stage | The question | nSTAT's role |
|---|---|---|
| 1. **Foundations** — spikes, the LFP, point processes | What does an electrode measure, and how do we represent spikes? | **Implemented** |
| 2. **Encoding** — point-process GLMs, goodness-of-fit | What drives a neuron, and does the model fit? | **Implemented** |
| 3. **State & decoding** — filters, state-space, BCI read-out | What is the latent state, and can we read it from spikes? | **Implemented** |
| 4. **Populations** — dimensionality, neural manifolds | What low-dimensional structure does a population carry? | **On-ramp** (PCA sketch) |
| 5. **Deep learning** — learned encoders & decoders | When do learned models beat hand-built ones? | **Bridge page** |
| 6. **Foundation models** — pretrained, transferable | Can one model transfer across sessions and subjects? | **Pointer** |

## Where each Concepts page sits

Stages 1–3 are covered in depth, each page pairing intuition, the matching nSTAT
objects, runnable code, and primary references:

| Stage | Concepts page | Runnable companion |
|---|---|---|
| 1. Foundations | [Microelectrode recordings](microelectrode_recordings.md) | `Tutorial_MicroelectrodeToDecoding.ipynb` |
| 2. Encoding | [Spike trains & GLMs](spike_trains_and_glms.md) · [LFP & spectra](lfp_and_spectral.md) | [`encoding_to_goodness_of_fit.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/encoding_to_goodness_of_fit.py) |
| 2. Model checking | [Goodness-of-fit & decoding](goodness_of_fit_and_decoding.md) · [Uncertainty & CIs](uncertainty_and_confidence.md) | [`model_comparison.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/model_comparison.py) |
| 3. State & decoding | [State-space & EM](state_space_and_em.md) · [Network connectivity](network_connectivity.md) | [`decoding_ppaf.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/decoding_ppaf.py) · [`network_coupling.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/network_coupling.py) |
| 1–3 together | **(capstone)** | [`place_cell_walkthrough.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/place_cell_walkthrough.py) — real data, end to end |

The [place-cell walkthrough](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/place_cell_walkthrough.py)
is the keystone: it runs stages 1→3 on a real recording and ends on the lesson
that ties the track together — *a model can be useful for decoding yet still
fail goodness-of-fit.*

## Where nSTAT stops, and the on-ramps it provides

nSTAT deliberately does not implement stages 4–6 — they are large fields with
their own tooling. But this track gives you the on-ramps, so the hand-off is
gradual rather than a cliff:

- **Stage 4 — Populations.** [Population geometry](population_geometry.md) shows,
  with a few lines of NumPy, how a high-dimensional recording collapses onto a
  low-dimensional **neural manifold**, and points to GPFA and the
  dimensionality-reduction literature
  ([Cunningham & Yu 2014](https://pubmed.ncbi.nlm.nih.gov/25151264/)).
- **Stage 5 — Deep learning.** [From filters to deep learning](from_filters_to_deep_learning.md)
  draws the line from the PPAF to RNN/transformer decoders and sequential
  autoencoders — what carries over (encoding underlies decoding; goodness-of-fit
  and uncertainty still matter) and what changes.
- **Stage 6 — Foundation models.** Covered as a pointer in the same page: large
  models pretrained across sessions and subjects, aiming for decoders that
  *transfer*. This is an active research frontier, mostly published at
  machine-learning venues.

## Using this with a course

This track is a self-contained foundation for stages 1–3 of any
point-processes-to-foundation-models curriculum, and a guided ramp into 4–6. A
natural path:

1. Read the Concepts pages in the [suggested order](index.md#suggested-learning-path).
2. Run the four topic tutorials, then the real-data
   [capstone](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/place_cell_walkthrough.py).
3. Test yourself with the [self-check](self_check.md), including its
   cross-cutting synthesis questions.
4. Cross the bridges into [population geometry](population_geometry.md) and
   [deep learning](from_filters_to_deep_learning.md), then continue in the
   curriculum's own materials.

> **A note on this map.** The six-stage arc above is the *general* shape of the
> field, used here to position nSTAT. To align it chapter-by-chapter with a
> specific syllabus, fill the "Stage" rows with that course's chapter numbers —
> the toolbox mapping itself does not change.

## See also

- [Concepts index](index.md) — the learning path and the object-model crosswalk.
- [Annotated bibliography](bibliography.md) — the primary sources behind every
  stage.
- [Self-check](self_check.md) — quizzes for stages 1–3 plus synthesis questions.
