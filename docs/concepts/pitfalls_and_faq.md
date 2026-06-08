# Common pitfalls &amp; FAQ

> **Goal of this page.** The practical wisdom that ties the other Concepts
> pages together — the mistakes that quietly invalidate an analysis, and how
> to avoid them. Skim it before your first real analysis, and return when a
> result looks wrong.

## Modeling and binning

**How do I choose the bin width?**
Two competing pressures. For the **time-rescaling KS test**
([goodness-of-fit page](goodness_of_fit_and_decoding.md)) the bins should be
fine enough that each holds **at most one spike** — otherwise the rescaled
intervals are quantized and the test rejects good models. At typical cortical
rates, 1 ms bins are safe. For **GLM fitting** alone, coarser bins (5–10 ms)
are fine and faster, as long as you keep the `log(bin_width)` offset so the
coefficients stay rates. When in doubt, bin at 1 ms.

**My KS plot fails right near 0 — why?**
That region reflects the **shortest** inter-spike intervals, which are governed
by the refractory period and short-timescale history. A failure there almost
always means the model has **no (or too little) spike-history**. Add history
terms (`history_window_times` in `TrialConfig`); see the
[GLM page](spike_trains_and_glms.md).

**My KS curve bows smoothly away from the diagonal.**
A smooth bow (not a near-0 spike) usually means the overall **rate is
mis-scaled** — often a units bug. Check that `sampleRate` and bin widths are
consistent (the original MATLAB toolbox had a `sampleRate`-vs-`1/sampleRate`
bug here; nSTAT-python fixes it).

## Goodness-of-fit and model comparison

**A model has the lowest AIC — am I done?**
No. AIC/BIC only rank models *relative to each other*; the winner can still be
absolutely wrong. **Always** confirm with goodness-of-fit
(`FitResult.computeKSStats`). Lowest AIC **and** passes KS → trust it.

**Every neuron passes its KS test, but the population model seems off.**
Per-neuron KS checks each neuron's *marginal* intensity in isolation; it is
blind to **coupling**. A pair of synchronous neurons modeled as independent
passes per-neuron but fails jointly. Use
`population_time_rescale` (the [Tao et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30298220/)
marked test) for population models.

**I tested 200 neurons and ~10 "significantly" fail at p&lt;0.05.**
That is the expected false-positive rate under the null. With many neurons,
**correct for multiple comparisons** (e.g. Benjamini–Hochberg FDR) before
declaring misfit.

**My tuning estimate looks unstable across the session.**
The plain GLM assumes **fixed** tuning. If it genuinely drifts (learning,
adaptation), that is a modeling choice, not noise — move to the state-space
GLM (`nstColl.ssglm`/`ssglmFB`); see the
[state-space page](state_space_and_em.md).

## Spectral analysis (LFP/EEG)

**My spectrum is noisy and spiky.**
You are likely looking at a raw **periodogram**, whose variance does not shrink
with more data. Use the **multitaper** estimate (`SignalObj.MTMspectrum`); see
the [LFP page](lfp_and_spectral.md).

**Two nearby peaks blur into one (or a real peak splits).**
That is the **time–bandwidth (`NW`) trade-off**. Large `NW` over-smooths and
merges close peaks; too-small `NW` is noisy. Pick `NW` for the question —
small to separate close rhythms, larger for broadband power.

**I see a sharp line at 60 Hz (or 50 Hz).**
That is **mains/line noise**, not brain activity. Notch-filter or exclude that
band before interpreting gamma.

## Decoding and state-space EM

**My decode from one neuron is terrible.**
Expected — a single neuron is ambiguous. **Decoding is a population operation;**
RMSE drops sharply as you add cells with diverse tuning (the
[decoding tutorial](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/decoding_ppaf.py)
shows this directly).

**My EM fit collapsed (transition matrix → 0, or wildly different per run).**
Point-process state-space EM has a **weak-observability** failure mode and only
finds a *local* optimum. Use the multi-restart workflow with held-out
predictive log-likelihood (`fit_point_process_em_best_of`), and the
`init="log_empirical_rate"` / `ridge_lambda` options; see the
[state-space page](state_space_and_em.md) and the
[EM extras guide](../extras/em_dynamax.md).

## Data and provenance

**Where do spike times come from? Does nSTAT sort spikes?**
No. nSTAT consumes **already-sorted** spike trains. Detection and sorting are a
separate pipeline (e.g.
[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)); bring the
results in via the [interop bridges](../extras.rst). See the
[microelectrode page](microelectrode_recordings.md).

**Does spike-sorting error affect my results?**
Yes — misassigned spikes bias encoding and decoding
([Harris et al. 2000](https://pubmed.ncbi.nlm.nih.gov/10899214/)). If sorting
is unreliable, consider **clusterless** decoding, which skips sorting entirely
(`nstat.extras.decoding.clusterless_bridge`).

## See also

- [Glossary](glossary.md) · [Annotated bibliography](bibliography.md)
- Back to the [Concepts overview](index.md).
