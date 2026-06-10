# Self-check: test your understanding

> **Goal of this page.** A single place to test what you have learned. Each
> concepts page ends with a short *Check your understanding* quiz (with
> answers); this page collects them, then adds **synthesis questions** that
> span several pages — the kind of reasoning a real analysis demands.
>
> **Need a refresher on a term?** Every term used in the quizzes is defined
> in the [glossary](glossary.md), with HTML anchors you can deep-link to.

## Per-topic quizzes

Work through the page, then try its quiz. Each links to the questions (answers
are collapsible, right there on the page):

| Topic | Quiz |
|---|---|
| Microelectrode recordings: spikes and the LFP | [Check your understanding](microelectrode_recordings.md#check-your-understanding) |
| Spike trains and point-process GLMs | [Check your understanding](spike_trains_and_glms.md#check-your-understanding) |
| The LFP and spectral analysis | [Check your understanding](lfp_and_spectral.md#check-your-understanding) |
| Goodness-of-fit and decoding | [Check your understanding](goodness_of_fit_and_decoding.md#check-your-understanding) |
| State-space models and EM | [Check your understanding](state_space_and_em.md#check-your-understanding) |
| Network connectivity and coupling | [Check your understanding](network_connectivity.md#check-your-understanding) |
| Uncertainty and confidence intervals | [Check your understanding](uncertainty_and_confidence.md#check-your-understanding) |
| Rhythmic firing and the clinical microelectrode | [Check your understanding](rhythmic_firing_and_clinical_microelectrode.md#check-your-understanding) |

## Synthesis questions

These cut across topics. Try to answer in a sentence or two before expanding.

1. **From electrode to encoding model.** You are handed a single broadband
   extracellular trace and asked to build an encoding model of one neuron's
   stimulus tuning. List the steps, in order, and name the nSTAT object or
   function at each.
2. **Two models, two verdicts.** Model B has a lower AIC than model A, but model
   B leaves the time-rescaling KS band while model A stays inside it. Which do
   you report, and why doesn't the lower AIC settle it?
3. **Useful vs. correct.** In the place-cell capstone the place-field model
   decodes position several times better than chance, yet fails goodness-of-fit
   for nearly every cell. Explain how both can be true, and name two things you
   would add to the encoding model to close the gap.
4. **Correlation is not connection.** Two neurons have a sharp peak at lag 0 in
   their cross-correlogram. Give two *distinct* explanations, only one of which
   is a direct synaptic connection, and say what analysis would help tell them
   apart.
5. **When the answer needs an interval.** You report that a neuron's stimulus
   coefficient is `β₁ = 0.5`. A colleague asks whether the neuron is really
   stimulus-driven. What single additional quantity do you need, and how would
   you compute it from the GLM fit?
6. **Static vs. evolving tuning.** You fit a GLM per trial and the stimulus
   coefficient seems to drift upward across a session. What model would you use
   to estimate that trajectory properly, and what does it give you that 100
   independent per-trial fits do not?
7. **A rhythm is just a covariate.** A neuron fires in time with a 5 Hz tremor
   and you have no external stimulus to regress against. Explain how you would
   still fit a point-process GLM that captures the rhythm, how you would prove it
   fit, and — using the *same* electrode — where you would look for the beta
   biomarker that drives adaptive DBS.

<details>
<summary>Show answers</summary>

1. **Filter → detect/sort → represent → specify → fit → check.** Band-split the
   broadband trace (high-pass >300 Hz for spikes, low-pass <300 Hz for the LFP;
   see [microelectrode recordings](microelectrode_recordings.md)); detect and
   sort spikes into an `nspikeTrain`; build the stimulus as a `Covariate`;
   bundle data + model in a `Trial` and `TrialConfig`; fit with `Analysis`
   (or `fit_poisson_glm`) to get a `FitResult`; then check it with
   `computeKSStats`. See the [spike-train GLM page](spike_trains_and_glms.md).
2. **Report model A.** AIC only ranks models *relative to each other*; it never
   certifies absolute fit. Leaving the KS band means model B is **misspecified**
   in absolute terms, so its lower AIC is the lowest score among inadequate
   models. Prefer the lowest-AIC model that **also passes** goodness-of-fit.
3. Decoding only needs the model to **rank positions correctly enough** to pick
   the right one; goodness-of-fit asks whether the model reproduces the spiking
   **exactly**. The place-field model captures coarse spatial tuning (so it
   decodes) but omits **spike history / refractoriness** and the **theta
   rhythm**, and uses a single broad Gaussian bump — so KS rejects it. Add
   history terms and a richer spatial basis (e.g. Paper Example 04's Zernike
   fields).
4. (a) A **direct synaptic connection** from one to the other; (b) a **shared
   common input** (a stimulus or network rhythm) driving both. A connection
   typically shows a short, *asymmetric, lagged* peak; common input shows a
   *symmetric* peak near zero lag. Conditioning on the stimulus/other neurons in
   an **ensemble GLM**, or a Granger-style test, helps separate them. See
   [network connectivity](network_connectivity.md).
5. You need the coefficient's **confidence interval** (equivalently its standard
   error). Compute the Fisher information `Xᵀ diag(λ) X`, invert it for the
   covariance, take `se = √diag`, and report `β₁ ± 1.96·se`. If the interval
   excludes 0 the neuron is convincingly stimulus-driven. See
   [uncertainty and confidence intervals](uncertainty_and_confidence.md).
6. The **state-space GLM (SSGLM)**, fit by EM. It treats the coefficient as a
   latent state evolving across trials and **shares statistical strength**
   between neighboring trials, giving a smoothed trajectory *with* credible
   intervals — far less noisy than 100 independent fits, and able to say whether
   the drift is real. See [state-space models and EM](state_space_and_em.md).
7. Build the rhythm itself as a **periodic covariate** — a `sin`/`cos` pair (or
   band-limited drive) at the tremor frequency — and fit the point-process GLM
   exactly as for any stimulus, with a **history** term for refractoriness.
   Prove it fit with the **time-rescaling KS test**: the rhythm-aware model stays
   in the band while a constant-rate model with the *same mean rate* is rejected
   for getting the timing wrong. Then **low-pass the same electrode** to a field
   potential and read **beta-band (13–30 Hz)** power with `SignalObj.MTMspectrum`
   (use `spectrogram` for burst structure). See
   [Rhythmic firing and the clinical microelectrode](rhythmic_firing_and_clinical_microelectrode.md).

</details>

## Where to go next

- Apply it end-to-end on real data:
  [`examples/tutorials/place_cell_walkthrough.py`](https://github.com/cajigaslab/nSTAT-python/blob/main/examples/tutorials/place_cell_walkthrough.py).
- See where to go beyond nSTAT: [Further study](further_study.md).
- Avoid the classic mistakes: [Common pitfalls & FAQ](pitfalls_and_faq.md).
