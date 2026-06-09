# From filters to deep learning

> **Goal of this page.** nSTAT's decoders — the point-process adaptive filter
> and its relatives — are **linear, model-based** estimators with a long, proven
> track record. Modern brain–computer interfaces increasingly add
> **deep-learning** decoders on top. This page draws the line from one to the
> other: what carries over, what changes, and where to go to learn the rest. It
> deliberately does not reimplement those methods — it places nSTAT in the arc.

## What nSTAT's decoders already give you

The [decoding page](goodness_of_fit_and_decoding.md) builds up the
point-process adaptive filter (PPAF) and hybrid filter (PPHF). Stripped to
essentials, every one of these decoders is a **predict/correct** loop:

1. **Predict** the latent state forward with a dynamics model (a random walk, or
   linear kinematics).
2. **Correct** that prediction using the spikes just observed, weighted by each
   neuron's fitted tuning (its conditional intensity).

This is the point-process cousin of the Kalman filter, and it has real virtues:
it is **interpretable** (every term is a tuning curve or a dynamics
coefficient), **data-efficient**, and it reports **calibrated uncertainty**
(the [credible band](uncertainty_and_confidence.md)). For many BCIs a linear
filter is still a strong, hard-to-beat baseline.

Its limits are equally clear. The encoding is **linear in the state** (through
the link function), the dynamics are **linear**, and the model is **specified by
hand**. When tuning is strongly nonlinear, non-stationary, or high-dimensional,
a learned function can do better.

## What deep-learning decoders change

Deep-learning decoders keep the *goal* — map spikes to a stimulus, intention,
or movement — but replace hand-specified pieces with **learned** ones:

| Piece | Classical (nSTAT) | Deep-learning decoder |
|---|---|---|
| Encoding / tuning | fitted CIF, linear-in-state | learned nonlinear function |
| Temporal dynamics | linear (random walk / kinematics) | RNN / LSTM / temporal CNN / transformer |
| Model specification | written by the analyst | learned from data |
| Uncertainty | closed-form posterior covariance | needs explicit modeling (often absent) |
| Data appetite | modest | large |

Concretely: recurrent-network decoders and sequence models often outperform a
Kalman filter on rich motor BCIs
([Glaser et al. 2020](https://pubmed.ncbi.nlm.nih.gov/32737181/)); the
high-performance handwriting BCI paired a recurrent decoder with intracortical
spikes ([Willett et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33981047/)); and
sequential autoencoders such as **LFADS** infer single-trial population dynamics
that a linear filter cannot capture
([Pandarinath et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30224673/)). The
trade is the one in the table: more flexibility and accuracy, at the cost of
more data, less interpretability, and uncertainty you must add back deliberately.

## …toward foundation models

The current frontier extends this further: **foundation models** for neural
data — large sequence models pretrained across many sessions, subjects, and
tasks, then adapted to a new recording with little data. They treat populations
of spikes the way language models treat tokens, aiming for decoders that
**transfer** rather than being retrained from scratch each session. This is an
active, fast-moving research area (much of it published at machine-learning
venues rather than indexed in PubMed); the [further-study
map](curriculum_and_further_study.md) points to where a full curriculum covers
it.

## How to think about the jump

Three ideas carry all the way from the PPAF to a foundation model, and keep you
oriented:

- **Encoding still underlies decoding.** Whether the tuning is a fitted CIF or a
  learned network, decoding inverts an encoding model. The
  [encoding GLM](spike_trains_and_glms.md) is the concept that never goes away.
- **Goodness-of-fit still matters.** A flexible model can overfit; the
  discipline of [checking the model](goodness_of_fit_and_decoding.md) and
  holding out data is *more* important as capacity grows, not less.
- **Uncertainty does not come for free.** The PPAF hands you a calibrated band;
  most deep decoders do not. Knowing *how confident* a decode is remains
  essential for a safe BCI — see [uncertainty](uncertainty_and_confidence.md).

Start where nSTAT is strong — interpretable, well-calibrated, model-based
decoding — and you will understand exactly what a learned decoder is buying you,
and what it is quietly giving up.

## See also

- [Goodness-of-fit and decoding](goodness_of_fit_and_decoding.md) — the
  classical decoders this page builds from.
- [Population geometry](population_geometry.md) — the low-dimensional view that
  modern population models exploit.
- [nSTAT and further study](curriculum_and_further_study.md) — the full map from
  this toolbox to a foundation-model curriculum.
