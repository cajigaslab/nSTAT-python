# Annotated bibliography

The Concepts pages cite these works inline (each citation links straight to
PubMed or the publisher). This page collects the full references with a
one-line note on *why* it matters for users of nSTAT. PMIDs were verified
against PubMed; classic engineering/statistics works that predate PubMed
indexing are listed without a PMID.

## Microelectrode recordings, spikes, and the LFP

- **Buzsáki G, Anastassiou CA, Koch C (2012).** The origin of extracellular
  fields and currents — EEG, ECoG, LFP and spikes. *Nature Reviews
  Neuroscience* 13:407–420.
  [PMID 22595786](https://pubmed.ncbi.nlm.nih.gov/22595786/) ·
  [doi:10.1038/nrn3241](https://doi.org/10.1038/nrn3241).
  *The canonical explanation of what a microelectrode actually measures —
  why fast transients are spikes and the low-frequency remainder is the LFP.*
- **Einevoll GT, Kayser C, Logothetis NK, Panzeri S (2013).** Modelling and
  analysis of local field potentials for studying the function of cortical
  circuits. *Nature Reviews Neuroscience* 14:770–785.
  [PMID 24135696](https://pubmed.ncbi.nlm.nih.gov/24135696/) ·
  [doi:10.1038/nrn3599](https://doi.org/10.1038/nrn3599).
  *What the LFP reflects and how to interpret it — background for the
  spectral tools in nSTAT.*
- **Pesaran B, Vinck M, Einevoll GT, et al. (2018).** Investigating
  large-scale brain dynamics using field potential recordings: analysis and
  interpretation. *Nature Neuroscience* 21:903–919.
  [PMID 29942039](https://pubmed.ncbi.nlm.nih.gov/29942039/) ·
  [doi:10.1038/s41593-018-0171-8](https://doi.org/10.1038/s41593-018-0171-8).
  *Modern practical guide to field-potential analysis, including the
  pitfalls of spectral estimation that multitaper methods address.*
- **Stevenson IH, Kording KP (2011).** How advances in neural recording
  affect data analysis. *Nature Neuroscience* 14:139–142.
  [PMID 21270781](https://pubmed.ncbi.nlm.nih.gov/21270781/) ·
  [doi:10.1038/nn.2731](https://doi.org/10.1038/nn.2731).
  *Why population-scale recordings demand statistical models like the
  point-process GLMs at the heart of nSTAT.*

## Spike sorting

- **Lewicki MS (1998).** A review of methods for spike sorting: the detection
  and classification of neural action potentials. *Network: Computation in
  Neural Systems* 9:R53–R78.
  [PMID 10221571](https://pubmed.ncbi.nlm.nih.gov/10221571/) ·
  [doi:10.1088/0954-898X_9_4_001](https://doi.org/10.1088/0954-898X_9_4_001).
  *Foundational review of the spike-sorting problem nSTAT assumes is already
  solved (it works on sorted spike trains).*
- **Harris KD, Henze DA, Csicsvari J, Hirase H, Buzsáki G (2000).** Accuracy
  of tetrode spike separation as determined by simultaneous intracellular and
  extracellular measurements. *Journal of Neurophysiology* 84:401–414.
  [PMID 10899214](https://pubmed.ncbi.nlm.nih.gov/10899214/) ·
  [doi:10.1152/jn.2000.84.1.401](https://doi.org/10.1152/jn.2000.84.1.401).
  *Quantifies how imperfect spike sorting is — motivation for clusterless
  decoding (see below).*
- **Quian Quiroga R, Nadasdy Z, Ben-Shaul Y (2004).** Unsupervised spike
  detection and sorting with wavelets and superparamagnetic clustering.
  *Neural Computation* 16:1661–1687.
  [PMID 15228749](https://pubmed.ncbi.nlm.nih.gov/15228749/) ·
  [doi:10.1162/089976604774201631](https://doi.org/10.1162/089976604774201631).
  *A widely used spike-sorting algorithm; good background on spike features.*

## Point processes, GLMs, and goodness-of-fit

- **Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005).** A point
  process framework for relating neural spiking activity to spiking history,
  neural ensemble, and extrinsic covariate effects. *Journal of
  Neurophysiology* 93:1074–1089.
  [PMID 15356183](https://pubmed.ncbi.nlm.nih.gov/15356183/) ·
  [doi:10.1152/jn.00697.2004](https://doi.org/10.1152/jn.00697.2004).
  *The point-process-GLM framework nSTAT implements: stimulus + history +
  ensemble terms in one conditional intensity function.*
- **Paninski L (2004).** Maximum likelihood estimation of cascade
  point-process neural encoding models. *Network: Computation in Neural
  Systems* 15:243–262.
  [PMID 15600233](https://pubmed.ncbi.nlm.nih.gov/15600233/) ·
  [doi:10.1088/0954-898X_15_4_002](https://doi.org/10.1088/0954-898X_15_4_002).
  *Why the log-likelihood of a point-process GLM is concave — the reason GLM
  fitting in nSTAT converges reliably.*
- **Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002).** The
  time-rescaling theorem and its application to neural spike train data
  analysis. *Neural Computation* 14:325–346.
  [PMID 11802915](https://pubmed.ncbi.nlm.nih.gov/11802915/) ·
  [doi:10.1162/08997660252741149](https://doi.org/10.1162/08997660252741149).
  *The theorem behind `FitResult.computeKSStats` — how to turn a fitted CIF
  into a Kolmogorov–Smirnov goodness-of-fit test.*
- **Tao L, Weber KM, Arai K, Eden UT (2018).** A common goodness-of-fit
  framework for neural population models using marked point process
  time-rescaling. *Journal of Computational Neuroscience* 45:147–162.
  [PMID 30298220](https://pubmed.ncbi.nlm.nih.gov/30298220/) ·
  [doi:10.1007/s10827-018-0698-4](https://doi.org/10.1007/s10827-018-0698-4).
  *The multivariate population goodness-of-fit implemented by
  `nstat.population_time_rescale`.*
- **Lewis PAW, Shedler GS (1979).** Simulation of nonhomogeneous Poisson
  processes by thinning. *Naval Research Logistics Quarterly* 26:403–413.
  [doi:10.1002/nav.3800260304](https://doi.org/10.1002/nav.3800260304).
  *The thinning algorithm nSTAT uses to simulate spike trains from a
  time-varying rate.*

## State-space models and decoding

- **Smith AC, Brown EN (2003).** Estimating a state-space model from point
  process observations. *Neural Computation* 15:965–991.
  [PMID 12803953](https://pubmed.ncbi.nlm.nih.gov/12803953/) ·
  [doi:10.1162/089976603765202622](https://doi.org/10.1162/089976603765202622).
  *The EM algorithm behind the state-space GLM (SSGLM) for across-trial
  learning dynamics.*
- **Eden UT, Frank LM, Barbieri R, Solo V, Brown EN (2004).** Dynamic
  analysis of neural encoding by point process adaptive filtering. *Neural
  Computation* 16:971–998.
  [PMID 15070506](https://pubmed.ncbi.nlm.nih.gov/15070506/) ·
  [doi:10.1162/089976604773135069](https://doi.org/10.1162/089976604773135069).
  *The point-process adaptive filter (PPAF) used for decoding in nSTAT.*
- **Zhang K, Ginzburg I, McNaughton BL, Sejnowski TJ (1998).** Interpreting
  neuronal population activity by reconstruction: unified framework with
  application to hippocampal place cells. *Journal of Neurophysiology*
  79:1017–1044.
  [PMID 9463459](https://pubmed.ncbi.nlm.nih.gov/9463459/) ·
  [doi:10.1152/jn.1998.79.2.1017](https://doi.org/10.1152/jn.1998.79.2.1017).
  *The Bayesian population-reconstruction decoder used in the place-cell
  walkthrough (`examples/tutorials/place_cell_walkthrough.py`).*
- **Brown EN, Frank LM, Tang D, Quirk MC, Wilson MA (1998).** A statistical
  paradigm for neural spike train decoding applied to position prediction
  from ensemble firing patterns of rat hippocampal place cells. *Journal of
  Neuroscience* 18:7411–7425.
  [PMID 9736661](https://pubmed.ncbi.nlm.nih.gov/9736661/) ·
  [doi:10.1523/JNEUROSCI.18-18-07411.1998](https://doi.org/10.1523/JNEUROSCI.18-18-07411.1998).
  *Foundational decoding of position from place-cell ensembles; the dataset
  family behind the place-cell walkthrough.*
- **Denovellis EL, Gillespie AK, Coulter ME, et al. (2021).** Hippocampal
  replay of experience at real-world speeds. *eLife* 10:e64505.
  [PMID 34570699](https://pubmed.ncbi.nlm.nih.gov/34570699/) ·
  [doi:10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505).
  *The clusterless state-space decoder bridged by
  `nstat.extras.decoding.clusterless_bridge`.*

## Spectral estimation

- **Thomson DJ (1982).** Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE* 70:1055–1096.
  [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433).
  *The original multitaper (Slepian/DPSS) spectral estimator implemented by
  `SignalObj.MTMspectrum` / `spectrogram`.*
- **Mitra PP, Pesaran B (1999).** Analysis of dynamic brain imaging data.
  *Biophysical Journal* 76:691–708.
  [PMID 9929474](https://pubmed.ncbi.nlm.nih.gov/9929474/) ·
  [doi:10.1016/S0006-3495(99)77236-X](https://doi.org/10.1016/S0006-3495(99)77236-X).
  *Brought multitaper methods to neuroscience; the practical reference for
  spectrograms of LFP/EEG.*

## The toolbox

- **Cajigas I, Malik WQ, Brown EN (2012).** nSTAT: Open-source neural spike
  train analysis toolbox for Matlab. *Journal of Neuroscience Methods*
  211:245–264.
  [PMID 22981419](https://pubmed.ncbi.nlm.nih.gov/22981419/) ·
  [doi:10.1016/j.jneumeth.2012.08.009](https://doi.org/10.1016/j.jneumeth.2012.08.009).
  *The toolbox paper. Cite this if you use nSTAT in your work.*
- **Daley DJ, Vere-Jones D (2003).** *An Introduction to the Theory of Point
  Processes, Vol. I: Elementary Theory and Methods* (2nd ed.). Springer.
  [doi:10.1007/b97277](https://doi.org/10.1007/b97277).
  *The mathematical reference for point processes and conditional
  intensity functions.*
