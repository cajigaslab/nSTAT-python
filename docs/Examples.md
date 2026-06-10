# Example Index

A visual gallery of the runnable nSTAT examples, mirroring the MATLAB `Examples` help index. Start with the five **paper examples** (each a self-contained script with a generated figure gallery), then explore the workflow tutorial notebooks. See also the [Paper-Aligned Toolbox Map](PaperOverview.md).

## Paper examples

The five canonical examples from Cajigas, Malik & Brown (2012), each reproduced as a standalone Python script. Click a card for its full figure gallery; the [Paper Examples](paper_examples.md) page lists every script, run command, and output.

<div class="ex-grid">
<a class="ex-card" href="figures/example01/README.html">
  <img class="ex-thumb" src="_static/gallery/example01.png" alt="mEPSC Poisson models" loading="lazy">
  <span class="ex-body">
    <span class="ex-eyebrow">Example 01</span>
    <span class="ex-title">mEPSC Poisson models</span>
    <span class="ex-desc">Do mEPSCs follow constant vs piecewise Poisson firing under Mg²⁺ washout?</span>
  </span>
</a>
<a class="ex-card" href="figures/example02/README.html">
  <img class="ex-thumb" src="_static/gallery/example02.png" alt="Whisker-stimulus GLM" loading="lazy">
  <span class="ex-body">
    <span class="ex-eyebrow">Example 02</span>
    <span class="ex-title">Whisker-stimulus GLM</span>
    <span class="ex-desc">Do explicit stimulus and spike history sharpen thalamic GLM fits?</span>
  </span>
</a>
<a class="ex-card" href="figures/example03/README.html">
  <img class="ex-thumb" src="_static/gallery/example03.png" alt="PSTH and SSGLM dynamics" loading="lazy">
  <span class="ex-body">
    <span class="ex-eyebrow">Example 03</span>
    <span class="ex-title">PSTH and SSGLM dynamics</span>
    <span class="ex-desc">How do PSTH and state-space GLM capture within- and across-trial dynamics?</span>
  </span>
</a>
<a class="ex-card" href="figures/example04/README.html">
  <img class="ex-thumb" src="_static/gallery/example04.png" alt="Place-cell receptive fields" loading="lazy">
  <span class="ex-body">
    <span class="ex-eyebrow">Example 04</span>
    <span class="ex-title">Place-cell receptive fields</span>
    <span class="ex-desc">Which basis — Gaussian or Zernike — better fits hippocampal place cells?</span>
  </span>
</a>
<a class="ex-card" href="figures/example05/README.html">
  <img class="ex-thumb" src="_static/gallery/example05.png" alt="PPAF and PPHF decoding" loading="lazy">
  <span class="ex-body">
    <span class="ex-eyebrow">Example 05</span>
    <span class="ex-title">PPAF and PPHF decoding</span>
    <span class="ex-desc">How well do adaptive/hybrid point-process filters decode stimulus and reach state?</span>
  </span>
</a>
</div>

## Workflow tutorials

Executable notebooks that follow their MATLAB help-page counterparts step for step.

### Simulation & example data

<div class="ex-grid">
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PPSimExample.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">PPSimExample</span>
    <span class="ex-desc">Simulate spike trains from a conditional intensity function.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PPThinning.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">PPThinning</span>
    <span class="ex-desc">Generate point-process spike trains by thinning.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/PSTHEstimation.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">PSTHEstimation</span>
    <span class="ex-desc">Estimate a peri-stimulus time histogram.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/ValidationDataSet.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">ValidationDataSet</span>
    <span class="ex-desc">Reproduce the toolbox validation dataset.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/mEPSCAnalysis.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">mEPSCAnalysis</span>
    <span class="ex-desc">Analyze miniature-EPSC event trains.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/ExplicitStimulusWhiskerData.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">ExplicitStimulusWhiskerData</span>
    <span class="ex-desc">Explicit-stimulus GLM on thalamic whisker data.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/HippocampalPlaceCellExample.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">HippocampalPlaceCellExample</span>
    <span class="ex-desc">Place-cell receptive fields from position.</span>
  </span>
</a>
</div>

### Fitting & analysis

<div class="ex-grid">
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/AnalysisExamples.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">AnalysisExamples</span>
    <span class="ex-desc">Fit and assess point-process GLMs.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/AnalysisExamples2.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">AnalysisExamples2</span>
    <span class="ex-desc">Further GLM fitting and model selection.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/ConfidenceIntervalOverview.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">ConfidenceIntervalOverview</span>
    <span class="ex-desc">Work with time-varying confidence intervals.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/TrialExamples.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">TrialExamples</span>
    <span class="ex-desc">Assemble Trial objects from spikes and covariates.</span>
  </span>
</a>
</div>

### Decoding & networks

<div class="ex-grid">
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/DecodingExample.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">DecodingExample</span>
    <span class="ex-desc">Point-process adaptive-filter stimulus decoding.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/DecodingExampleWithHist.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">DecodingExampleWithHist</span>
    <span class="ex-desc">Decoding with spike-history covariates.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/StimulusDecode2D.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">StimulusDecode2D</span>
    <span class="ex-desc">Decode a 2-D stimulus from population spikes.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/HybridFilterExample.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">HybridFilterExample</span>
    <span class="ex-desc">Hybrid discrete/continuous point-process filter.</span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/NetworkTutorial.ipynb">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">NetworkTutorial</span>
    <span class="ex-desc">Functional connectivity via GLM coupling.</span>
  </span>
</a>
</div>

## Object & class references

Per-class scaffolds that mirror the MATLAB help index method-for-method. These are API reference stubs rather than standalone tutorials — see the [Class Definitions](ClassDefinitions.md) and [API Reference](api.rst) for the rendered docstrings.

<p class="ex-refs">
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/SignalObjExamples.ipynb">SignalObjExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/CovariateExamples.ipynb">CovariateExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/CovCollExamples.ipynb">CovCollExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/nSpikeTrainExamples.ipynb">nSpikeTrainExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/nstCollExamples.ipynb">nstCollExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/EventsExamples.ipynb">EventsExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/HistoryExamples.ipynb">HistoryExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/TrialConfigExamples.ipynb">TrialConfigExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/ConfigCollExamples.ipynb">ConfigCollExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/FitResultExamples.ipynb">FitResultExamples</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/FitResultReference.ipynb">FitResultReference</a>
<a href="https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks/FitResSummaryExamples.ipynb">FitResSummaryExamples</a>
</p>

## Supplementary checks

Small README demos that double as quick install and plotting checks.

<div class="ex-grid">
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/examples/readme_examples/example1_multitaper_and_spectrogram.py">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">Multitaper spectrum & spectrogram</span>
    <span class="ex-desc"><code>python examples/readme_examples/example1_multitaper_and_spectrogram.py</code></span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/examples/readme_examples/example2_simulate_cif_spiketrain_10s.py">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">Simulated CIF spike train</span>
    <span class="ex-desc"><code>python examples/readme_examples/example2_simulate_cif_spiketrain_10s.py</code></span>
  </span>
</a>
<a class="ex-card ex-mini" href="https://github.com/cajigaslab/nSTAT-python/blob/main/examples/readme_examples/example3_nstcoll_raster_from_example2.py">
  <svg class="ex-glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h3l2-7 3 14 3-10 2 4h4"/></svg>
  <span class="ex-body">
    <span class="ex-title">Spike-train raster</span>
    <span class="ex-desc"><code>python examples/readme_examples/example3_nstcoll_raster_from_example2.py</code></span>
  </span>
</a>
</div>
