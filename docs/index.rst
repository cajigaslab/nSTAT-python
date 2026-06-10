nSTAT Python Documentation
==========================

**Neural Spike Train Analysis Toolbox for Python** — a faithful port of the
MATLAB `nSTAT <https://github.com/cajigaslab/nSTAT>`_ toolbox
(`Cajigas, Malik & Brown, 2012 <https://doi.org/10.1016/j.jneumeth.2012.08.009>`_),
plus an opt-in :mod:`nstat.extras` namespace for modern Python-only additions.

.. rst-class:: nstat-hero

   | Point-process GLMs · time-rescaling goodness-of-fit · adaptive & hybrid decoding
   | State-space EM · clusterless decoding · data-interop & validation bridges
   | ``pip install nstat-toolbox``

.. raw:: html

   <div class="nstat-scope" aria-hidden="true">
     <div class="bar">
       <span class="dot live"></span><span class="dot"></span><span class="dot"></span>
       time-rescaling scope
       <span class="ch">CH&nbsp;01 · 1&nbsp;kHz</span>
     </div>
     <svg viewBox="0 0 1080 132" preserveAspectRatio="none">
       <path class="trace-2" d="M0,66 C90,30 180,102 270,66 S450,30 540,66 720,102 810,66 990,30 1080,66"/>
       <path class="trace" d="M0,66 L60,66 L70,20 L80,66 L210,66 L220,108 L230,66 L360,66 L368,34 L376,66 L470,66 L478,96 L486,66 L600,66 L608,18 L616,66 L740,66 L748,100 L756,66 L880,66 L888,40 L896,66 L1010,66 L1018,90 L1026,66 L1080,66"/>
     </svg>
     <div class="raster">
       <i style="animation-delay:0s"></i><i style="animation-delay:.18s"></i><i style="animation-delay:.05s"></i><i style="animation-delay:.4s"></i><i style="animation-delay:.22s"></i><i style="animation-delay:.7s"></i><i style="animation-delay:.31s"></i><i style="animation-delay:.9s"></i><i style="animation-delay:.5s"></i><i style="animation-delay:1.2s"></i><i style="animation-delay:.62s"></i><i style="animation-delay:1.5s"></i><i style="animation-delay:.8s"></i><i style="animation-delay:.12s"></i><i style="animation-delay:1.1s"></i><i style="animation-delay:.35s"></i><i style="animation-delay:1.4s"></i><i style="animation-delay:.55s"></i><i style="animation-delay:1.7s"></i><i style="animation-delay:.95s"></i><i style="animation-delay:.27s"></i><i style="animation-delay:1.3s"></i><i style="animation-delay:.6s"></i><i style="animation-delay:1.9s"></i><i style="animation-delay:.45s"></i><i style="animation-delay:1.05s"></i><i style="animation-delay:.78s"></i><i style="animation-delay:.2s"></i><i style="animation-delay:1.6s"></i><i style="animation-delay:.88s"></i>
     </div>
   </div>

New here? Start with the friendly, illustrated
`5-minute intro <intro.html>`_ — runnable snippets, the ``nstat.extras``
bridges, and the paper-example gallery.

Quickstart
----------

.. code-block:: bash

   pip install nstat-toolbox
   nstat-install --download-example-data always   # ~150 MB figshare dataset

.. code-block:: python

   import numpy as np
   from nstat import nspikeTrain

   times = np.sort(np.random.default_rng(0).uniform(0, 1, 100))
   st = nspikeTrain(times, name="neuron1", sampleRate=1000,
                    minTime=0.0, maxTime=1.0)
   print(f"{st.numSpikes} spikes")

What's inside
-------------

- **Core** (:mod:`nstat`) — the MATLAB-faithful object model:
  :class:`~nstat.SignalObj`, :class:`~nstat.nspikeTrain`,
  :class:`~nstat.Trial`, :class:`~nstat.Analysis`,
  :class:`~nstat.FitResult`, :class:`~nstat.DecodingAlgorithms`, and the
  multivariate population goodness-of-fit
  :func:`~nstat.population_time_rescale`.
- **Extras** (:mod:`nstat.extras`) — opt-in bridges with no MATLAB
  counterpart: state-space EM (Dynamax), clusterless decoding,
  cross-validation oracles (NeMoS / pykalman / statsmodels), spike-train
  metrics (PySpike), and data interop (Neo / pynapple / pynwb).
- **Five paper examples** — the canonical Cajigas 2012 analyses, each
  runnable in under a minute on the example dataset.

Find your way around
---------------------

- `Friendly 5-minute intro <intro.html>`_ — the best starting point.
- :doc:`Concepts & Background <concepts/index>` — learn the neuroscience and
  statistics behind the toolbox (microelectrode recordings, spikes & the LFP,
  point-process GLMs, goodness-of-fit, decoding), with worked snippets and
  cited literature.
- :doc:`PaperOverview` — crosswalk between the 2012 paper and the Python API.
- :doc:`ClassDefinitions` — method catalog for every core class.
- :doc:`api` — full API reference (auto-generated from docstrings).
- :doc:`extras` — narrative guides for each ``nstat.extras`` bridge, plus
  the `visual summary <extras_summary.html>`_.
- :doc:`paper_examples` — the figure gallery.
- `What's New <whats_new.html>`_ — per-release change summaries.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Concepts & Background

   concepts/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   NeuralSpikeAnalysis_top
   PaperOverview
   ClassDefinitions
   Examples
   DocumentationSetup
   api
   extras
   data_installation
   paper_examples

Citing nSTAT
------------

If you use nSTAT in your work, please cite:

   Cajigas I, Malik WQ, Brown EN. *nSTAT: Open-source neural spike train
   analysis toolbox for Matlab.* Journal of Neuroscience Methods 211:
   245–264, Nov. 2012. `doi:10.1016/j.jneumeth.2012.08.009
   <https://doi.org/10.1016/j.jneumeth.2012.08.009>`_

nSTAT is distributed under the GPL-2.0 license.
