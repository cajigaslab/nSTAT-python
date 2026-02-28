# Paper Overview and Equation Mapping

Primary reference:

- Cajigas I, Malik WQ, Brown EN. nSTAT: Open-source neural spike train analysis toolbox for Matlab.
  *Journal of Neuroscience Methods* 211:245-264 (2012).
- DOI: `10.1016/j.jneumeth.2012.08.009`
- PMID: `22981419`
- PubMed: <https://pubmed.ncbi.nlm.nih.gov/22981419/>
- Full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3491120/>

## Equation/Section mapping

| Python module/class | Mathematical role | Paper mapping |
|---|---|---|
| `nstat.cif.CIFModel` | Conditional intensity models | Sec. 2, Sec. 3 examples |
| `nstat.analysis.Analysis` | GLM fitting and model estimation | Sec. 2.2, Sec. 3 |
| `nstat.history.HistoryBasis` | Spike-history effects | Sec. 2 methodology |
| `nstat.decoding.DecodingAlgorithms` | Decoding and confidence comparisons | Sec. 3.4 workflows |
| `nstat.fit.FitResult`/`FitSummary` | AIC/BIC/log-likelihood summaries | Sec. 2 model assessment |
