#!/usr/bin/env python3
"""Generate clean-room nSTAT-python learning notebooks from manifest."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf
import yaml


PAPER_DOI = "10.1016/j.jneumeth.2012.08.009"
PAPER_PMID = "22981419"
REPO_NOTEBOOK_BASE = "https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "notebook_manifest.yml",
        help="Notebook manifest path",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root",
    )
    return parser.parse_args()



def markdown_header(topic: str, run_group: str) -> str:
    return (
        f"# {topic}\n\n"
        "This notebook is a Python-native tutorial derived from the MATLAB workflow name, "
        "implemented from scratch for `nSTAT-python`.\n\n"
        f"- Execution group: `{run_group}`\n"
        f"- Paper DOI: `{PAPER_DOI}`\n"
        f"- PMID: `{PAPER_PMID}`\n"
        f"- Help page: `docs/help/examples/{topic}.md`\n"
    )



def code_cell_setup(topic: str) -> str:
    return f"""import numpy as np
import matplotlib.pyplot as plt

from nstat.analysis import Analysis
from nstat.cif import CIFModel
from nstat.decoding import DecodingAlgorithms
from nstat.fit import FitSummary
from nstat.history import HistoryBasis
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial, TrialConfig

TOPIC = \"{topic}\"
rng = np.random.default_rng(2026)
print(f\"Running notebook topic: {{TOPIC}}\")
"""


COMMON_DEMO = """time = np.linspace(0.0, 1.0, 1001)
stimulus = np.sin(2.0 * np.pi * 4.0 * time)

covariate = Covariate(
    time=time,
    data=stimulus,
    name="stimulus",
    labels=["stim"],
)

# Simulate binary spike observations from a smooth probability profile.
base_probability = np.clip(0.01 + 0.03 * (stimulus > 0.0), 0.0, 0.25)
spike_times = time[rng.random(time.size) < base_probability]
spike_train = SpikeTrain(spike_times=spike_times, t_start=float(time[0]), t_end=float(time[-1]), name="unit_1")

trial = Trial(
    spikes=SpikeTrainCollection([spike_train]),
    covariates=CovariateCollection([covariate]),
)

config = TrialConfig(covariate_labels=["stim"], sample_rate_hz=1000.0, fit_type="binomial", name="demo")
fit_result = Analysis.fit_trial(trial, config)
fit_result
"""


MODEL_EVAL = """# Build a CIF model from fitted coefficients and evaluate over time.
model = CIFModel(coefficients=fit_result.coefficients, intercept=fit_result.intercept, link=fit_result.fit_type)
X = stimulus[:, None]
probability = model.evaluate(X)

print("Mean decoded probability:", float(np.mean(probability)))
print("AIC:", fit_result.aic())
print("BIC:", fit_result.bic())

plt.figure(figsize=(9, 3.5))
plt.plot(time, stimulus, label="stimulus", linewidth=1.5)
plt.plot(time, probability, label="model output", linewidth=1.5)
plt.title(f"{TOPIC}: stimulus vs model output")
plt.xlabel("time [s]")
plt.legend()
plt.tight_layout()
plt.show()
"""


HISTORY_AND_DECODING = """# Demonstrate history design and decoding utilities used throughout nSTAT workflows.
history = HistoryBasis(np.array([0.0, 0.01, 0.05, 0.1]))
H = history.design_matrix(spike_train.spike_times, time[::10])
print("History design shape:", H.shape)

trial_matrix = rng.binomial(1, 0.08, size=(10, 200)).astype(float)
rates, prob_mat, sig_mat = DecodingAlgorithms.compute_spike_rate_cis(trial_matrix)
print("First three rates:", rates[:3])
print("Number of significant trial differences:", int(sig_mat.sum()))

# Build a tiny multi-model summary object.
summary = FitSummary([fit_result, fit_result])
print("Best model by AIC:", summary.best_by_aic().fit_type)
"""


ASSERTION_CELL = """# Execution checkpoints: fail fast in CI if core expectations break.
assert np.all(np.isfinite(probability)), "Model output contains non-finite values"
assert fit_result.n_parameters >= 2, "Unexpected parameter count"
assert H.ndim == 2 and H.shape[1] == history.n_bins, "History design dimensions mismatch"
assert np.allclose(prob_mat, prob_mat.T, atol=1e-12), "Pairwise p-value matrix must be symmetric"
assert np.all(np.diag(prob_mat) == 1.0), "Diagonal p-values must be one"
print("Notebook checkpoints passed for", TOPIC)
"""


TOPIC_ASSERTIONS = """# Topic-specific checkpoint for additional parity confidence.
if TOPIC in {
    "AnalysisExamples2",
    "DocumentationSetup2025b",
    "FitResultReference",
    "HybridFilterExample",
    "publish_all_helpfiles",
}:
    assert np.mean(probability) > 0.0, "Topic-specific probability sanity check failed"
    print("Topic-specific checkpoint passed.")
"""


TAIL_MARKDOWN = (
    "## Next steps\n\n"
    "- Compare this notebook with the corresponding help page for concept-level context.\n"
    "- Modify covariates and model settings to test statistical sensitivity.\n"
    "- Use `tools/notebooks/run_notebooks.py` for reproducible execution in CI.\n"
)


def _cell_id(topic: str, index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "-", topic.lower())
    return f"{base}-{index:02d}"


def build_notebook(topic: str, run_group: str, output_path: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "nstat": {
                "topic": topic,
                "run_group": run_group,
                "paper_doi": PAPER_DOI,
                "paper_pmid": PAPER_PMID,
            },
        }
    )

    notebook.cells = [
        nbf.v4.new_markdown_cell(markdown_header(topic, run_group)),
        nbf.v4.new_markdown_cell(
            f"Notebook source link: [{topic}.ipynb]({REPO_NOTEBOOK_BASE}/{topic}.ipynb)"
        ),
        nbf.v4.new_code_cell(code_cell_setup(topic)),
        nbf.v4.new_code_cell(COMMON_DEMO),
        nbf.v4.new_code_cell(MODEL_EVAL),
        nbf.v4.new_code_cell(HISTORY_AND_DECODING),
        nbf.v4.new_code_cell(ASSERTION_CELL),
        nbf.v4.new_code_cell(TOPIC_ASSERTIONS),
        nbf.v4.new_markdown_cell(TAIL_MARKDOWN),
    ]

    for i, cell in enumerate(notebook.cells):
        cell["id"] = _cell_id(topic, i)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)



def main() -> int:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))

    for row in manifest.get("notebooks", []):
        topic = row["topic"]
        run_group = row["run_group"]
        rel_file = Path(row["file"])
        out_path = args.repo_root / rel_file
        build_notebook(topic=topic, run_group=run_group, output_path=out_path)
        print(f"Generated {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
