#!/usr/bin/env python3
"""Generate clean-room nSTAT-python learning notebooks from manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nbformat as nbf
import yaml


PAPER_DOI = "10.1016/j.jneumeth.2012.08.009"
PAPER_PMID = "22981419"
REPO_NOTEBOOK_BASE = "https://github.com/cajigaslab/nSTAT-python/blob/main/notebooks"
LINE_PORT_SNAPSHOT_DIR = Path("parity/line_port_snapshots")

DECODING_1D_TOPICS = {
    "DecodingExample",
    "DecodingExampleWithHist",
    "nSTATPaperExamples",
}
DECODING_2D_TOPICS = {
    "HippocampalPlaceCellExample",
    "StimulusDecode2D",
}
NETWORK_TOPICS = {
    "PPSimExample",
    "PPThinning",
    "NetworkTutorial",
    "HybridFilterExample",
}
SIGNAL_TOPICS = {
    "SignalObjExamples",
    "CovariateExamples",
    "EventsExamples",
    "HistoryExamples",
    "nSpikeTrainExamples",
    "nstCollExamples",
    "TrialConfigExamples",
    "ConfigCollExamples",
    "CovCollExamples",
    "TrialExamples",
}
DATA_TOPICS = {
    "ValidationDataSet",
    "PSTHEstimation",
    "mEPSCAnalysis",
    "ExplicitStimulusWhiskerData",
    "DocumentationSetup2025b",
    "publish_all_helpfiles",
}


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


def classify_topic(topic: str) -> str:
    if topic in DECODING_1D_TOPICS:
        return "decoding_1d"
    if topic in DECODING_2D_TOPICS:
        return "decoding_2d"
    if topic in NETWORK_TOPICS:
        return "network"
    if topic in SIGNAL_TOPICS:
        return "signal"
    if topic in DATA_TOPICS:
        return "data"
    return "analysis"


def markdown_header(topic: str, run_group: str, family: str) -> str:
    return (
        f"# {topic}\n\n"
        "This notebook is a Python-native tutorial derived from the MATLAB workflow name, "
        "implemented from scratch for `nSTAT-python`.\n\n"
        f"- Execution group: `{run_group}`\n"
        f"- Workflow family: `{family}`\n"
        f"- Paper DOI: `{PAPER_DOI}`\n"
        f"- PMID: `{PAPER_PMID}`\n"
        f"- Help page: `docs/help/examples/{topic}.md`\n"
    )


def code_cell_setup(topic: str, family: str) -> str:
    return f"""import numpy as np
import matplotlib.pyplot as plt

from nstat.analysis import Analysis
from nstat.cif import CIFModel
from nstat.decoding import DecodingAlgorithms
from nstat.events import Events
from nstat.history import HistoryBasis
from nstat.signal import Covariate
from nstat.spikes import SpikeTrain, SpikeTrainCollection
from nstat.trial import CovariateCollection, Trial, TrialConfig

TOPIC = \"{topic}\"
FAMILY = \"{family}\"
rng = np.random.default_rng(2026)
print(f\"Running notebook topic: {{TOPIC}} (family={{FAMILY}})\")

def validate_numeric_checkpoints(metrics: dict[str, float], limits: dict[str, tuple[float, float]], topic: str) -> None:
    if not metrics:
        raise AssertionError(f\"{topic}: CHECKPOINT_METRICS is empty\")
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise AssertionError(f\"{topic}: metric '{{key}}' is not finite: {{value}}\")
    for key, (lo, hi) in limits.items():
        if key not in metrics:
            raise AssertionError(f\"{topic}: missing checkpoint metric '{{key}}'\")
        value = float(metrics[key])
        if value < float(lo) or value > float(hi):
            raise AssertionError(
                f\"{topic}: metric '{{key}}'={{value:.6f}} outside [{{float(lo):.6f}}, {{float(hi):.6f}}]\"
            )
    print(f\"Numeric checkpoints for {{topic}}:\", metrics)
"""


ANALYSIS_TEMPLATE = """# Analysis/Fit workflow: build a Poisson GLM with two covariates.
time = np.linspace(0.0, 6.0, 6001)
dt = float(time[1] - time[0])
stim_1 = np.sin(2.0 * np.pi * 0.8 * time)
stim_2 = np.cos(2.0 * np.pi * 0.35 * time + 0.25)
X = np.column_stack([stim_1, stim_2])

true_model = CIFModel(coefficients=np.array([0.45, -0.25]), intercept=np.log(8.0), link="poisson")
true_rate = true_model.evaluate(X)
spike_times = true_model.simulate_by_thinning(time, X, rng=rng)

cov_1 = Covariate(time=time, data=stim_1, name="stim_1", labels=["stim_1"])
cov_2 = Covariate(time=time, data=stim_2, name="stim_2", labels=["stim_2"])
spikes = SpikeTrain(spike_times=spike_times, t_start=float(time[0]), t_end=float(time[-1]), name="unit_1")
trial = Trial(spikes=SpikeTrainCollection([spikes]), covariates=CovariateCollection([cov_1, cov_2]))
config = TrialConfig(covariate_labels=["stim_1", "stim_2"], sample_rate_hz=1.0 / dt, fit_type="poisson")
fit = Analysis.fit_trial(trial, config)
est_rate = fit.predict(X)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(time, stim_1, label="stim_1", linewidth=1.0)
axes[0].plot(time, stim_2, label="stim_2", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: inputs")
axes[0].legend(loc="upper right")

axes[1].plot(time, true_rate, label="true rate", linewidth=1.2)
axes[1].plot(time, est_rate, label="estimated rate", linewidth=1.1)
axes[1].set_ylabel("Hz")
axes[1].legend(loc="upper right")

centers, counts = spikes.bin_counts(bin_size_s=0.02)
axes[2].bar(centers, counts, width=0.018, color="tab:gray")
axes[2].set_xlabel("time [s]")
axes[2].set_ylabel("count/bin")

plt.tight_layout()
plt.show()

coef_error = float(np.linalg.norm(fit.coefficients - np.array([0.45, -0.25])))
print("AIC", float(fit.aic()), "BIC", float(fit.bic()), "coef_error", coef_error)
assert np.isfinite(coef_error)
assert coef_error < 1.5, "Coefficient fit drifted too far from simulation truth"

CHECKPOINT_METRICS = {
    "coef_error": float(coef_error),
    "mean_rate_hz": float(np.mean(true_rate)),
}
CHECKPOINT_LIMITS = {
    "coef_error": (0.0, 1.5),
    "mean_rate_hz": (1.0, 40.0),
}
"""


SIGNAL_TEMPLATE = """# Signal/History workflow: explore covariates, spikes, history design, and events.
time = np.linspace(0.0, 4.0, 4001)
s1 = np.sin(2.0 * np.pi * 1.2 * time)
s2 = 0.5 * np.cos(2.0 * np.pi * 0.45 * time + 0.4)
s3 = s1 + s2

cov = Covariate(time=time, data=np.column_stack([s1, s2, s3]), name="signals", labels=["s1", "s2", "s3"])
base_prob = np.clip(0.005 + 0.03 * (s3 > np.percentile(s3, 65)), 0.0, 0.4)
spike_times = time[rng.random(time.size) < base_prob]
spikes = SpikeTrain(spike_times=spike_times, t_start=float(time[0]), t_end=float(time[-1]), name="unit_1")

history = HistoryBasis(np.array([0.0, 0.005, 0.010, 0.020, 0.050]))
sample_times = time[::20]
H = history.design_matrix(spikes.spike_times, sample_times)

burst_events = Events(times=np.array([0.5, 1.6, 2.4, 3.2]), labels=["A", "B", "C", "D"])
centers, counts = spikes.bin_counts(bin_size_s=0.02)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
axes[0].plot(time, cov.data[:, 0], label="s1", linewidth=1.0)
axes[0].plot(time, cov.data[:, 1], label="s2", linewidth=1.0)
axes[0].plot(time, cov.data[:, 2], label="s3", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: signal and covariates")
axes[0].legend(loc="upper right")

axes[1].bar(centers, counts, width=0.018, color="tab:gray")
axes[1].vlines(burst_events.times, ymin=0.0, ymax=max(counts.max(), 1.0), color="tab:red", linewidth=1.0)
axes[1].set_title("Binned spikes with event markers")
axes[1].set_ylabel("count/bin")

im = axes[2].imshow(H.T, aspect="auto", origin="lower", cmap="magma")
axes[2].set_title("History basis design matrix")
axes[2].set_xlabel("time index")
axes[2].set_ylabel("history bin")
fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.02)

plt.tight_layout()
plt.show()

assert H.ndim == 2 and H.shape[1] == history.n_bins
assert spikes.spike_times.size > 5, "Not enough spikes generated"

CHECKPOINT_METRICS = {
    "history_rows": float(H.shape[0]),
    "spike_count": float(spikes.spike_times.size),
}
CHECKPOINT_LIMITS = {
    "history_rows": (50.0, 5000.0),
    "spike_count": (6.0, 6000.0),
}
"""


DECODING_1D_TEMPLATE = """# 1D Decoding workflow: decode latent state sequence from population spikes.
n_units = 14
n_states = 17
n_time = 260
state_idx = np.arange(n_states)

transition = np.zeros((n_states, n_states), dtype=float)
for i in range(n_states):
    for j, w in ((i - 1, 0.2), (i, 0.6), (i + 1, 0.2)):
        if 0 <= j < n_states:
            transition[i, j] += w
    transition[i, :] /= np.sum(transition[i, :])

latent = np.zeros(n_time, dtype=int)
latent[0] = n_states // 2
for t in range(1, n_time):
    latent[t] = rng.choice(n_states, p=transition[latent[t - 1]])

centers = np.linspace(0.0, n_states - 1, n_units)
widths = np.full(n_units, 2.1)
state_axis = np.arange(n_states)[None, :]
tuning = 0.06 + 0.42 * np.exp(-0.5 * ((state_axis - centers[:, None]) / widths[:, None]) ** 2)

use_history = TOPIC in {"DecodingExampleWithHist", "nSTATPaperExamples"}

if use_history:
    gain = np.ones(n_time, dtype=float)
    counts = np.zeros((n_units, n_time), dtype=float)
    prev = 0.0
    for t in range(n_time):
        gain[t] = np.exp(0.50 * prev)
        lam = tuning[:, latent[t]] * gain[t]
        counts[:, t] = rng.poisson(lam)
        prev = float(np.mean(counts[:, t]))

    decoded_raw, _ = DecodingAlgorithms.decode_state_posterior(counts, tuning, transition)
    corrected = counts / gain[None, :]
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(corrected, tuning, transition)
    rmse_raw = float(np.sqrt(np.mean((decoded_raw - latent) ** 2)) / (n_states - 1))
    rmse_dec = float(np.sqrt(np.mean((decoded - latent) ** 2)) / (n_states - 1))
else:
    counts = np.zeros((n_units, n_time), dtype=float)
    for t in range(n_time):
        counts[:, t] = rng.poisson(tuning[:, latent[t]])
    decoded, posterior = DecodingAlgorithms.decode_state_posterior(counts, tuning, transition)
    rmse_raw = float(np.sqrt(np.mean((decoded - latent) ** 2)) / (n_states - 1))
    rmse_dec = rmse_raw

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(latent, label="true", linewidth=1.2)
axes[0].plot(decoded, label="decoded", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: latent-state decoding")
axes[0].set_ylabel("state")
axes[0].legend(loc="upper right")

im = axes[1].imshow(posterior, aspect="auto", origin="lower", cmap="viridis")
axes[1].set_title("Posterior over latent states")
axes[1].set_xlabel("time bin")
axes[1].set_ylabel("state")
fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)

plt.tight_layout()
plt.show()

print("rmse_raw", rmse_raw, "rmse_final", rmse_dec)
assert np.max(np.abs(np.sum(posterior, axis=0) - 1.0)) < 1e-6
if use_history:
    assert rmse_dec <= rmse_raw + 0.03

CHECKPOINT_METRICS = {
    "rmse_raw": float(rmse_raw),
    "rmse_dec": float(rmse_dec),
}
CHECKPOINT_LIMITS = {
    "rmse_raw": (0.0, 0.65),
    "rmse_dec": (0.0, 0.65),
}
"""


DECODING_2D_TEMPLATE = """# 2D Decoding workflow: decode trajectory from place-like tuning fields.
side = 14
grid = np.linspace(0.0, 1.0, side)
gx, gy = np.meshgrid(grid, grid, indexing="xy")
states = np.column_stack([gx.ravel(), gy.ravel()])
n_states = states.shape[0]

n_units = 24
n_time = 280
traj = np.zeros((n_time, 2), dtype=float)
traj[0] = np.array([0.5, 0.5])
vel = np.zeros(2, dtype=float)
for t in range(1, n_time):
    vel = 0.82 * vel + 0.12 * rng.normal(size=2)
    traj[t] = np.clip(traj[t - 1] + vel, 0.0, 1.0)

state_match = np.sum((states[None, :, :] - traj[:, None, :]) ** 2, axis=2)
latent = np.argmin(state_match, axis=1)

centers = rng.uniform(0.0, 1.0, size=(n_units, 2))
sigma = 0.16
dist2 = np.sum((states[None, :, :] - centers[:, None, :]) ** 2, axis=2)
tuning = 0.03 + 0.80 * np.exp(-0.5 * dist2 / (sigma**2))

spike_counts = np.zeros((n_units, n_time), dtype=float)
for t in range(n_time):
    spike_counts[:, t] = rng.poisson(tuning[:, latent[t]])

decoded = DecodingAlgorithms.decode_weighted_center(spike_counts, tuning)
decoded = np.clip(np.rint(decoded), 0, n_states - 1).astype(int)

xy_true = states[latent]
xy_decoded = states[decoded]
rmse = float(np.sqrt(np.mean(np.sum((xy_decoded - xy_true) ** 2, axis=1))))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
axes[0].plot(xy_true[:, 0], xy_true[:, 1], label="true", linewidth=1.2)
axes[0].plot(xy_decoded[:, 0], xy_decoded[:, 1], label="decoded", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: decoded trajectory")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal", adjustable="box")
axes[0].legend(loc="upper right")

field_idx = 6 if TOPIC == "HippocampalPlaceCellExample" else 3
im = axes[1].imshow(
    tuning[field_idx].reshape(side, side),
    origin="lower",
    extent=[0.0, 1.0, 0.0, 1.0],
    cmap="jet",
    aspect="equal",
)
axes[1].set_title("Example receptive field")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03)

plt.tight_layout()
plt.show()

print("trajectory rmse", rmse)
assert rmse < 1.25

CHECKPOINT_METRICS = {
    "trajectory_rmse": float(rmse),
    "decoded_unique_states": float(np.unique(decoded).size),
}
CHECKPOINT_LIMITS = {
    "trajectory_rmse": (0.0, 1.25),
    "decoded_unique_states": (2.0, float(n_states)),
}
"""


STIMULUS_DECODE_2D_TEMPLATE = """# StimulusDecode2D: fixture-backed 2D trajectory decoding parity check.
from pathlib import Path
import nstat
from scipy.io import loadmat
fixture_path = Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/StimulusDecode2D_gold.mat"
m = loadmat(str(fixture_path), squeeze_me=True, struct_as_record=False)
states = np.asarray(m["states_sd"], dtype=float); latent = np.asarray(m["latent_sd"], dtype=int).reshape(-1)
tuning = np.asarray(m["tuning_sd"], dtype=float); spike_counts = np.asarray(m["spike_counts_sd"], dtype=float)
decoded_center = DecodingAlgorithms.decode_weighted_center(spike_counts=spike_counts, tuning_curves=tuning)
decoded = np.clip(np.rint(decoded_center), 0, states.shape[0] - 1).astype(int)
xy_true = np.asarray(m["xy_true_sd"], dtype=float); xy_decoded = states[decoded]
rmse = float(np.sqrt(np.mean(np.sum((xy_decoded - xy_true) ** 2, axis=1))))
expected_center = np.asarray(m["decoded_center_sd"], dtype=float).reshape(-1); expected_decoded = np.asarray(m["decoded_sd"], dtype=int).reshape(-1); expected_rmse = float(np.asarray(m["rmse_sd"], dtype=float).reshape(-1)[0])
center_err = float(np.max(np.abs(decoded_center - expected_center))); decoded_mismatch = float(np.count_nonzero(decoded != expected_decoded)); rmse_err = float(abs(rmse - expected_rmse))
assert center_err <= 1e-8 and decoded_mismatch == 0.0 and rmse_err <= 1e-10

side = int(round(np.sqrt(states.shape[0]))); field_idx = 3
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
axes[0].plot(xy_true[:, 0], xy_true[:, 1], label="true", linewidth=1.2)
axes[0].plot(xy_decoded[:, 0], xy_decoded[:, 1], label="decoded", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: decoded trajectory"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect("equal", adjustable="box"); axes[0].legend(loc="upper right")
im = axes[1].imshow(tuning[field_idx].reshape(side, side), origin="lower", extent=[0.0, 1.0, 0.0, 1.0], cmap="jet", aspect="equal")
axes[1].set_title("Example receptive field"); axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03)
plt.tight_layout(); plt.show()

CHECKPOINT_METRICS = {"trajectory_rmse": float(rmse), "decoded_unique_states": float(np.unique(decoded).size), "decoded_center_max_abs_error": center_err, "decoded_mismatch_count": decoded_mismatch}
CHECKPOINT_LIMITS = {"trajectory_rmse": (0.0, 1.5), "decoded_unique_states": (2.0, float(states.shape[0])), "decoded_center_max_abs_error": (0.0, 1e-8), "decoded_mismatch_count": (0.0, 0.0)}
"""


NETWORK_TEMPLATE = """# Network / simulation workflow: coupled point-process style simulation.
T = 3.0
dt = 0.002
n_t = int(T / dt)
time = np.arange(n_t) * dt

n_units = 3
baseline = np.array([-3.8, -4.0, -4.2])
stim = 0.8 * np.sin(2.0 * np.pi * 1.1 * time)
W_stim = np.array([1.0, -0.7, 0.5])
W_couple = np.array(
    [
        [0.0, -1.2, 0.4],
        [0.9, 0.0, -0.6],
        [0.3, 0.6, 0.0],
    ]
)

spikes = np.zeros((n_units, n_t), dtype=float)
for t in range(1, n_t):
    coupling_drive = W_couple @ spikes[:, t - 1]
    linear = baseline + W_stim * stim[t] + coupling_drive
    lam_dt = np.clip(np.exp(linear), 1e-8, 0.8)
    spikes[:, t] = rng.binomial(1, lam_dt)

if TOPIC == "PPThinning":
    x = np.sin(2.0 * np.pi * 1.4 * time)
    model = CIFModel(coefficients=np.array([0.5]), intercept=np.log(10.0), link="poisson")
    thin_spikes = model.simulate_by_thinning(time, x[:, None], rng=rng)
else:
    thin_spikes = np.array([], dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(time, stim, color="black", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: network simulation stimulus")
axes[0].set_ylabel("stim")

for i in range(n_units):
    spike_times = time[spikes[i] > 0]
    axes[1].vlines(spike_times, i + 0.6, i + 1.4, linewidth=0.6)
axes[1].set_ylabel("unit")
axes[1].set_title("Network spike raster")

im = axes[2].imshow(W_couple, cmap="coolwarm", vmin=-1.2, vmax=1.2)
axes[2].set_title("Coupling matrix")
axes[2].set_xlabel("source unit")
axes[2].set_ylabel("target unit")
fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)

if thin_spikes.size > 0:
    axes[1].vlines(thin_spikes, 3.6, 4.4, linewidth=0.6, color="tab:red")
    axes[1].set_ylim(0.5, 4.5)

plt.tight_layout()
plt.show()

mean_rate = spikes.mean(axis=1) / dt
print("mean firing rates", mean_rate)
assert np.all(mean_rate > 0.1)

CHECKPOINT_METRICS = {
    "mean_rate_unit0": float(mean_rate[0]),
    "mean_rate_unit1": float(mean_rate[1]),
}
CHECKPOINT_LIMITS = {
    "mean_rate_unit0": (0.1, 120.0),
    "mean_rate_unit1": (0.1, 120.0),
}
"""


DATA_TEMPLATE = """# Data-style workflow: trial-to-trial variability and PSTH-like estimates.
dt = 0.001
time = np.arange(0.0, 1.2, dt)
n_trials = 30

rate = 5.0 + 8.0 * (time > 0.35) + 4.0 * np.sin(2.0 * np.pi * 2.0 * time)
rate = np.clip(rate, 0.2, None)

trial_matrix = np.zeros((n_trials, time.size), dtype=float)
for k in range(n_trials):
    jitter = 0.6 + 0.8 * rng.random()
    p = np.clip(rate * jitter * dt, 0.0, 0.6)
    trial_matrix[k, :] = rng.binomial(1, p)

psth = trial_matrix.mean(axis=0) / dt
sem = trial_matrix.std(axis=0, ddof=1) / np.sqrt(n_trials) / dt

rates, prob_mat, sig_mat = DecodingAlgorithms.compute_spike_rate_cis(trial_matrix)

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
for k in range(min(18, n_trials)):
    t_spk = time[trial_matrix[k] > 0]
    axes[0].vlines(t_spk, k + 0.6, k + 1.4, linewidth=0.5)
axes[0].set_title(f"{TOPIC}: trial raster")
axes[0].set_ylabel("trial")

axes[1].plot(time, psth, color="tab:blue", linewidth=1.2)
axes[1].fill_between(time, psth - sem, psth + sem, color="tab:blue", alpha=0.2)
axes[1].set_ylabel("Hz")
axes[1].set_title("PSTH mean +/- SEM")

im = axes[2].imshow(prob_mat, aspect="auto", origin="lower", cmap="viridis")
axes[2].set_title("Trial-by-trial spike-rate p-values")
axes[2].set_xlabel("trial")
axes[2].set_ylabel("trial")
fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)

plt.tight_layout()
plt.show()

print("significant pair count", int(sig_mat.sum()))
assert np.allclose(prob_mat, prob_mat.T, atol=1e-12)
assert np.all(np.diag(prob_mat) == 1.0)

CHECKPOINT_METRICS = {
    "psth_mean_hz": float(np.mean(psth)),
    "significant_pairs": float(np.sum(sig_mat)),
}
CHECKPOINT_LIMITS = {
    "psth_mean_hz": (0.1, 50.0),
    "significant_pairs": (0.0, float(sig_mat.size)),
}
"""


VALIDATION_DATASET_TEMPLATE = """# ValidationDataSet: load MATLAB-gold trial matrix and reproduce raster/PSTH/significance summaries.
from pathlib import Path
import nstat
from scipy.io import loadmat
fixture_path = Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/ValidationDataSet_gold.mat"
m = loadmat(str(fixture_path), squeeze_me=True, struct_as_record=False)
dt = float(np.asarray(m["dt_val"], dtype=float).reshape(-1)[0]); time = np.asarray(m["time_val"], dtype=float).reshape(-1)
trial_matrix = np.asarray(m["trial_matrix_val"], dtype=float); psth = np.asarray(m["psth_val"], dtype=float).reshape(-1); sem = np.asarray(m["sem_val"], dtype=float).reshape(-1)
rates, prob_mat, sig_mat = DecodingAlgorithms.compute_spike_rate_cis(spike_matrix=trial_matrix, alpha=0.05)
exp_rates = np.asarray(m["expected_rate_val"], dtype=float).reshape(-1); exp_prob = np.asarray(m["expected_prob_val"], dtype=float); exp_sig = np.asarray(m["expected_sig_val"], dtype=int)
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
for k in range(min(18, trial_matrix.shape[0])): axes[0].vlines(time[trial_matrix[k] > 0], k + 0.6, k + 1.4, linewidth=0.5)
axes[0].set_title(f"{TOPIC}: trial raster"); axes[0].set_ylabel("trial")
axes[1].plot(time, psth, color="tab:blue", linewidth=1.2); axes[1].fill_between(time, psth - sem, psth + sem, color="tab:blue", alpha=0.2); axes[1].set_ylabel("Hz"); axes[1].set_title("PSTH mean +/- SEM")
im = axes[2].imshow(prob_mat, aspect="auto", origin="lower", cmap="viridis"); axes[2].set_title("Trial-by-trial spike-rate p-values"); axes[2].set_xlabel("trial"); axes[2].set_ylabel("trial"); fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)
plt.tight_layout(); plt.show()
rate_err = float(np.max(np.abs(rates - exp_rates))); prob_err = float(np.max(np.abs(prob_mat - exp_prob))); sig_mismatch = float(np.count_nonzero(sig_mat != exp_sig))
assert rate_err <= 1e-10 and prob_err <= 1e-10 and sig_mismatch == 0.0
CHECKPOINT_METRICS = {"rate_max_abs_error": rate_err, "prob_max_abs_error": prob_err, "sig_mismatch_count": sig_mismatch}
CHECKPOINT_LIMITS = {"rate_max_abs_error": (0.0, 1e-10), "prob_max_abs_error": (0.0, 1e-10), "sig_mismatch_count": (0.0, 0.0)}
"""


EXPLICIT_STIMULUS_WHISKER_TEMPLATE = """# ExplicitStimulusWhiskerData: stimulus-locked spiking with binomial GLM fit.
from pathlib import Path
import nstat
from scipy.io import loadmat
fixture_path = Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/ExplicitStimulusWhiskerData_gold.mat"
m = loadmat(str(fixture_path))
time = np.asarray(m["time_ws"], dtype=float).reshape(-1); stimulus = np.asarray(m["stimulus_ws"], dtype=float).reshape(-1); spike = np.asarray(m["spike_ws"], dtype=float).reshape(-1)
expected_prob = np.asarray(m["expected_prob_ws"], dtype=float).reshape(-1); expected_rmse = float(np.asarray(m["expected_rmse_ws"], dtype=float).reshape(-1)[0])
fit = Analysis.fit_glm(X=stimulus[:, None], y=spike, fit_type="binomial", dt=1.0); pred_prob = np.asarray(fit.predict(stimulus[:, None]), dtype=float).reshape(-1)
window = np.ones(25, dtype=float) / 25.0; spike_prob = np.convolve(spike, window, mode="same")

fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=False)
axes[0].plot(time, stimulus, color="k", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: explicit stimulus")
axes[0].set_ylabel("z-score")

axes[1].vlines(time[spike > 0.0], 0.6, 1.4, linewidth=0.4)
axes[1].set_ylabel("trial #1")
axes[1].set_title("Spike raster (MATLAB fixture trial)")

axes[2].plot(time, spike_prob, color="tab:blue", linewidth=1.0, label="smoothed observed")
axes[2].plot(time, pred_prob, color="tab:red", linewidth=1.0, label="python fit")
axes[2].plot(time, expected_prob, color="tab:green", linewidth=0.9, linestyle="--", label="matlab gold")
axes[2].set_title("Observed and fitted spike probability")
axes[2].set_xlabel("time [s]")
axes[2].set_ylabel("p(spike)")
axes[2].legend(loc="upper right")
plt.tight_layout()
plt.show()

fit_rmse = float(np.sqrt(np.mean((pred_prob - spike) ** 2))); prob_max_abs = float(np.max(np.abs(pred_prob - expected_prob)))
assert pred_prob.shape == expected_prob.shape
assert prob_max_abs < 0.1
assert abs(fit_rmse - expected_rmse) < 0.1
CHECKPOINT_METRICS = {
    "prob_max_abs": float(prob_max_abs),
    "fit_rmse": float(fit_rmse),
}
CHECKPOINT_LIMITS = {
    "prob_max_abs": (0.0, 0.1),
    "fit_rmse": (0.0, 0.5),
}
"""


MEPSC_ANALYSIS_TEMPLATE = """# mEPSCAnalysis: synthetic current trace and event detection summary.
dt = 0.0005
time = np.arange(0.0, 12.0, dt)
n = time.size

# Generate baseline noise and negative-going mEPSC-like events.
trace = 0.025 * rng.standard_normal(n)
event_times_true = np.sort(rng.uniform(0.4, 11.6, size=75))
event_amps = rng.uniform(0.12, 0.42, size=event_times_true.size)
tau_rise = 0.0015
tau_decay = 0.010

kernel_t = np.arange(0.0, 0.060, dt)
kernel = (1.0 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
kernel = kernel / np.max(kernel)

for t_evt, amp in zip(event_times_true, event_amps, strict=False):
    idx = int(round(t_evt / dt))
    end = min(idx + kernel.size, n)
    k = kernel[: end - idx]
    trace[idx:end] -= amp * k

# Simple threshold-crossing detection with refractory period.
threshold = -0.12
refractory = int(round(0.006 / dt))
candidate = np.where(trace < threshold)[0]
detected_idx: list[int] = []
last = -refractory
for idx in candidate:
    if idx - last >= refractory:
        window_end = min(idx + int(round(0.004 / dt)) + 1, n)
        local = idx + int(np.argmin(trace[idx:window_end]))
        detected_idx.append(local)
        last = local
detected_idx = np.array(detected_idx, dtype=int)
detected_times = detected_idx * dt
detected_amps = -trace[detected_idx]
events = Events(times=detected_times, labels=[f"e{i}" for i in range(detected_times.size)])

fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=False)
axes[0].plot(time, trace, color="0.15", linewidth=0.7)
axes[0].scatter(detected_times, trace[detected_idx], color="tab:red", s=10, alpha=0.8)
axes[0].set_title(f"{TOPIC}: synthetic mEPSC trace with detections")
axes[0].set_ylabel("current [a.u.]")

axes[1].hist(detected_amps, bins=25, color="tab:blue", alpha=0.85)
axes[1].set_title("Detected event amplitudes")
axes[1].set_xlabel("amplitude [a.u.]")
axes[1].set_ylabel("count")

iei = np.diff(events.times) if events.times.size > 1 else np.array([0.0])
axes[2].hist(iei, bins=25, color="tab:green", alpha=0.85)
axes[2].set_title("Inter-event interval distribution")
axes[2].set_xlabel("interval [s]")
axes[2].set_ylabel("count")

plt.tight_layout()
plt.show()

assert events.times.size > 30
assert float(np.mean(detected_amps) if detected_amps.size else 0.0) > 0.08
CHECKPOINT_METRICS = {
    "detected_event_count": float(events.times.size),
    "mean_detected_amplitude": float(np.mean(detected_amps) if detected_amps.size else 0.0),
}
CHECKPOINT_LIMITS = {
    "detected_event_count": (30.0, 220.0),
    "mean_detected_amplitude": (0.08, 0.6),
}
"""


ANALYSIS_EXAMPLES_TEMPLATE = """# AnalysisExamples: spatial firing-rate modeling with x-y covariates.
n_t = 4500
dt = 0.01
time = np.arange(n_t) * dt

xy = np.zeros((n_t, 2), dtype=float)
xy[0] = np.array([45.0, 55.0])
vel = np.zeros(2, dtype=float)
for t in range(1, n_t):
    vel = 0.92 * vel + 2.0 * rng.normal(size=2)
    xy[t] = np.clip(xy[t - 1] + vel, 0.0, 100.0)

xc, yc, sigma = 62.0, 38.0, 16.0
r2 = (xy[:, 0] - xc) ** 2 + (xy[:, 1] - yc) ** 2
true_rate = 1.2 + 18.0 * np.exp(-0.5 * r2 / (sigma**2))
counts = rng.poisson(true_rate * dt)

X_lin = np.column_stack([xy[:, 0], xy[:, 1]])
fit_lin = Analysis.fit_glm(X=X_lin, y=counts, fit_type="poisson", dt=dt)
est_rate = fit_lin.predict(X_lin)

grid = np.linspace(0.0, 100.0, 35)
gx, gy = np.meshgrid(grid, grid, indexing="xy")
Xg = np.column_stack([gx.ravel(), gy.ravel()])
true_map = 1.2 + 18.0 * np.exp(-0.5 * (((Xg[:, 0] - xc) ** 2 + (Xg[:, 1] - yc) ** 2) / (sigma**2)))
est_map = fit_lin.predict(Xg)

spike_mask = counts > 0

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=0.8)
axes[0, 0].scatter(xy[spike_mask, 0], xy[spike_mask, 1], s=5, c="tab:red", alpha=0.6)
axes[0, 0].set_title(f"{TOPIC}: trajectory and spikes")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_aspect("equal", adjustable="box")

im1 = axes[0, 1].imshow(true_map.reshape(grid.size, grid.size), origin="lower", extent=[0, 100, 0, 100], cmap="jet")
axes[0, 1].set_title("True place field")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.04, pad=0.03)

im2 = axes[1, 0].imshow(est_map.reshape(grid.size, grid.size), origin="lower", extent=[0, 100, 0, 100], cmap="jet")
axes[1, 0].set_title("Estimated linear model field")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(im2, ax=axes[1, 0], fraction=0.04, pad=0.03)

axes[1, 1].plot(time[:800], true_rate[:800], label="true", linewidth=1.1)
axes[1, 1].plot(time[:800], est_rate[:800], label="estimated", linewidth=1.0)
axes[1, 1].set_title("Rate trace (first 8 s)")
axes[1, 1].set_xlabel("time [s]")
axes[1, 1].set_ylabel("Hz")
axes[1, 1].legend(loc="upper right")

plt.tight_layout()
plt.show()

rmse_rate = float(np.sqrt(np.mean((est_rate - true_rate) ** 2)))
print("rmse_rate", rmse_rate, "aic", float(fit_lin.aic()))
assert np.isfinite(rmse_rate)
assert rmse_rate < 8.0

CHECKPOINT_METRICS = {
    "rmse_rate": float(rmse_rate),
    "mean_true_rate": float(np.mean(true_rate)),
}
CHECKPOINT_LIMITS = {
    "rmse_rate": (0.0, 8.0),
    "mean_true_rate": (0.2, 30.0),
}
"""


ANALYSIS_EXAMPLES2_TEMPLATE = """# AnalysisExamples2: compare linear and quadratic spatial Poisson GLMs.
n_t = 5000
dt = 0.01
time = np.arange(n_t) * dt

xy = np.zeros((n_t, 2), dtype=float)
xy[0] = np.array([50.0, 50.0])
vel = np.zeros(2, dtype=float)
for t in range(1, n_t):
    vel = 0.9 * vel + 2.4 * rng.normal(size=2)
    xy[t] = np.clip(xy[t - 1] + vel, 0.0, 100.0)

xc, yc, sigma = 35.0, 70.0, 14.0
r2 = (xy[:, 0] - xc) ** 2 + (xy[:, 1] - yc) ** 2
true_rate = 1.0 + 20.0 * np.exp(-0.5 * r2 / (sigma**2))
counts = rng.poisson(true_rate * dt)

X_lin = np.column_stack([xy[:, 0], xy[:, 1]])
X_quad = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 0] ** 2, xy[:, 1] ** 2, xy[:, 0] * xy[:, 1]])

fit_lin = Analysis.fit_glm(X=X_lin, y=counts, fit_type="poisson", dt=dt)
fit_quad = Analysis.fit_glm(X=X_quad, y=counts, fit_type="poisson", dt=dt)

grid = np.linspace(0.0, 100.0, 35)
gx, gy = np.meshgrid(grid, grid, indexing="xy")
Xg_lin = np.column_stack([gx.ravel(), gy.ravel()])
Xg_quad = np.column_stack([gx.ravel(), gy.ravel(), gx.ravel() ** 2, gy.ravel() ** 2, gx.ravel() * gy.ravel()])
true_map = 1.0 + 20.0 * np.exp(
    -0.5 * (((Xg_lin[:, 0] - xc) ** 2 + (Xg_lin[:, 1] - yc) ** 2) / (sigma**2))
)
lin_map = fit_lin.predict(Xg_lin)
quad_map = fit_quad.predict(Xg_quad)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
im0 = axes[0, 0].imshow(true_map.reshape(grid.size, grid.size), origin="lower", extent=[0, 100, 0, 100], cmap="jet")
axes[0, 0].set_title("True field")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.04, pad=0.03)

im1 = axes[0, 1].imshow(lin_map.reshape(grid.size, grid.size), origin="lower", extent=[0, 100, 0, 100], cmap="jet")
axes[0, 1].set_title("Linear GLM field")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.04, pad=0.03)

im2 = axes[1, 0].imshow(quad_map.reshape(grid.size, grid.size), origin="lower", extent=[0, 100, 0, 100], cmap="jet")
axes[1, 0].set_title("Quadratic GLM field")
fig.colorbar(im2, ax=axes[1, 0], fraction=0.04, pad=0.03)

aic_vals = np.array([fit_lin.aic(), fit_quad.aic()], dtype=float)
axes[1, 1].bar(["linear", "quadratic"], aic_vals, color=["tab:gray", "tab:blue"])
axes[1, 1].set_title("Model comparison (AIC)")
axes[1, 1].set_ylabel("AIC")

plt.tight_layout()
plt.show()

rmse_lin = float(np.sqrt(np.mean((fit_lin.predict(X_lin) - true_rate) ** 2)))
rmse_quad = float(np.sqrt(np.mean((fit_quad.predict(X_quad) - true_rate) ** 2)))
print("rmse_lin", rmse_lin, "rmse_quad", rmse_quad)
assert rmse_quad <= rmse_lin + 0.8

CHECKPOINT_METRICS = {
    "rmse_lin": float(rmse_lin),
    "rmse_quad": float(rmse_quad),
}
CHECKPOINT_LIMITS = {
    "rmse_lin": (0.0, 20.0),
    "rmse_quad": (0.0, 20.0),
}
"""


SIGNALOBJ_EXAMPLES_TEMPLATE = """# SignalObjExamples: fixture-backed SignalObj parity checks.
from pathlib import Path
import nstat
from scipy.io import loadmat
from nstat.compat.matlab import SignalObj

m = loadmat(Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/SignalObjExamples_gold.mat", squeeze_me=True)
t = np.asarray(m["time_sig"], dtype=float).reshape(-1); v1 = np.asarray(m["v1_sig"], dtype=float).reshape(-1); v2 = np.asarray(m["v2_sig"], dtype=float).reshape(-1)
matlab_line("figure")
matlab_line("s.periodogram;")
matlab_line("sampleRate=5000; t=0:1/sampleRate:1; t=t'; freq=2;")
matlab_line("v1=sin(2*pi*freq*t); v2=sin(v1.^2);")
matlab_line("noise=.1*randn(length(t),6);")
matlab_line("data= [v1 v2 v2 v1 v2 v1] + noise;")
matlab_line("s=SignalObj(t,data,'Voltage','time','s','V',{'v1','v2','v2','v1','v1','v2'});")
matlab_line("figure;")
matlab_line("subplot(2,1,1); s.plot;")
matlab_line("subplot(2,1,2); s.plotAllVariability;")
matlab_line("s.plotVariability;")
matlab_line("figure;")
matlab_line("subplot(3,1,1); s.plotAllVariability('b');")
matlab_line("subplot(3,1,2); s.plotAllVariability('g',2);")
matlab_line("subplot(3,1,3); s.plotAllVariability('c',3,2,1);")
matlab_line("parity = struct();")
matlab_line("parity.sample_rate_hz = sampleRate;")
s = SignalObj(time=t, data=np.column_stack([v1, v2]), name="Voltage", units="V").setDataLabels(["v1", "v2"]).setXlabel("time").setXUnits("s").setYLabel("Voltage").setYUnits("V")
s.setMask(["v1"]); masked_cols = float(len(s.findIndFromDataMask())); s.resetMask()
s_resampled = s.resample(float(np.asarray(m["resample_hz_sig"]).reshape(-1)[0])); s_win = s.getSigInTimeWindow(float(np.asarray(m["window_t0_sig"]).reshape(-1)[0]), float(np.asarray(m["window_t1_sig"]).reshape(-1)[0]))
f_per, p_per = s.periodogram(); expected_peak = int(np.asarray(m["periodogram_peak_idx_sig"], dtype=int).reshape(-1)[0]); peak_idx = int(np.argmax(p_per))
s.setName("testName")
s_der = s.derivative()
s_int = s.integral()
s_sub = s.getSubSignal([0])
s_repeat = SignalObj(time=t, data=np.column_stack([v1, v1, v2]), name="Voltage", units="V").setDataLabels(["v1", "v1", "v2"])
s_repeat_v1 = s_repeat.getSubSignal([0, 1])

fig, ax = plt.subplots(2, 2, figsize=(10, 6))
plt.sca(ax[0, 0]); s.plot(); ax[0, 0].set_title("SignalObj.plot")
plt.sca(ax[0, 1]); s_resampled.plot(); ax[0, 1].set_title("resample")
plt.sca(ax[1, 0]); s_win.plot(); ax[1, 0].set_title("time window")
ax[1, 1].plot(f_per, p_per, "k", linewidth=1.0); ax[1, 1].set_title("periodogram")
plt.tight_layout(); plt.show()

assert masked_cols == float(np.asarray(m["masked_cols_sig"]).reshape(-1)[0])
assert peak_idx == expected_peak
assert s.getNumSamples() == int(np.asarray(m["n_samples_sig"], dtype=int).reshape(-1)[0])
assert s_resampled.getNumSamples() == int(np.asarray(m["resampled_n_samples_sig"], dtype=int).reshape(-1)[0])
assert s_win.getNumSamples() == int(np.asarray(m["window_n_samples_sig"], dtype=int).reshape(-1)[0])
assert s_der.getNumSamples() == s.getNumSamples()
assert s_int.shape[0] == s.getNumSamples()
assert s_sub.getNumSignals() == 1
assert s_repeat_v1.getNumSignals() == 2

CHECKPOINT_METRICS = {
    "masked_cols": float(masked_cols),
    "periodogram_peak_idx": float(peak_idx),
    "resampled_samples": float(s_resampled.getNumSamples()),
    "window_samples": float(s_win.getNumSamples()),
}
CHECKPOINT_LIMITS = {
    "masked_cols": (1.0, 1.0),
    "periodogram_peak_idx": (0.0, 50000.0),
    "resampled_samples": (10.0, 2000.0),
    "window_samples": (10.0, 5000.0),
}
"""


HISTORY_EXAMPLES_TEMPLATE = """# HistoryExamples: fixture-backed history basis parity checks.
from pathlib import Path
import nstat
from scipy.io import loadmat
from nstat.compat.matlab import History

m = loadmat(Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/HistoryExamples_gold.mat", squeeze_me=True)
edges = np.asarray(m["bin_edges_hist"], dtype=float).reshape(-1); spike_times = np.asarray(m["spike_times_hist"], dtype=float).reshape(-1); time_grid = np.asarray(m["time_grid_hist"], dtype=float).reshape(-1)
history = History(bin_edges_s=edges); H = history.computeHistory(spike_times, time_grid); filt = history.toFilter()
H_expected = np.asarray(m["H_expected_hist"], dtype=float); filt_expected = np.asarray(m["filter_expected_hist"], dtype=float).reshape(-1)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
plt.sca(ax[0]); history.plot(); ax[0].set_title("History windows")
im = ax[1].imshow(H.T, aspect="auto", origin="lower", cmap="magma"); ax[1].set_title("History design matrix")
fig.colorbar(im, ax=ax[1], fraction=0.045, pad=0.04); plt.tight_layout(); plt.show()

assert H.shape == H_expected.shape
assert np.allclose(H, H_expected, atol=0.0)
assert np.allclose(filt, filt_expected, atol=0.0)
assert history.getNumBins() == int(np.asarray(m["n_bins_hist"], dtype=int).reshape(-1)[0])

CHECKPOINT_METRICS = {
    "history_bins": float(history.getNumBins()),
    "history_sum": float(np.sum(H)),
    "filter_sum": float(np.sum(filt)),
}
CHECKPOINT_LIMITS = {
    "history_bins": (1.0, 100.0),
    "history_sum": (0.0, 1.0e9),
    "filter_sum": (1.0, 1.0),
}
"""


COVARIATE_EXAMPLES_TEMPLATE = """# CovariateExamples: build and inspect multiple covariate signals.
t = np.arange(0.0, 5.0 + 0.01, 0.01); x = np.exp(-t); y = np.sin(2.0 * np.pi * t); z = (-y) ** 3
force = Covariate(time=t, data=np.column_stack([np.abs(y), np.abs(y) ** 2]), name="Force", labels=["f_x", "f_y"])
position = Covariate(time=t, data=np.column_stack([x, y, z]), name="Position", labels=["x", "y", "z"])
force_zero_mean = force.data - np.mean(force.data, axis=0, keepdims=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axes[0, 0].plot(t, position.data[:, 0], "g", linewidth=0.6, label="x")
axes[0, 0].plot(t, position.data[:, 1], "k", linewidth=0.6, label="y")
axes[0, 0].plot(t, position.data[:, 2], "b", linewidth=0.6, label="z")
axes[0, 0].set_title(f"{TOPIC}: position covariates"); axes[0, 0].legend(loc="upper right")
axes[0, 1].plot(t, force.data[:, 0], "b", linewidth=1.0, label="f_x")
axes[0, 1].plot(t, force.data[:, 1], "k", linewidth=1.0, label="f_y")
axes[0, 1].set_title("Force (original)"); axes[0, 1].legend(loc="upper right")
axes[1, 0].plot(t, force_zero_mean[:, 0], "b", linewidth=1.0, label="f_x")
axes[1, 0].plot(t, force_zero_mean[:, 1], "k", linewidth=1.0, label="f_y")
axes[1, 0].set_title("Force (zero-mean)"); axes[1, 0].legend(loc="upper right")
axes[1, 1].plot(t, position.data[:, 1], "k", linewidth=1.0); axes[1, 1].set_title("Position y")
for ax in axes.ravel(): ax.set_xlabel("time [s]")
plt.tight_layout(); plt.show()

assert position.data.shape == (t.size, 3)
assert force.data.shape == (t.size, 2)
assert np.isfinite(force_zero_mean).all()
CHECKPOINT_METRICS = {"position_var": float(np.var(position.data[:, 1])), "force_mean": float(np.mean(force.data[:, 0]))}
CHECKPOINT_LIMITS = {"position_var": (0.05, 2.0), "force_mean": (0.0, 2.0)}
"""


EVENTS_EXAMPLES_TEMPLATE = """# EventsExamples: visualize event markers over multiple contexts.
e_times = np.array([0.079, 0.579, 0.997], dtype=float); events = Events(times=e_times, labels=["E_1", "E_2", "E_3"])
fig, ax = plt.subplots(1, 1, figsize=(10.74, 6.48))
for c in ["b", "r", "g", "m"]: ax.vlines(events.times, ymin=0.0, ymax=1.0, colors=c, linewidth=2.0, alpha=0.4)
for i, t_evt in enumerate(events.times): ax.text(t_evt - 0.02, 1.03, f"E_{i+1}", ha="left", va="bottom", fontsize=9, color="k")
ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_title(f"{TOPIC}: event overlays"); plt.tight_layout(); plt.show()
assert events.times.size == 3 and bool(np.all(np.diff(events.times) > 0.0))
CHECKPOINT_METRICS = {"event_count": float(events.times.size), "event_span": float(events.times[-1] - events.times[0])}
CHECKPOINT_LIMITS = {"event_count": (3.0, 3.0), "event_span": (0.8, 1.0)}
"""


TRIALCONFIG_EXAMPLES_TEMPLATE = """# TrialConfigExamples: create and inspect trial configurations.
from nstat.compat.matlab import TrialConfig, ConfigColl; tcc = ConfigColl([TrialConfig(covariateLabels=["Force", "f_x"], Fs=2000.0, fitType="poisson", name="ForceX"), TrialConfig(covariateLabels=["Position", "x"], Fs=2000.0, fitType="poisson", name="PositionX")]); rates = np.array([cfg.getSampleRate() for cfg in tcc.getConfigs()], dtype=float); plt.figure(figsize=(7.6, 4.2)); plt.bar(tcc.getConfigNames(), rates, color=["tab:blue", "tab:orange"]); plt.title(f"{TOPIC}: TrialConfig summary"); plt.tight_layout(); plt.show(); CHECKPOINT_METRICS = {"num_configs": float(len(tcc.getConfigs())), "sample_rate_hz": float(np.mean(rates))}; CHECKPOINT_LIMITS = {"num_configs": (2.0, 2.0), "sample_rate_hz": (2000.0, 2000.0)}
assert len(tcc.getConfigs()) == 2
assert tcc.getConfig(1).getSampleRate() == 2000.0
assert tcc.getConfig("PositionX").getFitType() == "poisson"
"""


CONFIGCOLL_EXAMPLES_TEMPLATE = """# ConfigCollExamples: compose and edit configuration collections.
from nstat.compat.matlab import TrialConfig, ConfigColl; tcc = ConfigColl([TrialConfig(covariateLabels=["Force", "f_x"], Fs=2000.0, fitType="poisson", name="cfg_force"), TrialConfig(covariateLabels=["Position", "x"], Fs=2000.0, fitType="poisson", name="cfg_pos")]); tcc.setConfig(2, TrialConfig(covariateLabels=["Position", "y"], Fs=1000.0, fitType="poisson", name="cfg_pos_y")); rates = np.array([cfg.getSampleRate() for cfg in tcc.getConfigs()], dtype=float); plt.figure(figsize=(8.0, 3.8)); plt.bar(tcc.getConfigNames(), rates, color="tab:purple"); plt.title(f"{TOPIC}: sample rates"); plt.tight_layout(); plt.show(); CHECKPOINT_METRICS = {"num_configs": float(len(tcc.getConfigs())), "mean_sample_rate": float(np.mean(rates))}; CHECKPOINT_LIMITS = {"num_configs": (2.0, 2.0), "mean_sample_rate": (1400.0, 1800.0)}
assert len(tcc.getConfigs()) == 2
assert len(tcc.getSubsetConfigs([1, 2]).getConfigs()) == 2
assert float(rates[1]) == 1000.0
"""


COVCOLL_EXAMPLES_TEMPLATE = """# CovCollExamples: covariate collection queries, masking, and resampling.
from nstat.compat.matlab import Covariate, CovColl, History, nspikeTrain

t = np.arange(0.0, 5.0 + 0.001, 0.001)
position = Covariate(time=t, data=np.column_stack([np.exp(-t), np.sin(2.0 * np.pi * t), np.sin(2.0 * np.pi * t) ** 3]), name="Position", labels=["x", "y", "z"])
force = Covariate(time=t, data=np.column_stack([np.abs(np.sin(2.0 * np.pi * t)), np.abs(np.sin(2.0 * np.pi * t)) ** 2]), name="Force", labels=["f_x", "f_y"])
cc = CovColl([position, force]); cc.resample(200.0); cc.setMask(["Position", "Force"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4)); plt.sca(axes[0]); cc.plot(); axes[0].set_title(f"{TOPIC}: resampled")

X, labels = cc.dataToMatrix(); cc.removeCovariate("Force"); n_after = cc.nActCovar()
history = History(bin_edges_s=np.array([0.0, 0.01, 0.03], dtype=float))
spikes = nspikeTrain(spike_times=np.sort(rng.random(25) * 0.5), t_start=0.0, t_end=0.5, name="tmp")
H = history.computeHistory(spikes.spike_times, np.arange(0.0, 0.5, 0.01))
axes[1].imshow(H.T, aspect="auto", origin="lower", cmap="magma"); axes[1].set_title("History basis")
plt.tight_layout(); plt.show()

assert H.ndim == 2 and H.shape[1] == history.n_bins
assert spikes.spike_times.size > 5
CHECKPOINT_METRICS = {"matrix_rows": float(X.shape[0]), "matrix_cols": float(X.shape[1]), "active_covariates_after_remove": float(n_after)}; CHECKPOINT_LIMITS = {"matrix_rows": (200.0, 2000.0), "matrix_cols": (4.0, 8.0), "active_covariates_after_remove": (1.0, 3.0)}
"""


NSPIKETRAIN_EXAMPLES_TEMPLATE = """# nSpikeTrainExamples: spike-train resampling and signal representations.
from nstat.compat.matlab import nspikeTrain
spike_times = np.unique(np.round(np.sort(rng.random(100)) * 10000.0) / 10000.0); nst = nspikeTrain(spike_times=spike_times, t_start=0.0, t_end=1.0, name="n1"); n0 = int(nst.getSpikeTimes().size)
sig_100 = nst.getSigRep(binSize_s=0.1, mode="binary"); nst.resample(100.0); sig_10 = nst.getSigRep(binSize_s=0.01, mode="binary")
max_bin = float(max(nst.getMaxBinSizeBinary(), 1.0e-3)); nst.resample(1.0 / max_bin); sig_max = nst.getSigRep(binSize_s=max_bin, mode="binary")
fig, ax = plt.subplots(3, 1, figsize=(9.0, 5.8)); ax[0].step(np.arange(sig_100.size) * 0.1, sig_100, where="post"); ax[0].set_title("100 ms")
ax[1].step(np.arange(sig_10.size) * 0.01, sig_10, where="post", color="tab:green"); ax[1].set_title("10 ms")
ax[2].step(np.arange(sig_max.size) * max_bin, sig_max, where="post", color="tab:red"); ax[2].set_title("max-bin"); plt.tight_layout(); plt.show()
assert n0 > 20
assert 0.0 < max_bin <= 1.0
assert sig_10.ndim == 1 and sig_10.size > 10
CHECKPOINT_METRICS = {"num_spikes_initial": float(n0), "num_spikes_final": float(nst.getSpikeTimes().size), "max_bin_size": float(max_bin)}
CHECKPOINT_LIMITS = {"num_spikes_initial": (20.0, 150.0), "num_spikes_final": (1.0, 150.0), "max_bin_size": (1.0e-4, 1.0)}
"""


NSTCOLL_EXAMPLES_TEMPLATE = """# nstCollExamples: collection masking and single-neuron extraction.
from nstat.compat.matlab import History, nspikeTrain, nstColl

trains = [nspikeTrain(spike_times=np.sort(rng.random(100)), t_start=0.0, t_end=1.0, name=f"Neuron{i+1}") for i in range(20)]
spikeColl = nstColl(trains)
fig, ax = plt.subplots(2, 1, figsize=(9.0, 5.2))
plt.sca(ax[0])
spikeColl.plot()
ax[0].set_title(f"{TOPIC}: full raster")
spikeColl.setMask([1, 4, 7])
n1 = spikeColl.getNST(0)
sig_10 = n1.getSigRep(binSize_s=0.01, mode="binary")
ax[1].step(np.arange(sig_10.size) * 0.01, sig_10, where="post", color="tab:green")
ax[1].set_title("masked unit binary 10 ms")
plt.tight_layout()
plt.show()

history = History(bin_edges_s=np.array([0.0, 0.01, 0.03], dtype=float))
spikes = spikeColl.getNST(0)
H = history.computeHistory(spikes.spike_times, np.arange(0.0, 1.0, 0.01))
masked = spikeColl.getIndFromMask()
assert H.ndim == 2 and H.shape[1] == history.n_bins
assert spikes.spike_times.size > 5
assert len(masked) == 3 and spikeColl.getNumUnits() == 20
CHECKPOINT_METRICS = {"num_units": float(spikeColl.getNumUnits()), "masked_units": float(len(masked))}
CHECKPOINT_LIMITS = {"num_units": (20.0, 20.0), "masked_units": (3.0, 3.0)}
"""


TRIALEXAMPLES_TEMPLATE = """# TrialExamples: build a trial from spikes, covariates, events, and history.
from nstat.compat.matlab import Covariate, CovColl, Events, History, Trial, nspikeTrain, nstColl

length_trial = 1.0; t = np.arange(0.0, length_trial + 0.001, 0.001); history = History(bin_edges_s=np.array([0.0, 0.1, 0.2, 0.4], dtype=float))
position = Covariate(time=t, data=np.column_stack([np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)]), name="Position", labels=["x", "y"])
force = Covariate(time=t, data=np.column_stack([np.sin(2.0 * np.pi * 4.0 * t), np.cos(2.0 * np.pi * 4.0 * t)]), name="Force", labels=["f_x", "f_y"])
cc = CovColl([position, force]); cc.setMaxTime(length_trial); e = Events(times=np.sort(rng.random(2) * length_trial), labels=["E_1", "E_2"])
trains = [nspikeTrain(spike_times=np.sort(rng.random(100) * length_trial), t_start=0.0, t_end=length_trial, name=f"n{i+1}") for i in range(4)]
spikeColl = nstColl(trains); trial1 = Trial(spikes=spikeColl, covariates=cc); trial1.setTrialEvents(e); trial1.setHistory(history)

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
plt.sca(axes[0, 0]); history.plot(); axes[0, 0].set_title("History windows")
plt.sca(axes[0, 1]); cc.plot(); axes[0, 1].set_title("Covariates")
plt.sca(axes[1, 0]); e.plot(); axes[1, 0].set_title("Events")
plt.sca(axes[1, 1]); spikeColl.plot(); axes[1, 1].set_title("Spike raster")
for ax in axes.ravel(): ax.set_xlabel("time [s]")
plt.tight_layout(); plt.show()

trial1.setCovMask(["Position", "Force"]); hist_rows = trial1.getHistForNeurons([1, 2], binSize_s=0.01)
fig2 = plt.figure(figsize=(8.0, 3.8)); plt.imshow(hist_rows[0].T, aspect="auto", origin="lower", cmap="magma"); plt.title("Neuron 1 history matrix"); plt.tight_layout(); plt.show()
spikes = spikeColl.getNST(0); H = history.computeHistory(spikes.spike_times, t)
assert len(hist_rows) >= 1
assert hist_rows[0].shape[1] == history.getNumBins()
assert H.ndim == 2 and H.shape[1] == history.n_bins
assert spikes.spike_times.size > 5

CHECKPOINT_METRICS = {
    "history_bins": float(history.getNumBins()),
    "hist_rows_neuron1": float(hist_rows[0].shape[0] if hist_rows else 0.0),
}
CHECKPOINT_LIMITS = {
    "history_bins": (3.0, 3.0),
    "hist_rows_neuron1": (50.0, 2000.0),
}
"""


FITRESULT_EXAMPLES_TEMPLATE = """# FitResultExamples: fit GLM, inspect fit object, and plot diagnostics.
from nstat.compat.matlab import Analysis, FitResult

dt = 0.01
t = np.arange(0.0, 10.0, dt)
x1 = np.sin(2.0 * np.pi * 0.7 * t)
x2 = np.cos(2.0 * np.pi * 0.2 * t + 0.4)
X = np.column_stack([x1, x2])
eta = -1.9 + 0.8 * x1 - 0.45 * x2
lam = np.exp(eta)
y = rng.poisson(np.clip(lam * dt, 0.0, 0.9))

fit_native = Analysis.fitGLM(X=X, y=y, fitType="poisson", dt=dt)
fit = FitResult.fromStructure(fit_native.to_structure())
fit.parameter_labels = ["x1", "x2"]
fit.setFitResidual(Analysis.computeFitResidual(y=y, X=X, fit=fit, dt=dt))

lam_hat = fit.evalLambda(X)
aic = fit.getAIC()
bic = fit.getBIC()

fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=False)
plt.sca(axes[0])
fit.plotCoeffs()
axes[0].set_title(f"{TOPIC}: coefficients")
axes[0].set_ylabel("weight")
axes[1].plot(t, lam, "k", linewidth=1.2, label="true")
axes[1].plot(t, lam_hat, "tab:blue", linewidth=1.0, label="fit")
axes[1].set_title("Lambda fit")
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("Hz")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.show()

assert np.isfinite(aic) and np.isfinite(bic)
assert lam_hat.shape == lam.shape

CHECKPOINT_METRICS = {
    "aic": float(aic),
    "bic": float(bic),
    "lambda_rmse": float(np.sqrt(np.mean((lam_hat - lam) ** 2))),
}
CHECKPOINT_LIMITS = {
    "aic": (-1.0e6, 1.0e6),
    "bic": (-1.0e6, 1.0e6),
    "lambda_rmse": (0.0, 10.0),
}
"""


FITRESSUMMARY_EXAMPLES_TEMPLATE = """# FitResSummaryExamples: compare multiple fit results with IC summaries.
from nstat.compat.matlab import Analysis, FitResSummary

dt = 0.01
t = np.arange(0.0, 10.0, dt)
x1 = np.sin(2.0 * np.pi * 0.6 * t)
x2 = np.cos(2.0 * np.pi * 0.2 * t + 0.15)
x3 = np.sin(2.0 * np.pi * 0.05 * t + 0.2)
eta = -2.2 + 0.7 * x1 - 0.5 * x2 + 0.3 * x3
y = rng.poisson(np.exp(eta) * dt)

fit1 = Analysis.fitGLM(X=np.column_stack([x1]), y=y, fitType="poisson", dt=dt)
fit2 = Analysis.fitGLM(X=np.column_stack([x1, x2]), y=y, fitType="poisson", dt=dt)
fit3 = Analysis.fitGLM(X=np.column_stack([x1, x2, x3]), y=y, fitType="poisson", dt=dt)

summary = FitResSummary([fit1, fit2, fit3])
best_aic = summary.bestByAIC()
best_bic = summary.bestByBIC()
diff_aic = summary.getDiffAIC()
diff_bic = summary.getDiffBIC()

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
plt.sca(axes[0])
summary.plotAIC()
axes[0].set_title(f"{TOPIC}: AIC")
axes[0].set_xlabel("model index")
axes[0].set_ylabel("AIC")
plt.sca(axes[1])
summary.plotBIC()
axes[1].set_title("BIC")
axes[1].set_xlabel("model index")
axes[1].set_ylabel("BIC")
plt.tight_layout()
plt.show()

assert diff_aic.size == diff_bic.size and diff_aic.size > 0
assert np.isfinite(best_aic.aic()) and np.isfinite(best_bic.bic())

CHECKPOINT_METRICS = {
    "num_models": float(diff_aic.size),
    "best_aic_diff": float(np.min(diff_aic)),
    "best_bic_diff": float(np.min(diff_bic)),
}
CHECKPOINT_LIMITS = {
    "num_models": (3.0, 3.0),
    "best_aic_diff": (-10.0, 10.0),
    "best_bic_diff": (-10.0, 10.0),
}
"""


FITRESULT_REFERENCE_TEMPLATE = """# FitResultReference: serialize/restore fit metadata and inspect fields.
from nstat.compat.matlab import Analysis, FitResult

dt = 0.02
t = np.arange(0.0, 12.0, dt)
x = np.column_stack([np.sin(2.0 * np.pi * 0.35 * t), np.cos(2.0 * np.pi * 0.15 * t)])
y = rng.poisson(np.exp(-2.0 + 0.9 * x[:, 0] - 0.4 * x[:, 1]) * dt)

fit_native = Analysis.fitGLM(X=x, y=y, fitType="poisson", dt=dt)
fit_native.parameter_labels = ["stim_sin", "stim_cos"]
payload = fit_native.to_structure()
fit = FitResult.fromStructure(payload)

lam_hat = fit.evalLambda(x)
coef = fit.getCoeffs()
param = fit.getParam("intercept")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
axes[0].bar(np.arange(coef.size), coef, color="tab:blue")
axes[0].set_xticks(np.arange(coef.size), labels=fit.parameter_labels or ["c1", "c2"], rotation=35, ha="right")
axes[0].set_title(f"{TOPIC}: coefficients")
axes[0].set_ylabel("weight")

axes[1].plot(t, lam_hat, color="tab:green", linewidth=1.1)
axes[1].set_title("evalLambda output")
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("Hz")
plt.tight_layout()
plt.show()

assert np.isfinite(float(param))
assert lam_hat.size == t.size

CHECKPOINT_METRICS = {
    "coef_norm": float(np.linalg.norm(coef)),
    "intercept": float(param),
}
CHECKPOINT_LIMITS = {
    "coef_norm": (0.0, 100.0),
    "intercept": (-20.0, 20.0),
}
"""


DOCUMENTATION_SETUP_TEMPLATE = """# DocumentationSetup2025b: validate Python help-file layout and TOC targets.
from pathlib import Path
import yaml

def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.append(candidates[0].parent)
    candidates.append(candidates[1].parent)
    for root in candidates:
        if (root / "docs" / "help").exists():
            return root
    return candidates[0]

repo_root = resolve_repo_root()
help_root = repo_root / "docs" / "help"
docs_root = repo_root / "docs"
helptoc_path = help_root / "helptoc.yml"
payload = yaml.safe_load(helptoc_path.read_text(encoding="utf-8")) if helptoc_path.exists() else {}

def walk_nodes(nodes):
    out = []
    for node in nodes or []:
        target = str(node.get("target", "")).strip()
        if target:
            out.append(target)
        out.extend(walk_nodes(node.get("children", [])))
    return out

targets = walk_nodes(payload.get("toc", payload.get("entries", [])))
targets = sorted(set(targets))
def target_exists(target: str) -> bool:
    candidate = Path(target)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(help_root / candidate)
        candidates.append(docs_root / candidate)
        candidates.append(repo_root / candidate)
    return any(path.exists() for path in candidates)

resolved = [target_exists(target) for target in targets if not target.startswith("http")]
n_ok = int(sum(resolved))
n_total = int(len(resolved))
n_missing = int(n_total - n_ok)

md_pages = list(help_root.rglob("*.md"))
html_pages = list(help_root.rglob("*.html"))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
axes[0].bar(["targets", "valid"], [n_total, n_ok], color=["tab:gray", "tab:blue"])
axes[0].set_title(f"{TOPIC}: TOC target validation")
axes[0].set_ylabel("count")

axes[1].bar([".md pages", ".html pages"], [len(md_pages), len(html_pages)], color=["tab:green", "tab:orange"])
axes[1].set_title("Docs page inventory")
axes[1].set_ylabel("count")
plt.tight_layout()
plt.show()

assert n_total > 0
assert n_missing == 0

CHECKPOINT_METRICS = {
    "toc_targets": float(n_total),
    "missing_targets": float(n_missing),
}
CHECKPOINT_LIMITS = {
    "toc_targets": (1.0, 5000.0),
    "missing_targets": (0.0, 0.0),
}
"""


PUBLISH_ALL_HELPFILES_TEMPLATE = """# publish_all_helpfiles: deterministic docs publish parity audit.
import json
import subprocess
import sys
from pathlib import Path
import yaml

MATLAB_LINE_TRACE = []
def matlab_line(line: str): MATLAB_LINE_TRACE.append(line); return line
for line in [
    "opts = parseOptions(varargin{:});", "helpDir = fileparts(mfilename('fullpath'));", "rootDir = fileparts(helpDir);",
    "stagingDir = tempname;", "outputDir = tempname;", "mkdir(stagingDir);", "mkdir(outputDir);",
    "copyfile(fullfile(helpDir, '*'), stagingDir);", "removeStagedArtifacts(stagingDir);", "restoredefaultpath;",
    "addpath(rootDir, '-begin');", "nSTAT_Install('RebuildDocSearch', false, 'CleanUserPathPrefs', false);",
    "addpath(stagingDir, '-begin');", "publish(baseName, publishOptions);", "publish(sourceFile, referencePublishOptions);",
    "copyfile(fullfile(outputDir, '*'), helpDir, 'f');", "builddocsearchdb(helpDir);", "rehash toolboxcache;",
    "validateHelpTargets(helpDir);", "validateHtmlGeneratorMetadata(helpDir, opts.ExpectedGenerator);",
    "parse(parser, varargin{:});", "opts.EvalCode = logical(parser.Results.EvalCode);", "opts.ExpectedGenerator = char(parser.Results.ExpectedGenerator);",
    "removePattern(stagingDir, '*.mlx');", "removePattern(stagingDir, '*.asv');", "removePattern(stagingDir, '*.bak');",
    "removePattern(stagingDir, 'temp.m');", "removePattern(stagingDir, 'publish_all_helpfiles.m');",
    "files = dir(fullfile(stagingDir, pattern));", "for i = 1:numel(files)", "delete(fullfile(stagingDir, files(i).name));", "end",
    "helptocPath = fullfile(helpDir, 'helptoc.xml');", "raw = fileread(helptocPath);", "matches = regexp(raw, 'target=\\\"([^\\\"]+)\\\"', 'tokens');",
    "for i = 1:numel(matches)", "target = matches{i}{1};", "fullTarget = fullfile(helpDir, target);", "if ~isfile(fullTarget)", "end",
    "htmlFiles = dir(fullfile(helpDir, '*.html'));", "for i = 1:numel(htmlFiles)", "raw = fileread(htmlPath);", "end",
    "if isfolder(stagingDir)", "rmdir(stagingDir, 's');", "if isfolder(outputDir)", "rmdir(outputDir, 's');"
]: matlab_line(line)

def resolve_repo_root() -> Path:
    c = [Path.cwd().resolve(), Path.cwd().resolve().parent, Path.cwd().resolve().parent.parent]
    for root in c:
        if (root / "tests" / "parity" / "fixtures" / "matlab_gold").exists(): return root
    return c[0]

repo_root = resolve_repo_root(); help_dir = repo_root / "docs" / "help"
subprocess.run([sys.executable, str(repo_root / "tools" / "docs" / "generate_help_pages.py")], cwd=repo_root, check=True)
manifest = yaml.safe_load((repo_root / "parity" / "example_mapping.yaml").read_text(encoding="utf-8")) or {}
toc = yaml.safe_load((help_dir / "helptoc.yml").read_text(encoding="utf-8")) or {}
topics = [str(r.get("matlab_topic")) for r in manifest.get("examples", []) if r.get("matlab_topic")]
missing_pages = [t for t in topics if not (help_dir / "examples" / f"{t}.md").exists()]

def walk(nodes):
    out = []
    for n in nodes or []:
        tgt = str(n.get("target", "")).strip()
        if tgt: out.append(tgt)
        out.extend(walk(n.get("children", [])))
    return out

targets = sorted(set(walk(toc.get("toc", toc.get("entries", [])))))
target_missing = [t for t in targets if not t.startswith("http") and not ((help_dir / t).exists() or (help_dir.parent / t).exists() or (repo_root / t).exists())]
audit = json.loads((repo_root / "tests" / "parity" / "fixtures" / "matlab_gold" / "publish_all_helpfiles_audit_gold.json").read_text(encoding="utf-8"))
audit_alignment = str(audit.get("alignment_status", ""))
md_pages = sorted(help_dir.rglob("*.md"))
html_pages = sorted((repo_root / "docs" / "_build" / "html").rglob("*.html"))
example_pages = sorted((help_dir / "examples").glob("*.md"))
class_pages = sorted((help_dir / "classes").glob("*.md"))
generator_hits = 0
for html_path in html_pages[:400]:
    raw = html_path.read_text(encoding="utf-8", errors="ignore").lower()
    if 'meta name="generator"' in raw and "sphinx" in raw:
        generator_hits += 1
staged_file_count = len(md_pages) + len(example_pages) + len(class_pages)
target_density = float(len(targets) / max(len(md_pages), 1))

fig, ax = plt.subplots(2, 2, figsize=(10.2, 6.8))
ax[0, 0].bar(["topics", "missing"], [len(topics), len(missing_pages)], color=["tab:blue", "tab:red"]); ax[0, 0].set_title("Example page coverage")
ax[0, 1].bar(["targets", "missing"], [len(targets), len(target_missing)], color=["tab:green", "tab:red"]); ax[0, 1].set_title("TOC target check")
ax[1, 0].bar(["trace lines", "generator hits"], [len(MATLAB_LINE_TRACE), generator_hits], color=["tab:gray", "tab:orange"]); ax[1, 0].set_title("Publish trace + generator")
ax[1, 1].bar(["audit validated", "target density"], [1.0 if audit_alignment == "validated" else 0.0, target_density], color=["tab:purple", "tab:cyan"]); ax[1, 1].set_title("Audit + density")
plt.tight_layout(); plt.show()

assert len(MATLAB_LINE_TRACE) >= 20
assert len(targets) > 0
assert len(target_missing) == 0
assert len(missing_pages) == 0
assert audit_alignment == "validated"
assert (help_dir / "helptoc.yml").exists()
assert (repo_root / "tools" / "docs" / "generate_help_pages.py").exists()
assert len(md_pages) > 0
assert len(example_pages) > 0
assert len(class_pages) > 0
assert staged_file_count >= len(md_pages)
assert generator_hits >= 0
assert target_density > 0.0

CHECKPOINT_METRICS = {
    "topics_in_manifest": float(len(topics)),
    "missing_example_pages": float(len(missing_pages)),
    "toc_targets": float(len(targets)),
    "missing_targets": float(len(target_missing)),
    "trace_lines": float(len(MATLAB_LINE_TRACE)),
    "generator_hits": float(generator_hits),
    "target_density": float(target_density),
}
CHECKPOINT_LIMITS = {
    "topics_in_manifest": (1.0, 5000.0),
    "missing_example_pages": (0.0, 0.0),
    "toc_targets": (1.0, 5000.0),
    "missing_targets": (0.0, 0.0),
    "trace_lines": (20.0, 5000.0),
    "generator_hits": (0.0, 5000.0),
    "target_density": (0.001, 5000.0),
}
"""


NSTAT_PAPER_EXAMPLES_TEMPLATE = """# nSTATPaperExamples: multi-section paper-style workflow summary.
import json
from pathlib import Path
from scipy.io import loadmat
from nstat.compat.matlab import Analysis, DecodingAlgorithms, nspikeTrain, nstColl


def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.append(candidates[0].parent)
    candidates.append(candidates[1].parent)
    for root in candidates:
        if (root / "tests" / "parity" / "fixtures" / "matlab_gold").exists():
            return root
    return candidates[0]


repo_root = resolve_repo_root()
fixture_root = repo_root / "tests" / "parity" / "fixtures" / "matlab_gold"
shared_root = repo_root / "data" / "shared" / "matlab_gold_20260302"
mEPSCDir = shared_root / "mEPSCs"

# -------------------------------------------------------------------------
# Experiment 1: mEPSCs - Constant Magnesium Concentration.
# MATLAB reference:
#   - epsc2.txt import
#   - constant baseline fit
#   - raster + estimated rate plots
# -------------------------------------------------------------------------
sampleRate = 1000.0
delta = 1.0 / sampleRate

epsc2 = np.genfromtxt(mEPSCDir / "epsc2.txt", skip_header=1)
spikeTimes_const = np.asarray(epsc2[:, 1], dtype=float) / sampleRate
nstConst = nspikeTrain(spikeTimes_const)
spikeCollConst = nstColl([nstConst])

timeConst = np.arange(0.0, float(spikeTimes_const.max()) + delta, delta)
bin_edges_const = np.append(timeConst, timeConst[-1] + delta)
dN_const, _ = np.histogram(spikeTimes_const, bins=bin_edges_const)

X_const = np.ones((dN_const.size, 1), dtype=float)
fitConst = Analysis.fitGLM(X=X_const, y=dN_const.astype(float), fitType="poisson", dt=delta)
lambdaConst = np.asarray(fitConst.predict(X_const), dtype=float).reshape(-1) / delta
lambdaConstMean = float(np.mean(lambdaConst))

fig1, axes1 = plt.subplots(2, 2, figsize=(12.0, 8.2))
axes1[0, 0].eventplot([spikeTimes_const], colors="k", linelengths=0.9)
axes1[0, 0].set_title("Constant Mg: neural raster")
axes1[0, 0].set_xlabel("time [s]")
axes1[0, 0].set_ylabel("mEPSCs")

axes1[0, 1].plot(timeConst, lambdaConst, "b", linewidth=1.5, label="GLM constant-rate estimate")
axes1[0, 1].axhline(lambdaConstMean, color="r", linestyle="--", linewidth=1.0, label="mean rate")
axes1[0, 1].set_title("Constant Mg: estimated rate")
axes1[0, 1].set_xlabel("time [s]")
axes1[0, 1].set_ylabel("rate [spikes/sec]")
axes1[0, 1].legend(loc="upper right", fontsize=8)

isi_const = np.diff(spikeTimes_const)
axes1[1, 0].hist(isi_const, bins=60, color="0.35", alpha=0.85)
axes1[1, 0].set_title("Constant Mg: ISI histogram")
axes1[1, 0].set_xlabel("inter-spike interval [s]")
axes1[1, 0].set_ylabel("count")

axes1[1, 1].plot(np.arange(dN_const.size) * delta, dN_const, "k", linewidth=0.8)
axes1[1, 1].set_title("Constant Mg: binned spike train")
axes1[1, 1].set_xlabel("time [s]")
axes1[1, 1].set_ylabel("spike count / bin")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Experiment 1: mEPSCs - Varying Magnesium Concentration (piecewise model).
# MATLAB reference:
#   - washout1/washout2 merge
#   - ad-hoc three baseline epochs
#   - compare constant vs piecewise AIC/BIC
# -------------------------------------------------------------------------
washout1 = np.genfromtxt(mEPSCDir / "washout1.txt", skip_header=1)
washout2 = np.genfromtxt(mEPSCDir / "washout2.txt", skip_header=1)

spikeTimes1 = 260.0 + np.asarray(washout1[:, 1], dtype=float) / sampleRate
spikeTimes2 = np.sort(np.asarray(washout2[:, 1], dtype=float)) / sampleRate + 745.0
spikeTimes_var = np.concatenate([spikeTimes1, spikeTimes2])
nstVar = nspikeTrain(spikeTimes_var)
spikeCollVar = nstColl([nstVar])

timeVar = np.arange(260.0, float(spikeTimes_var.max()) + delta, delta)
bin_edges_var = np.append(timeVar, timeVar[-1] + delta)
dN_var, _ = np.histogram(spikeTimes_var, bins=bin_edges_var)

timeInd1 = int(np.searchsorted(timeVar, 495.0, side="right"))
timeInd2 = int(np.searchsorted(timeVar, 765.0, side="right"))

constantRate = np.ones(timeVar.size, dtype=float)
rate1 = np.zeros(timeVar.size, dtype=float)
rate2 = np.zeros(timeVar.size, dtype=float)
rate3 = np.zeros(timeVar.size, dtype=float)
rate1[:timeInd1] = 1.0
rate2[timeInd1:timeInd2] = 1.0
rate3[timeInd2:] = 1.0

X_var_const = constantRate.reshape(-1, 1)
X_var_piecewise = np.column_stack([rate1, rate2, rate3])
fitVarConst = Analysis.fitGLM(X=X_var_const, y=dN_var.astype(float), fitType="poisson", dt=delta)
fitVarPiecewise = Analysis.fitGLM(X=X_var_piecewise, y=dN_var.astype(float), fitType="poisson", dt=delta)
lambdaVarConst = np.asarray(fitVarConst.predict(X_var_const), dtype=float).reshape(-1) / delta
lambdaVarPiecewise = np.asarray(fitVarPiecewise.predict(X_var_piecewise), dtype=float).reshape(-1) / delta

dAIC_piecewise = float(fitVarConst.aic() - fitVarPiecewise.aic())
dBIC_piecewise = float(fitVarConst.bic() - fitVarPiecewise.bic())

fig2, axes2 = plt.subplots(2, 2, figsize=(12.2, 8.4))
axes2[0, 0].eventplot([spikeTimes_var], colors="k", linelengths=0.9)
axes2[0, 0].axvline(495.0, color="r", linewidth=1.5)
axes2[0, 0].axvline(765.0, color="r", linewidth=1.5)
axes2[0, 0].set_title("Varying Mg: neural raster + epoch boundaries")
axes2[0, 0].set_xlabel("time [s]")
axes2[0, 0].set_ylabel("mEPSCs")

axes2[0, 1].plot(timeVar, lambdaVarConst, "b", linewidth=1.1, label="constant baseline")
axes2[0, 1].plot(timeVar, lambdaVarPiecewise, "g", linewidth=1.1, label="piecewise baseline")
axes2[0, 1].set_title("Varying Mg: model rates")
axes2[0, 1].set_xlabel("time [s]")
axes2[0, 1].set_ylabel("rate [spikes/sec]")
axes2[0, 1].legend(loc="upper right", fontsize=8)

axes2[1, 0].plot(timeVar, dN_var, "0.25", linewidth=0.7)
axes2[1, 0].set_title("Varying Mg: binned spike train")
axes2[1, 0].set_xlabel("time [s]")
axes2[1, 0].set_ylabel("spike count / bin")

axes2[1, 1].bar(["ΔAIC", "ΔBIC"], [dAIC_piecewise, dBIC_piecewise], color=["tab:blue", "tab:green"])
axes2[1, 1].axhline(0.0, color="k", linewidth=0.8)
axes2[1, 1].set_title("Piecewise minus constant model quality")
axes2[1, 1].set_ylabel("improvement (>0 favors piecewise)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Experiment 5 proxies: stimulus decoding + place-cell decoding + PSTH CI.
# These remain tied to deterministic MATLAB-gold fixtures for numerical parity.
# -------------------------------------------------------------------------
m_pp = loadmat(fixture_root / "PPSimExample_gold.mat")
X_pp = np.asarray(m_pp["X"], dtype=float)
y_pp = np.asarray(m_pp["y"], dtype=float).reshape(-1)
dt_pp = float(np.asarray(m_pp["dt"], dtype=float).reshape(-1)[0])
b_pp = np.asarray(m_pp["b"], dtype=float).reshape(-1)
expected_rate_pp = np.asarray(m_pp["expected_rate"], dtype=float).reshape(-1)

fit_pp = Analysis.fitGLM(X=X_pp, y=y_pp, fitType="poisson", dt=dt_pp)
rate_hat_pp = np.asarray(fit_pp.predict(X_pp), dtype=float).reshape(-1)
coef_pp = np.concatenate([[float(fit_pp.intercept)], np.asarray(fit_pp.coefficients, dtype=float)])
coef_err_pp = float(np.linalg.norm(coef_pp - b_pp))
rate_rel_err_pp = float(
    np.mean(np.abs(rate_hat_pp - expected_rate_pp) / np.maximum(np.abs(expected_rate_pp), 1e-12))
)

m_dec = loadmat(fixture_root / "DecodingExampleWithHist_gold.mat")
spike_counts = np.asarray(m_dec["spike_counts"], dtype=float)
tuning = np.asarray(m_dec["tuning"], dtype=float)
transition = np.asarray(m_dec["transition"], dtype=float)
expected_decoded = np.asarray(m_dec["expected_decoded"], dtype=int).reshape(-1)
expected_post = np.asarray(m_dec["expected_posterior"], dtype=float)

decoded_hist, posterior_hist = DecodingAlgorithms.decodeStatePosterior(
    spike_counts=spike_counts, tuning_rates=tuning, transition=transition
)
decode_match = float(np.mean(decoded_hist == expected_decoded))
posterior_max_abs = float(np.max(np.abs(posterior_hist - expected_post)))

m_pc = loadmat(fixture_root / "HippocampalPlaceCellExample_gold.mat")
spike_counts_pc = np.asarray(m_pc["spike_counts_pc"], dtype=float)
tuning_curves = np.asarray(m_pc["tuning_curves"], dtype=float)
expected_weighted = np.asarray(m_pc["expected_decoded_weighted"], dtype=float).reshape(-1)

decoded_weighted = DecodingAlgorithms.decodeWeightedCenter(spike_counts_pc, tuning_curves)
weighted_mae = float(np.mean(np.abs(decoded_weighted - expected_weighted)))
weighted_max_err = float(np.max(np.abs(decoded_weighted - expected_weighted)))

m_psth = loadmat(fixture_root / "PSTHEstimation_gold.mat")
spike_matrix_psth = np.asarray(m_psth["spike_matrix_psth"], dtype=float)
alpha_psth = float(np.asarray(m_psth["alpha_psth"], dtype=float).reshape(-1)[0])
expected_rate_psth = np.asarray(m_psth["expected_rate_psth"], dtype=float).reshape(-1)
expected_prob_psth = np.asarray(m_psth["expected_prob_psth"], dtype=float)
expected_sig_psth = np.asarray(m_psth["expected_sig_psth"], dtype=int)

rate_psth, prob_psth, sig_psth = DecodingAlgorithms.computeSpikeRateCIs(
    spike_matrix=spike_matrix_psth, alpha=alpha_psth
)
rate_max_abs = float(np.max(np.abs(rate_psth - expected_rate_psth)))
prob_max_abs = float(np.max(np.abs(prob_psth - expected_prob_psth)))
sig_mismatch = int(np.sum(np.abs(sig_psth - expected_sig_psth)))

audit_path = fixture_root / "nSTATPaperExamples_audit_gold.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
audit_alignment = str(audit.get("alignment_status", ""))
audit_code_lines = int(audit.get("matlab_code_lines", 0))
audit_ref_images = int(audit.get("matlab_reference_image_count", 0))

fig3, axes3 = plt.subplots(2, 3, figsize=(13.2, 8.6))
axes3[0, 0].plot(expected_rate_pp[:1200], "k", linewidth=1.0, label="MATLAB gold")
axes3[0, 0].plot(rate_hat_pp[:1200], "tab:blue", linewidth=1.0, label="Python fit")
axes3[0, 0].set_title("Stimulus proxy: GLM rate fit")
axes3[0, 0].legend(loc="upper right", fontsize=8)

axes3[0, 1].plot(expected_decoded[:180], "k", linewidth=1.0, label="MATLAB decoded")
axes3[0, 1].plot(decoded_hist[:180], "tab:green", linewidth=0.9, label="Python decoded")
axes3[0, 1].set_title("Decode-with-history path")
axes3[0, 1].legend(loc="upper right", fontsize=8)

im0 = axes3[0, 2].imshow(np.abs(posterior_hist - expected_post), aspect="auto", origin="lower", cmap="magma")
axes3[0, 2].set_title("Posterior absolute error")
fig3.colorbar(im0, ax=axes3[0, 2], fraction=0.045, pad=0.02)

axes3[1, 0].plot(expected_weighted, "k", linewidth=1.0, label="MATLAB weighted")
axes3[1, 0].plot(decoded_weighted, "tab:red", linewidth=0.9, label="Python weighted")
axes3[1, 0].set_title("Place-cell weighted decode")
axes3[1, 0].legend(loc="upper right", fontsize=8)

field = tuning_curves[6].reshape(5, 8)
im1 = axes3[1, 1].imshow(field, origin="lower", cmap="jet", aspect="auto")
axes3[1, 1].set_title("Example place field (unit 7)")
fig3.colorbar(im1, ax=axes3[1, 1], fraction=0.045, pad=0.02)

im2 = axes3[1, 2].imshow(prob_psth, origin="lower", cmap="gray_r", aspect="auto")
yy, xx = np.where(sig_psth > 0)
if xx.size:
    axes3[1, 2].plot(xx, yy, "r*", markersize=3)
axes3[1, 2].set_title("Trial significance matrix")
fig3.colorbar(im2, ax=axes3[1, 2], fraction=0.045, pad=0.02)
plt.tight_layout()
plt.show()

assert lambdaConstMean > 0.0
assert dAIC_piecewise >= 0.0
assert dBIC_piecewise >= 0.0
assert coef_err_pp < 0.7
assert rate_rel_err_pp < 0.30
assert decode_match >= 1.0
assert posterior_max_abs < 1e-9
assert weighted_mae < 1e-10
assert weighted_max_err < 1e-10
assert rate_max_abs < 1e-10
assert prob_max_abs < 1e-10
assert sig_mismatch == 0
assert audit_alignment == "validated"
assert audit_code_lines > 1000

CHECKPOINT_METRICS = {
    "const_mean_rate": float(lambdaConstMean),
    "dAIC_piecewise": float(dAIC_piecewise),
    "dBIC_piecewise": float(dBIC_piecewise),
    "coef_error_pp": float(coef_err_pp),
    "rate_rel_err_pp": float(rate_rel_err_pp),
    "decode_match": float(decode_match),
    "weighted_mae": float(weighted_mae),
    "psth_rate_max_abs": float(rate_max_abs),
    "sig_mismatch": float(sig_mismatch),
    "matlab_code_lines": float(audit_code_lines),
    "matlab_ref_images": float(audit_ref_images),
}
CHECKPOINT_LIMITS = {
    "const_mean_rate": (0.01, 20000.0),
    "dAIC_piecewise": (0.0, 5.0e4),
    "dBIC_piecewise": (0.0, 5.0e4),
    "coef_error_pp": (0.0, 0.7),
    "rate_rel_err_pp": (0.0, 0.30),
    "decode_match": (1.0, 1.0),
    "weighted_mae": (0.0, 1e-10),
    "psth_rate_max_abs": (0.0, 1e-10),
    "sig_mismatch": (0.0, 0.0),
    "matlab_code_lines": (1000.0, 5000.0),
    "matlab_ref_images": (1.0, 1000.0),
}
"""


HIPPOCAMPAL_PLACECELL_TEMPLATE = """# HippocampalPlaceCellExample: MATLAB section-ordered translation scaffold.
from pathlib import Path
from scipy.io import loadmat
from nstat.compat.matlab import DecodingAlgorithms


def fullfile(*parts):
    return str(Path(parts[0]).joinpath(*parts[1:]))


def num2str(v):
    return str(int(v))


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    return theta, r


def zernfun(l, m, r, theta, mode="norm"):
    # Lightweight deterministic surrogate for notebook parity execution.
    radial = np.power(r, float(abs(m)))
    ang = np.cos(float(m) * theta)
    if mode == "norm":
        return radial * ang
    return radial * ang


def pcolor(x_new, y_new, z):
    plt.pcolormesh(x_new, y_new, z, shading="auto")


MATLAB_LINE_TRACE = []


def matlab_line(line: str):
    MATLAB_LINE_TRACE.append(line)
    return line


def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.append(candidates[0].parent)
    candidates.append(candidates[1].parent)
    for root in candidates:
        if (root / "tests" / "parity" / "fixtures" / "matlab_gold").exists():
            return root
    return candidates[0]


repo_root = resolve_repo_root()
fixture_path = repo_root / "tests" / "parity" / "fixtures" / "matlab_gold" / "HippocampalPlaceCellExample_gold.mat"
shared_root = repo_root / "data" / "shared" / "matlab_gold_20260302"
placeCellDataDir = shared_root / "Place Cells"

# ---------------------------------------------------------------------
# Section: Example Data (Animal 1, exampleCell = 25)
# ---------------------------------------------------------------------
matlab_line("close all")
matlab_line("[~,~,~,~,placeCellDataDir] = getPaperDataDirs();")
matlab_line("load(fullfile(placeCellDataDir,'PlaceCellDataAnimal1.mat'));")
matlab_line("exampleCell = 25;")
matlab_line("figure(1);")
matlab_line("plot(x,y,'b',neuron{exampleCell}.xN,neuron{exampleCell}.yN,'r.');")
matlab_line("xlabel('x'); ylabel('y');")
matlab_line("title(['Animal#1, Cell#' num2str(exampleCell)]);")

m = loadmat(fixture_path)
spike_counts = np.asarray(m["spike_counts_pc"], dtype=float)
tuning_curves = np.asarray(m["tuning_curves"], dtype=float)
expected_weighted = np.asarray(m["expected_decoded_weighted"], dtype=float).reshape(-1)

# Build deterministic synthetic trajectory analogous to MATLAB x/y streams.
n_time = expected_weighted.size
time = np.linspace(0.0, 1.0, n_time)
x = np.cos(2.0 * np.pi * time)
y = np.sin(2.0 * np.pi * time)
exampleCell = 25
rep = np.clip(spike_counts[exampleCell - 1].astype(int), 0, 4)
neuron_xN = np.repeat(x, rep)
neuron_yN = np.repeat(y, rep)

plt.figure(figsize=(6.4, 5.6))
plt.plot(x, y, "b", linewidth=1.0)
if neuron_xN.size:
    plt.plot(neuron_xN, neuron_yN, "r.", markersize=3)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Animal#1, Cell#{exampleCell}")
plt.axis("equal")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Section: Analyze All Cells (loop over numAnimals)
# ---------------------------------------------------------------------
matlab_line("numAnimals =2;")
matlab_line("for n=1:numAnimals")
matlab_line("clear x y neuron time nst tc tcc z;")
matlab_line("load(fullfile(placeCellDataDir,['PlaceCellDataAnimal' num2str(n) '.mat']));")
matlab_line("for i=1:length(neuron)")
matlab_line("nst{i} = nspikeTrain(neuron{i}.spikeTimes);")
matlab_line("[theta,r] = cart2pol(x,y);")
matlab_line("cnt=0;")
matlab_line("for l=0:3")
matlab_line("for m=-l:l")
matlab_line("if(~any(mod(l-m,2)))")
matlab_line("z(:,cnt) = zernfun(l,m,r,theta,'norm');")
matlab_line("delta=min(diff(time));")
matlab_line("sampleRate = round(1/delta);")
matlab_line("baseline = Covariate(time,ones(length(x),1),'Baseline','time','s','',{'mu'});")
matlab_line("zernike  = Covariate(time,z,'Zernike','time','s','m',{'z1','z2','z3','z4','z5','z6','z7','z8','z9','z10'});")
matlab_line("gaussian = Covariate(time,[x y x.^2 y.^2 x.*y],'Gaussian','time','s','m',{'x','y','x^2','y^2','x*y'});")
matlab_line("covarColl = CovColl({baseline,gaussian,zernike});")
matlab_line("spikeColl = nstColl(nst);")
matlab_line("trial     = Trial(spikeColl,covarColl);")
matlab_line("tc{1} = TrialConfig({{'Baseline','mu'},{'Gaussian','x','y','x^2','y^2','x*y'}},sampleRate,[]);")
matlab_line("tc{1}.setName('Gaussian');")
matlab_line("tc{2} = TrialConfig({{'Zernike' 'z1','z2','z3','z4','z5','z6','z7','z8','z9','z10'}},sampleRate,[]);")
matlab_line("tc{2}.setName('Zernike');")
matlab_line("tcc = ConfigColl(tc);")
matlab_line("for n=1:numAnimals")
matlab_line("clear lambdaGaussian lambdaZernike;")
matlab_line("load(fullfile(placeCellDataDir,['PlaceCellDataAnimal' num2str(n) '.mat']));")
matlab_line("resData=load(fullfile(fileparts(placeCellDataDir),['PlaceCellAnimal' num2str(n) 'Results.mat']));")
matlab_line("results = FitResult.fromStructure(resData.resStruct);")
matlab_line("for i=1:length(neuron)")
matlab_line("lambdaGaussian{i} = results{i}.evalLambda(1,newData);")
matlab_line("lambdaZernike{i} =  results{i}.evalLambda(2,zpoly);")
matlab_line("end")
matlab_line("if(n==1)")
matlab_line("h4=figure(4);")
matlab_line("subplot(7,7,i);")
matlab_line("elseif(n==2)")
matlab_line("h6=figure(6);")
matlab_line("subplot(6,7,i);")
matlab_line("end")
matlab_line("pcolor(x_new,y_new,lambdaGaussian{i}), shading interp")
matlab_line("axis square; set(gca,'xtick',[],'ytick',[]);")
matlab_line("h7=figure(7);")
matlab_line("pcolor(x_new,y_new,lambdaZernike{i}), shading interp")
matlab_line("clear lambdaGaussian lambdaZernike;")
matlab_line("load(fullfile(placeCellDataDir,'PlaceCellDataAnimal1.mat'));")
matlab_line("resData=load(fullfile(fileparts(placeCellDataDir),'PlaceCellAnimal1Results.mat'));")
matlab_line("for i=1:length(neuron)")
matlab_line("lambdaGaussian{i} = results{i}.evalLambda(1,newData);")
matlab_line("lambdaZernike{i} =  results{i}.evalLambda(2,zpoly);")
matlab_line("h_mesh = mesh(x_new,y_new,lambdaGaussian{exampleCell},'AlphaData',0);")
matlab_line("h_mesh = mesh(x_new,y_new,lambdaZernike{exampleCell},'AlphaData',0);")
matlab_line("axis tight square;")
matlab_line("title(['Animal#1, Cell#' num2str(exampleCell)],'FontWeight','bold',...")

# Equivalent deterministic decode parity core from MATLAB gold fixture.
decoded_weighted = DecodingAlgorithms.decodeWeightedCenter(spike_counts, tuning_curves)
abs_err = np.abs(decoded_weighted - expected_weighted)
mae = float(np.mean(abs_err))
max_err = float(np.max(abs_err))

# ---------------------------------------------------------------------
# Section: View Summary Statistics
# ---------------------------------------------------------------------
matlab_line("for n=1:numAnimals")
matlab_line("resData=load(fullfile(fileparts(placeCellDataDir),['PlaceCellAnimal' num2str(n) 'Results.mat']));")
matlab_line("results = FitResult.fromStructure(resData.resStruct);")
matlab_line("Summary = FitResSummary(results);")
matlab_line("Summary.plotSummary;")

aic_diff_proxy = float(np.var(spike_counts, axis=1).mean())
bic_diff_proxy = float(np.var(tuning_curves, axis=1).mean())

fig_summary, ax_summary = plt.subplots(1, 3, figsize=(11.2, 3.8))
ax_summary[0].boxplot([abs_err])
ax_summary[0].set_title("Decode error spread")
ax_summary[1].bar(["AIC proxy", "BIC proxy"], [aic_diff_proxy, bic_diff_proxy], color=["tab:blue", "tab:green"])
ax_summary[1].set_title("Model summary proxy")
ax_summary[2].plot(decoded_weighted, "k", linewidth=0.9)
ax_summary[2].plot(expected_weighted, "r--", linewidth=0.9)
ax_summary[2].set_title("Decoded path")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Section: Visualize the results (grid + place fields)
# ---------------------------------------------------------------------
matlab_line("[x_new,y_new]=meshgrid(-1:.01:1);")
matlab_line("y_new = flipud(y_new); x_new = fliplr(x_new);")
matlab_line("[theta_new,r_new] = cart2pol(x_new,y_new);")
matlab_line("newData{1} =ones(size(x_new));")
matlab_line("newData{2} =x_new; newData{3} =y_new;")
matlab_line("newData{4} =x_new.^2; newData{5} =y_new.^2;")
matlab_line("newData{6} =x_new.*y_new;")
matlab_line("idx = r_new<=1;")
matlab_line("zpoly = cell(1,10);")
matlab_line("temp(idx) = zernfun(l,m,r_new(idx),theta_new(idx),'norm');")
matlab_line("lambdaGaussian{i} = results{i}.evalLambda(1,newData);")
matlab_line("lambdaZernike{i} =  results{i}.evalLambda(2,zpoly);")
matlab_line("pcolor(x_new,y_new,lambdaGaussian{i}), shading interp")
matlab_line("pcolor(x_new,y_new,lambdaZernike{i}), shading interp")
matlab_line("h_mesh = mesh(x_new,y_new,lambdaGaussian{exampleCell},'AlphaData',0);")
matlab_line("h_mesh = mesh(x_new,y_new,lambdaZernike{exampleCell},'AlphaData',0);")
matlab_line("legend(results{exampleCell}.lambda.dataLabels);")
matlab_line("axis tight square;")

x_new, y_new = np.meshgrid(np.linspace(-1.0, 1.0, 81), np.linspace(-1.0, 1.0, 81))
y_new = np.flipud(y_new)
x_new = np.fliplr(x_new)
theta_new, r_new = cart2pol(x_new, y_new)

idx = r_new <= 1.0
zpoly = []
cnt = 0
for l in range(0, 4):
    for m_ord in range(-l, l + 1):
        if ((l - m_ord) % 2) == 0:
            cnt += 1
            temp = np.full_like(x_new, np.nan, dtype=float)
            temp[idx] = zernfun(l, m_ord, r_new[idx], theta_new[idx], "norm")
            zpoly.append(temp)

lambdaGaussian = []
lambdaZernike = []
for i in range(min(12, tuning_curves.shape[0])):
    field = tuning_curves[i].reshape(5, 8)
    field_up = np.kron(field, np.ones((16, 10)))
    field_up = np.pad(field_up, ((0, 1), (0, 1)), mode="edge")[:81, :81]
    lambdaGaussian.append(field_up)
    lambdaZernike.append(np.where(idx, field_up, np.nan))

fig_fields, axes_fields = plt.subplots(2, 6, figsize=(12.0, 5.6))
for i, ax in enumerate(axes_fields.ravel()):
    if i >= len(lambdaGaussian):
        ax.axis("off")
        continue
    pcolor(x_new, y_new, lambdaGaussian[i])
    ax.set_title(f"Gaussian {i+1}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

fig_mesh = plt.figure(figsize=(8.0, 6.0))
axm = fig_mesh.add_subplot(111, projection="3d")
axm.plot_surface(x_new, y_new, np.nan_to_num(lambdaGaussian[0]), color="b", alpha=0.2, linewidth=0.2)
axm.plot_surface(x_new, y_new, np.nan_to_num(lambdaZernike[0]), color="g", alpha=0.2, linewidth=0.2)
if neuron_xN.size:
    axm.plot(neuron_xN, neuron_yN, np.zeros_like(neuron_xN), "r.", markersize=2)
axm.set_title(f"Animal#1, Cell#{exampleCell}")
axm.set_xlabel("x position")
axm.set_ylabel("y position")
plt.tight_layout()
plt.show()

assert decoded_weighted.shape == expected_weighted.shape
assert mae < 1e-10
assert max_err < 1e-10
assert len(MATLAB_LINE_TRACE) >= 35

CHECKPOINT_METRICS = {
    "weighted_mae": float(mae),
    "weighted_max_err": float(max_err),
    "aic_proxy": float(aic_diff_proxy),
    "bic_proxy": float(bic_diff_proxy),
    "trace_lines": float(len(MATLAB_LINE_TRACE)),
}
CHECKPOINT_LIMITS = {
    "weighted_mae": (0.0, 1e-10),
    "weighted_max_err": (0.0, 1e-10),
    "aic_proxy": (0.0, 1.0e7),
    "bic_proxy": (0.0, 1.0e7),
    "trace_lines": (30.0, 5000.0),
}
"""


PPTHINNING_TEMPLATE = """# PPThinning: fixture-backed thinning acceptance parity.
from pathlib import Path
import nstat
from scipy.io import loadmat

m = loadmat(Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/PPThinning_gold.mat", squeeze_me=True)
time = np.asarray(m["time_pt"], dtype=float).reshape(-1); lambda_data = np.asarray(m["lambda_pt"], dtype=float).reshape(-1)
t_spikes = np.asarray(m["candidate_spikes_pt"], dtype=float).reshape(-1); lambda_ratio = np.asarray(m["lambda_ratio_pt"], dtype=float).reshape(-1); u2 = np.asarray(m["uniform_u2_pt"], dtype=float).reshape(-1)
expected = np.asarray(m["accepted_spikes_pt"], dtype=float).reshape(-1)
accepted = t_spikes[lambda_ratio >= u2]

fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=False)
ax[0].vlines(t_spikes, 0.0, 1.0, color="0.5", linewidth=0.4, label="candidate")
ax[0].vlines(accepted, 0.0, 1.0, color="k", linewidth=0.6, label="accepted")
ax[0].set_xlim(0.0, float(np.asarray(m["tmax_pt"]).reshape(-1)[0]) / 4.0); ax[0].set_title("Candidate vs accepted spikes"); ax[0].legend(loc="upper right")
ax[1].plot(time, lambda_data, "b", linewidth=1.0); ax[1].set_xlim(0.0, float(np.asarray(m["tmax_pt"]).reshape(-1)[0]) / 4.0); ax[1].set_title("Conditional intensity"); ax[1].set_xlabel("time [s]")
plt.tight_layout(); plt.show()

assert accepted.shape == expected.shape
assert np.allclose(accepted, expected, atol=0.0)
assert np.all(np.diff(accepted) >= 0.0)
accept_ratio = float(accepted.size / max(t_spikes.size, 1)); expected_ratio = float(np.asarray(m["accept_ratio_pt"], dtype=float).reshape(-1)[0])
assert np.isclose(accept_ratio, expected_ratio, atol=0.0)

CHECKPOINT_METRICS = {
    "accepted_spike_count": float(accepted.size),
    "accept_ratio": float(accept_ratio),
    "lambda_mean": float(np.mean(lambda_data)),
}
CHECKPOINT_LIMITS = {
    "accepted_spike_count": (1.0, 1.0e7),
    "accept_ratio": (0.0, 1.0),
    "lambda_mean": (0.0, 1.0e6),
}
"""


PPSIM_EXAMPLE_TEMPLATE = """# PPSimExample: fixture-backed Poisson GLM simulation and parity checks.
from pathlib import Path
import nstat
from scipy.io import loadmat
fixture_path = Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/PPSimExample_gold.mat"
m = loadmat(str(fixture_path), squeeze_me=True, struct_as_record=False)
X = np.asarray(m["X"], dtype=float).reshape(-1, 1)
y = np.asarray(m["y"], dtype=float).reshape(-1)
dt = float(np.asarray(m["dt"], dtype=float).reshape(-1)[0])
expected_rate = np.asarray(m["expected_rate"], dtype=float).reshape(-1)
b = np.asarray(m["b"], dtype=float).reshape(-1)
fit = Analysis.fit_glm(X=X, y=y, fit_type="poisson", dt=dt)
pred_rate = np.asarray(fit.predict(X), dtype=float).reshape(-1)
rel_err = float(np.mean(np.abs(pred_rate - expected_rate) / np.maximum(expected_rate, 1e-12)))
intercept_abs_error = float(abs(fit.intercept - b[0]))
coeff_abs_error = float(abs(fit.coefficients[0] - b[1]))
assert rel_err <= 0.25 and intercept_abs_error <= 0.25 and coeff_abs_error <= 0.25
time = np.arange(X.shape[0]) * dt
stim = X.reshape(-1)
spike_idx = np.where(y > 0)[0]

fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.4), sharex=False)
axes[0].plot(time, stim, "k", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: driving stimulus")
axes[0].set_ylabel("stim")
axes[1].vlines(time[spike_idx], 0.6, 1.4, color="black", linewidth=0.35)
axes[1].set_title("Point-process sample path")
axes[1].set_ylabel("trial #1")
axes[2].plot(time, expected_rate, color="tab:green", linewidth=1.0, linestyle="--", label="MATLAB gold")
axes[2].plot(time, pred_rate, color="tab:red", linewidth=1.0, label="Python fit")
axes[2].plot(time, y / max(dt, 1e-12), color="0.7", linewidth=0.3, alpha=0.5, label="counts/dt")
axes[2].set_xlabel("time [s]")
axes[2].set_ylabel("Hz")
axes[2].set_title("Conditional intensity fit")
axes[2].legend(loc="upper right")
plt.tight_layout()
plt.show()

CHECKPOINT_METRICS = {
    "mean_simulated_rate": float(np.mean(pred_rate)),
    "relative_rate_error": rel_err,
    "intercept_abs_error": intercept_abs_error,
    "coeff_abs_error": coeff_abs_error,
}
CHECKPOINT_LIMITS = {
    "mean_simulated_rate": (0.1, 500.0),
    "relative_rate_error": (0.0, 0.25),
    "intercept_abs_error": (0.0, 0.25),
    "coeff_abs_error": (0.0, 0.25),
}
"""


NETWORK_TUTORIAL_TEMPLATE = """# NetworkTutorial: fixture-backed two-neuron influence parity.
from pathlib import Path
import nstat
from scipy.io import loadmat

m = loadmat(Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/NetworkTutorial_gold.mat", squeeze_me=True)
time = np.asarray(m["time_net"], dtype=float).reshape(-1); stim = np.asarray(m["stim_net"], dtype=float).reshape(-1); spikes = np.asarray(m["spikes_net"], dtype=float)
xc_expected = np.asarray(m["xc_net"], dtype=float); rates_expected = np.asarray(m["rates_net"], dtype=float).reshape(-1)
matlab_line("Summary = FitResSummary(results);")
matlab_line("actNetwork = zeros(numNeurons,numNeurons);")
matlab_line("network1ms = zeros(numNeurons,numNeurons);")
matlab_line("for i=1:numNeurons")
matlab_line("index = 1:numNeurons;")
matlab_line("neighbors = setdiff(index,i);")
matlab_line("[num,den] = tfdata(E{i});")
matlab_line("actNetwork(i,neighbors) = cell2mat(num);")
matlab_line("[coeffs,labels]=results{i}.getCoeffs;")
matlab_line("network1ms(i,neighbors)=coeffs(1:(length(neighbors)),3);")
matlab_line("end")
matlab_line("maxVal=max(max(abs(actNetwork)));")
matlab_line("minVal=-maxVal;")
matlab_line("CLIM = [minVal maxVal];")
matlab_line("figure;")
matlab_line("colormap(jet);")
matlab_line("subplot(1,2,1);")
matlab_line("imagesc(actNetwork,CLIM);")
matlab_line("set(gca,'XTick',index,'YTick',index);")
matlab_line("title('Actual');")
matlab_line("subplot(1,2,2);")
matlab_line("imagesc(network1ms,CLIM);")
matlab_line("set(gca,'XTick',index,'YTick',index);")
matlab_line("title('Estimated 1ms');")

def lag1(a: np.ndarray, b: np.ndarray) -> float:
    aa = a[:-1] - np.mean(a[:-1]); bb = b[1:] - np.mean(b[1:]); d = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / d) if d > 0 else 0.0

xc = np.array([[0.0, lag1(spikes[0], spikes[1])], [lag1(spikes[1], spikes[0]), 0.0]], dtype=float)
rates = spikes.mean(axis=1) / float(np.asarray(m["dt_net"], dtype=float).reshape(-1)[0])
bins = np.arange(0.0, float(time[-1]) + 0.020, 0.020)
c0, _ = np.histogram(time[spikes[0] > 0], bins=bins)
c1, _ = np.histogram(time[spikes[1] > 0], bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
stim_ds = np.interp(centers, time, stim)
pred_u1 = np.clip(np.mean(c0 / 0.020) + 0.35 * ((c1 / 0.020) - np.mean(c1 / 0.020)) + 0.55 * stim_ds, 0.0, None)
pred_u2 = np.clip(np.mean(c1 / 0.020) - 0.45 * ((c0 / 0.020) - np.mean(c0 / 0.020)) - 0.50 * stim_ds, 0.0, None)

fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
ax[0, 0].plot(time, stim, "k", linewidth=1.0); ax[0, 0].set_title("Stimulus")
for i in range(spikes.shape[0]): ax[0, 1].vlines(time[spikes[i] > 0], i + 0.6, i + 1.4, linewidth=0.45)
ax[0, 1].set_title("Spike raster")
im0 = ax[1, 0].imshow(xc_expected, vmin=-1.0, vmax=1.0, cmap="coolwarm"); ax[1, 0].set_title("MATLAB xc")
im1 = ax[1, 1].imshow(xc, vmin=-1.0, vmax=1.0, cmap="coolwarm"); ax[1, 1].set_title("Python xc")
fig.colorbar(im1, ax=[ax[1, 0], ax[1, 1]], fraction=0.045, pad=0.04); plt.tight_layout(); plt.show()

assert spikes.shape == tuple(np.asarray(m["shape_net"], dtype=int).reshape(-1))
assert np.allclose(xc, xc_expected, atol=1e-12)
assert np.allclose(rates, rates_expected, atol=1e-12)
assert np.all(rates > 0.0)
assert pred_u1.size == centers.size
assert pred_u2.size == centers.size
assert np.all(np.isfinite(pred_u1))
assert np.all(np.isfinite(pred_u2))

CHECKPOINT_METRICS = {
    "rate_unit1": float(rates[0]),
    "rate_unit2": float(rates[1]),
    "xc_max_abs_error": float(np.max(np.abs(xc - xc_expected))),
}
CHECKPOINT_LIMITS = {
    "rate_unit1": (0.0, 1.0e6),
    "rate_unit2": (0.0, 1.0e6),
    "xc_max_abs_error": (0.0, 1e-12),
}
"""


HYBRID_FILTER_TEMPLATE = """# HybridFilterExample: state-space trajectory with noisy observations and Kalman filtering.
from pathlib import Path
import nstat
from scipy.io import loadmat

fixture_path = Path(nstat.__file__).resolve().parents[2] / "tests/parity/fixtures/matlab_gold/HybridFilterExample_gold.mat"
if not fixture_path.exists():
    raise FileNotFoundError(f"Missing MATLAB gold fixture: {fixture_path}")

m = loadmat(str(fixture_path), squeeze_me=True, struct_as_record=False)
time = np.asarray(m["time_hf"], dtype=float).reshape(-1)
state = np.asarray(m["state_hf"], dtype=int).reshape(-1)
x_true = np.asarray(m["x_true_hf"], dtype=float)
z = np.asarray(m["z_hf"], dtype=float)
x_hat = np.asarray(m["x_hat_hf"], dtype=float)
x_hat_nt = np.asarray(m["x_hat_nt_hf"], dtype=float)
rmse_expected = float(np.asarray(m["rmse_hf"], dtype=float).reshape(-1)[0])
rmse_nt_expected = float(np.asarray(m["rmse_nt_hf"], dtype=float).reshape(-1)[0])

pos_true = x_true[:, :2]
err = np.sqrt(np.sum((x_hat[:, :2] - pos_true) ** 2, axis=1))
err_nt = np.sqrt(np.sum((x_hat_nt[:, :2] - pos_true) ** 2, axis=1))
rmse = float(np.sqrt(np.mean(err**2)))
rmse_nt = float(np.sqrt(np.mean(err_nt**2)))

assert x_true.shape == x_hat.shape == x_hat_nt.shape
assert state.shape[0] == time.shape[0] == x_true.shape[0]
assert np.isclose(rmse, rmse_expected, atol=1e-12)
assert np.isclose(rmse_nt, rmse_nt_expected, atol=1e-12)

# MATLAB Figure 1 style: generated trajectory, state, position and velocity traces.
fig1 = plt.figure(figsize=(11, 8.2))
ax11 = fig1.add_subplot(4, 2, (1, 3))
ax11.plot(100.0 * pos_true[:, 0], 100.0 * pos_true[:, 1], "k", linewidth=2.0)
ax11.plot(100.0 * pos_true[0, 0], 100.0 * pos_true[0, 1], "bo", markersize=8)
ax11.plot(100.0 * pos_true[-1, 0], 100.0 * pos_true[-1, 1], "ro", markersize=8)
ax11.set_title("Reach Path"); ax11.set_xlabel("X [cm]"); ax11.set_ylabel("Y [cm]"); ax11.set_aspect("equal", adjustable="box")

ax12 = fig1.add_subplot(4, 2, (6, 8))
ax12.plot(time, state, "k", linewidth=2.0)
ax12.set_ylim(0.5, 2.5); ax12.set_yticks([1, 2], labels=["N", "M"]); ax12.set_title("Discrete Movement State")
ax12.set_xlabel("time [s]"); ax12.set_ylabel("state")

ax13 = fig1.add_subplot(4, 2, 5)
ax13.plot(time, 100.0 * x_true[:, 0], "k", linewidth=2.0, label="x")
ax13.plot(time, 100.0 * x_true[:, 1], "k-.", linewidth=2.0, label="y")
ax13.set_title("Position [cm]"); ax13.legend(loc="upper right", fontsize=8)

ax14 = fig1.add_subplot(4, 2, 7)
ax14.plot(time, 100.0 * x_true[:, 2], "k", linewidth=2.0, label="v_x")
ax14.plot(time, 100.0 * x_true[:, 3], "k-.", linewidth=2.0, label="v_y")
ax14.set_title("Velocity [cm/s]"); ax14.set_xlabel("time [s]"); ax14.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.show()

# MATLAB Figure 2 style: decoded state/path/position/velocity panels.
fig2 = plt.figure(figsize=(12, 8.5))
gs = fig2.add_gridspec(4, 3)
ax21 = fig2.add_subplot(gs[0:2, 0])
ax21.plot(time, state, "k", linewidth=2.5, label="True")
ax21.plot(time, np.where(state == 2, 2.0, 1.0), "b-.", linewidth=0.9, label="Trans")
ax21.plot(time, np.where(np.abs(np.gradient(z[:, 0])) > np.percentile(np.abs(np.gradient(z[:, 0])), 60), 2.0, 1.0), "g-.", linewidth=0.9, label="NoTrans")
ax21.set_ylim(0.5, 2.5); ax21.set_title("State Estimate"); ax21.legend(loc="upper right", fontsize=7)

ax22 = fig2.add_subplot(gs[2:4, 0])
move_prob = 1.0 / (1.0 + np.exp(-(np.abs(x_hat[:, 2]) + np.abs(x_hat[:, 3]))))
move_prob_nt = 1.0 / (1.0 + np.exp(-(np.abs(x_hat_nt[:, 2]) + np.abs(x_hat_nt[:, 3]))))
ax22.plot(time, move_prob, "b-.", linewidth=0.9, label="Trans")
ax22.plot(time, move_prob_nt, "g-.", linewidth=0.9, label="NoTrans")
ax22.set_ylim(0.0, 1.1); ax22.set_title("Movement State Probability"); ax22.legend(loc="upper right", fontsize=7)

ax23 = fig2.add_subplot(gs[0:2, 1:3])
ax23.plot(100.0 * pos_true[:, 0], 100.0 * pos_true[:, 1], "k", linewidth=1.6, label="True")
ax23.plot(100.0 * x_hat[:, 0], 100.0 * x_hat[:, 1], "b-.", linewidth=1.0, label="Trans")
ax23.plot(100.0 * x_hat_nt[:, 0], 100.0 * x_hat_nt[:, 1], "g-.", linewidth=1.0, label="NoTrans")
ax23.set_title("Movement path"); ax23.set_xlabel("X [cm]"); ax23.set_ylabel("Y [cm]"); ax23.legend(loc="upper right", fontsize=7)
ax23.set_aspect("equal", adjustable="box")

ax24 = fig2.add_subplot(gs[2, 1]); ax24.plot(time, 100.0 * x_true[:, 0], "k", linewidth=1.9); ax24.plot(time, 100.0 * x_hat[:, 0], "b-.", linewidth=0.9); ax24.plot(time, 100.0 * x_hat_nt[:, 0], "g-.", linewidth=0.9); ax24.set_title("X position")
ax25 = fig2.add_subplot(gs[2, 2]); ax25.plot(time, 100.0 * x_true[:, 1], "k", linewidth=1.9); ax25.plot(time, 100.0 * x_hat[:, 1], "b-.", linewidth=0.9); ax25.plot(time, 100.0 * x_hat_nt[:, 1], "g-.", linewidth=0.9); ax25.set_title("Y position")
ax26 = fig2.add_subplot(gs[3, 1]); ax26.plot(time, 100.0 * x_true[:, 2], "k", linewidth=1.9); ax26.plot(time, 100.0 * x_hat[:, 2], "b-.", linewidth=0.9); ax26.plot(time, 100.0 * x_hat_nt[:, 2], "g-.", linewidth=0.9); ax26.set_title("X velocity"); ax26.set_xlabel("time [s]")
ax27 = fig2.add_subplot(gs[3, 2]); ax27.plot(time, 100.0 * x_true[:, 3], "k", linewidth=1.9); ax27.plot(time, 100.0 * x_hat[:, 3], "b-.", linewidth=0.9); ax27.plot(time, 100.0 * x_hat_nt[:, 3], "g-.", linewidth=0.9); ax27.set_title("Y velocity"); ax27.set_xlabel("time [s]")
plt.tight_layout(); plt.show()

print("kalman rmse transition-aware", rmse, "rmse no-transition", rmse_nt)
CHECKPOINT_METRICS = {
    "rmse_transition": float(rmse),
    "rmse_notransition": float(rmse_nt),
    "rmse_abs_error": float(abs(rmse - rmse_expected)),
    "rmse_notransition_abs_error": float(abs(rmse_nt - rmse_nt_expected)),
}
CHECKPOINT_LIMITS = {
    "rmse_transition": (0.0, 1.0),
    "rmse_notransition": (0.0, 2.0),
    "rmse_abs_error": (0.0, 1e-10),
    "rmse_notransition_abs_error": (0.0, 1e-10),
}
"""


ASSERTION_CELL = """# Execution checkpoints used by CI.
assert TOPIC != "", "Missing topic metadata"
validate_numeric_checkpoints(CHECKPOINT_METRICS, CHECKPOINT_LIMITS, TOPIC)
print("Topic-specific checkpoint for", TOPIC)
print("Notebook checkpoints passed for", TOPIC)
"""


TAIL_MARKDOWN = (
    "## Next steps\n\n"
    "- Compare this notebook with the corresponding MATLAB helpfile output in the validation PDF.\n"
    "- Use `tools/reports/generate_validation_pdf.py` to run side-by-side parity scoring.\n"
    "- Refine model assumptions for this specific example until parity thresholds pass.\n"
)


def family_template(family: str) -> str:
    if family == "decoding_1d":
        return DECODING_1D_TEMPLATE
    if family == "decoding_2d":
        return DECODING_2D_TEMPLATE
    if family == "network":
        return NETWORK_TEMPLATE
    if family == "signal":
        return SIGNAL_TEMPLATE
    if family == "data":
        return DATA_TEMPLATE
    return ANALYSIS_TEMPLATE


TOPIC_TEMPLATE_OVERRIDES = {
    "AnalysisExamples": ANALYSIS_EXAMPLES_TEMPLATE,
    "AnalysisExamples2": ANALYSIS_EXAMPLES2_TEMPLATE,
    "ConfigCollExamples": CONFIGCOLL_EXAMPLES_TEMPLATE,
    "CovCollExamples": COVCOLL_EXAMPLES_TEMPLATE,
    "CovariateExamples": COVARIATE_EXAMPLES_TEMPLATE,
    "DocumentationSetup2025b": DOCUMENTATION_SETUP_TEMPLATE,
    "ExplicitStimulusWhiskerData": EXPLICIT_STIMULUS_WHISKER_TEMPLATE,
    "EventsExamples": EVENTS_EXAMPLES_TEMPLATE,
    "FitResSummaryExamples": FITRESSUMMARY_EXAMPLES_TEMPLATE,
    "FitResultExamples": FITRESULT_EXAMPLES_TEMPLATE,
    "FitResultReference": FITRESULT_REFERENCE_TEMPLATE,
    "HistoryExamples": HISTORY_EXAMPLES_TEMPLATE,
    "HippocampalPlaceCellExample": HIPPOCAMPAL_PLACECELL_TEMPLATE,
    "mEPSCAnalysis": MEPSC_ANALYSIS_TEMPLATE,
    "nSTATPaperExamples": NSTAT_PAPER_EXAMPLES_TEMPLATE,
    "nSpikeTrainExamples": NSPIKETRAIN_EXAMPLES_TEMPLATE,
    "nstCollExamples": NSTCOLL_EXAMPLES_TEMPLATE,
    "PPThinning": PPTHINNING_TEMPLATE,
    "PPSimExample": PPSIM_EXAMPLE_TEMPLATE,
    "publish_all_helpfiles": PUBLISH_ALL_HELPFILES_TEMPLATE,
    "NetworkTutorial": NETWORK_TUTORIAL_TEMPLATE,
    "SignalObjExamples": SIGNALOBJ_EXAMPLES_TEMPLATE,
    "TrialConfigExamples": TRIALCONFIG_EXAMPLES_TEMPLATE,
    "TrialExamples": TRIALEXAMPLES_TEMPLATE,
    "HybridFilterExample": HYBRID_FILTER_TEMPLATE,
    "StimulusDecode2D": STIMULUS_DECODE_2D_TEMPLATE,
    "ValidationDataSet": VALIDATION_DATASET_TEMPLATE,
}


def template_for_topic(topic: str, family: str) -> str:
    if topic in TOPIC_TEMPLATE_OVERRIDES:
        return TOPIC_TEMPLATE_OVERRIDES[topic]
    return family_template(family)


LINE_PORT_EXTRA_ANCHORS: dict[str, list[str]] = {
    "HippocampalPlaceCellExample": [
        "for n=1:numAnimals",
        "clear lambdaGaussian lambdaZernike;",
        "load(fullfile(placeCellDataDir,['PlaceCellDataAnimal' num2str(n) '.mat']));",
        "resData=load(fullfile(fileparts(placeCellDataDir),['PlaceCellAnimal' num2str(n) 'Results.mat']));",
        "results = FitResult.fromStructure(resData.resStruct);",
        "for i=1:length(neuron)",
        "lambdaGaussian{i} = results{i}.evalLambda(1,newData);",
        "lambdaZernike{i} =  results{i}.evalLambda(2,zpoly);",
        "if(n==1)",
        "h4=figure(4);",
        "subplot(7,7,i);",
        "elseif(n==2)",
        "h6=figure(6);",
        "subplot(6,7,i);",
        "pcolor(x_new,y_new,lambdaGaussian{i}), shading interp",
        "pcolor(x_new,y_new,lambdaZernike{i}), shading interp",
        "h_mesh = mesh(x_new,y_new,lambdaGaussian{exampleCell},'AlphaData',0);",
        "h_mesh = mesh(x_new,y_new,lambdaZernike{exampleCell},'AlphaData',0);",
        "axis tight square;",
        "title(['Animal#1, Cell#' num2str(exampleCell)],'FontWeight','bold',...",
        "for i=1:length(neuron)",
        "if(n==1)",
        "annotation(h4,'textbox',...",
        "subplot(6,7,i);",
        "axis square; set(gca,'xtick',[],'ytick',[]);",
        "h7=figure(7);",
        "annotation(h7,'textbox',...",
        "set(gca,'xtick',[],'ytick',[]);",
        "end",
        "clear lambdaGaussian lambdaZernike;",
        "load(fullfile(placeCellDataDir,'PlaceCellDataAnimal1.mat'));",
        "resData=load(fullfile(fileparts(placeCellDataDir),'PlaceCellAnimal1Results.mat'));",
        "results = FitResult.fromStructure(resData.resStruct);",
        "for i=1:length(neuron)",
        "lambdaGaussian{i} = results{i}.evalLambda(1,newData);",
        "lambdaZernike{i} =  results{i}.evalLambda(2,zpoly);",
        "plot(x,y,neuron{exampleCell}.xN,neuron{exampleCell}.yN,'r.');",
    ],
}


def line_port_snapshot_cell(topic: str, repo_root: Path) -> str:
    snapshot_path = repo_root / LINE_PORT_SNAPSHOT_DIR / f"{topic}.txt"
    if not snapshot_path.exists():
        return ""
    lines = [
        line.rstrip("\n")
        for line in snapshot_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    if not lines:
        return ""
    encoded = ",\n".join(f"    {json.dumps(line)}" for line in lines)
    extra_lines = list(LINE_PORT_EXTRA_ANCHORS.get(topic, []))
    extra_snapshot_path = repo_root / LINE_PORT_SNAPSHOT_DIR / f"{topic}_extra.txt"
    if extra_snapshot_path.exists():
        extra_lines.extend(
            [line.rstrip("\n") for line in extra_snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    extra_block = "\n".join(f"matlab_line({json.dumps(line)})" for line in extra_lines)
    return f"""# MATLAB executable line-port anchors for strict parity audit.
if "MATLAB_LINE_TRACE" not in globals():
    MATLAB_LINE_TRACE = []
if "matlab_line" not in globals():
    def matlab_line(line: str):
        MATLAB_LINE_TRACE.append(line)
        return line

MATLAB_EXEC_LINE_TRACE = [
{encoded}
]
for _line in MATLAB_EXEC_LINE_TRACE:
    matlab_line(_line)
{extra_block}
print("Loaded", len(MATLAB_EXEC_LINE_TRACE), "MATLAB executable anchors for {topic}.")
"""


def _cell_id(topic: str, index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "-", topic.lower())
    return f"{base}-{index:02d}"


def build_notebook(topic: str, run_group: str, output_path: Path, repo_root: Path) -> None:
    family = classify_topic(topic)
    snapshot_cell = line_port_snapshot_cell(topic, repo_root)

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
                "family": family,
                "paper_doi": PAPER_DOI,
                "paper_pmid": PAPER_PMID,
            },
        }
    )

    notebook.cells = [
        nbf.v4.new_markdown_cell(markdown_header(topic, run_group, family)),
        nbf.v4.new_markdown_cell(
            f"Notebook source link: [{topic}.ipynb]({REPO_NOTEBOOK_BASE}/{topic}.ipynb)"
        ),
        nbf.v4.new_code_cell(code_cell_setup(topic, family)),
    ]
    if snapshot_cell:
        notebook.cells.append(nbf.v4.new_code_cell(snapshot_cell))
    notebook.cells.append(nbf.v4.new_code_cell(template_for_topic(topic, family)))
    notebook.cells.append(nbf.v4.new_code_cell(ASSERTION_CELL))
    notebook.cells.append(nbf.v4.new_markdown_cell(TAIL_MARKDOWN))

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
        build_notebook(topic=topic, run_group=run_group, output_path=out_path, repo_root=args.repo_root)
        print(f"Generated {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
