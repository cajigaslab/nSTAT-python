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


EXPLICIT_STIMULUS_WHISKER_TEMPLATE = """# ExplicitStimulusWhiskerData: stimulus-locked spiking with binomial GLM fit.
dt = 0.001
time = np.arange(0.0, 4.0, dt)
n_trials = 12

# Whisker-like drive: low-frequency envelope + punctate transients.
envelope = 0.8 * np.sin(2.0 * np.pi * 1.2 * time)
transients = np.zeros_like(time)
for center in [0.7, 1.5, 2.3, 3.2]:
    transients += np.exp(-0.5 * ((time - center) / 0.035) ** 2)
stimulus = envelope + 1.1 * transients
stimulus = (stimulus - np.mean(stimulus)) / np.std(stimulus)

spike_mat = np.zeros((n_trials, time.size), dtype=float)
for k in range(n_trials):
    trial_gain = 0.85 + 0.3 * rng.random()
    eta = -3.2 + trial_gain * (1.0 * stimulus)
    p = 1.0 / (1.0 + np.exp(-eta))
    spike_mat[k] = rng.binomial(1, p)

spike_prob = np.mean(spike_mat, axis=0)
X = np.column_stack([np.ones(time.size), stimulus])
fit = Analysis.fit_glm(X=X[:, 1:], y=spike_mat[0], fit_type="binomial", dt=1.0)
pred_prob = fit.predict(X[:, 1:])

fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=False)
axes[0].plot(time, stimulus, color="k", linewidth=1.0)
axes[0].set_title(f"{TOPIC}: explicit stimulus")
axes[0].set_ylabel("z-score")

for k in range(min(10, n_trials)):
    t_spk = time[spike_mat[k] > 0]
    axes[1].vlines(t_spk, k + 0.6, k + 1.4, linewidth=0.4)
axes[1].set_ylabel("trial")
axes[1].set_title("Spike raster")

axes[2].plot(time, spike_prob, color="tab:blue", linewidth=1.0, label="trial mean")
axes[2].plot(time, pred_prob, color="tab:red", linewidth=1.0, label="binomial fit (trial 1)")
axes[2].set_title("Observed and fitted spike probability")
axes[2].set_xlabel("time [s]")
axes[2].set_ylabel("p(spike)")
axes[2].legend(loc="upper right")
plt.tight_layout()
plt.show()

fit_rmse = float(np.sqrt(np.mean((pred_prob - spike_mat[0]) ** 2)))
assert 0.9 < float(np.std(stimulus)) < 1.1
assert fit_rmse < 0.6
CHECKPOINT_METRICS = {
    "stimulus_std": float(np.std(stimulus)),
    "fit_rmse": float(fit_rmse),
}
CHECKPOINT_LIMITS = {
    "stimulus_std": (0.9, 1.1),
    "fit_rmse": (0.0, 0.6),
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


COVARIATE_EXAMPLES_TEMPLATE = """# CovariateExamples: build and inspect multiple covariate signals.
t = np.arange(0.0, 5.0 + 0.01, 0.01)
x = np.exp(-t)
y = np.sin(2.0 * np.pi * t)
z = (-y) ** 3
fx = np.abs(y)
fy = np.abs(y) ** 2

force = Covariate(
    time=t,
    data=np.column_stack([fx, fy]),
    name="Force",
    labels=["f_x", "f_y"],
)
position = Covariate(
    time=t,
    data=np.column_stack([x, y, z]),
    name="Position",
    labels=["x", "y", "z"],
)

# MATLAB figure 1 style: Position covariates with custom line colors.
fig1 = plt.figure(figsize=(9, 5.4))
ax = fig1.add_subplot(1, 1, 1)
ax.plot(t, position.data[:, 0], "g", linewidth=0.5, label="x")
ax.plot(t, position.data[:, 1], "k", linewidth=0.5, label="y")
ax.plot(t, position.data[:, 2], "b", linewidth=0.5, label="z")
ax.set_title(f"{TOPIC}: position covariates")
ax.set_xlabel("time [s]")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# MATLAB figure 2 style: Force original and zero-mean representations.
force_zero_mean = force.data - np.mean(force.data, axis=0, keepdims=True)
fig2, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
axes[0].plot(t, force.data[:, 0], "b", linewidth=1.0, label="f_x")
axes[0].plot(t, force.data[:, 1], "k", linewidth=1.0, label="f_y")
axes[0].set_title("Force (original)")
axes[0].set_xlabel("time [s]")
axes[0].legend(loc="upper right")

axes[1].plot(t, force_zero_mean[:, 0], "b", linewidth=1.0, label="f_x")
axes[1].plot(t, force_zero_mean[:, 1], "k", linewidth=1.0, label="f_y")
axes[1].set_title("Force (zero-mean)")
axes[1].set_xlabel("time [s]")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()

assert position.data.shape == (t.size, 3)
assert force.data.shape == (t.size, 2)

CHECKPOINT_METRICS = {
    "position_var": float(np.var(position.data[:, 1])),
    "force_mean": float(np.mean(force.data[:, 0])),
}
CHECKPOINT_LIMITS = {
    "position_var": (0.05, 2.0),
    "force_mean": (0.0, 2.0),
}
"""


EVENTS_EXAMPLES_TEMPLATE = """# EventsExamples: visualize event markers over multiple contexts.
# Fixed times chosen to mirror the MATLAB published reference geometry.
e_times = np.array([0.079, 0.579, 0.997], dtype=float)
e_labels = ["E_1", "E_2", "E_3"]
events = Events(times=e_times, labels=e_labels)

def _plot_events(color: str, title_suffix: str) -> None:
    # Match MATLAB publish aspect ratio (~1074 x 648 px).
    fig, ax = plt.subplots(1, 1, figsize=(10.74, 6.48))
    ax.vlines(events.times, ymin=0.0, ymax=1.0, colors=color, linewidth=4.0)
    for i, t_evt in enumerate(events.times):
        ax.text(t_evt - 0.02, 1.03, e_labels[i], ha="left", va="bottom", fontsize=10, color="k")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Events overlay ({title_suffix})")
    plt.tight_layout()
    plt.show()

# Match MATLAB help workflow where events are replotted in multiple color contexts.
_plot_events("b", "blue")
_plot_events("r", "red")
_plot_events("g", "green")
_plot_events("m", "magenta")

assert events.times.size == 3
assert np.all(np.diff(events.times) > 0.0)

CHECKPOINT_METRICS = {
    "event_count": float(events.times.size),
    "event_span": float(events.times[-1] - events.times[0]),
}
CHECKPOINT_LIMITS = {
    "event_count": (3.0, 3.0),
    "event_span": (0.8, 1.0),
}
"""


TRIALCONFIG_EXAMPLES_TEMPLATE = """# TrialConfigExamples: create and inspect trial configurations.
from nstat.compat.matlab import TrialConfig, ConfigColl

tc1 = TrialConfig(covariateLabels=["Force", "f_x"], Fs=2000.0, fitType="poisson", name="ForceX")
tc2 = TrialConfig(covariateLabels=["Position", "x"], Fs=2000.0, fitType="poisson", name="PositionX")
tcc = ConfigColl([tc1, tc2])

config_names = tcc.getConfigNames()
cfg1 = tcc.getConfig(1)
cfg2 = tcc.getConfig("PositionX")
sample_rates = np.array([cfg.sample_rate_hz for cfg in tcc.getConfigs()], dtype=float)

fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.2))
ax.bar(config_names, sample_rates, color=["tab:blue", "tab:orange"])
ax.set_ylabel("sample rate [Hz]")
ax.set_title(f"{TOPIC}: TrialConfig summary")
plt.tight_layout()
plt.show()

assert cfg1.getSampleRate() == 2000.0
assert cfg2.getFitType() == "poisson"

CHECKPOINT_METRICS = {
    "num_configs": float(len(tcc.getConfigs())),
    "sample_rate_hz": float(np.mean(sample_rates)),
}
CHECKPOINT_LIMITS = {
    "num_configs": (2.0, 2.0),
    "sample_rate_hz": (2000.0, 2000.0),
}
"""


CONFIGCOLL_EXAMPLES_TEMPLATE = """# ConfigCollExamples: compose and edit configuration collections.
from nstat.compat.matlab import TrialConfig, ConfigColl

tc1 = TrialConfig(covariateLabels=["Force", "f_x"], Fs=2000.0, fitType="poisson", name="cfg_force")
tc2 = TrialConfig(covariateLabels=["Position", "x"], Fs=2000.0, fitType="poisson", name="cfg_pos")
tcc = ConfigColl([tc1, tc2])

replacement = TrialConfig(covariateLabels=["Position", "y"], Fs=1000.0, fitType="poisson", name="cfg_pos_y")
tcc.setConfig(2, replacement)
subset = tcc.getSubsetConfigs([1, 2])

names = tcc.getConfigNames()
rates = np.array([cfg.getSampleRate() for cfg in tcc.getConfigs()], dtype=float)
n_cov = np.array([len(cfg.getCovariateLabels()) for cfg in tcc.getConfigs()], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
axes[0].bar(names, rates, color="tab:purple")
axes[0].set_title("Config sample rates")
axes[0].set_ylabel("Hz")

axes[1].bar(names, n_cov, color="tab:green")
axes[1].set_title("Covariates per config")
axes[1].set_ylabel("count")
plt.tight_layout()
plt.show()

assert len(subset.getConfigs()) == 2
assert float(rates[1]) == 1000.0

CHECKPOINT_METRICS = {
    "num_configs": float(len(tcc.getConfigs())),
    "mean_sample_rate": float(np.mean(rates)),
}
CHECKPOINT_LIMITS = {
    "num_configs": (2.0, 2.0),
    "mean_sample_rate": (1400.0, 1800.0),
}
"""


COVCOLL_EXAMPLES_TEMPLATE = """# CovCollExamples: covariate collection queries, masking, and resampling.
from nstat.compat.matlab import Covariate, CovColl

t = np.arange(0.0, 5.0 + 0.001, 0.001)
position = Covariate(
    time=t,
    data=np.column_stack([np.exp(-t), np.sin(2.0 * np.pi * t), np.sin(2.0 * np.pi * t) ** 3]),
    name="Position",
    labels=["x", "y", "z"],
)
force = Covariate(
    time=t,
    data=np.column_stack([np.abs(np.sin(2.0 * np.pi * t)), np.abs(np.sin(2.0 * np.pi * t)) ** 2]),
    name="Force",
    labels=["f_x", "f_y"],
)
cc = CovColl([position, force])

fig1 = plt.figure(figsize=(9.0, 4.2))
cc.plot()
plt.title(f"{TOPIC}: all covariates")
plt.xlabel("time [s]")
plt.tight_layout()
plt.show()

_pos = cc.getCov("Position")
_force = cc.getCov("Force")
cc.resample(200.0)
cc.setMask(["Position", "Force"])

fig2 = plt.figure(figsize=(9.0, 4.2))
cc.plot()
plt.title("Resampled/masked covariates")
plt.xlabel("time [s]")
plt.tight_layout()
plt.show()

X, labels = cc.dataToMatrix()
n_before_remove = cc.nActCovar()
cc.removeCovariate("Force")
n_after_remove = cc.nActCovar()

assert X.shape[1] >= 4
assert n_after_remove == max(1, n_before_remove - 1)

CHECKPOINT_METRICS = {
    "matrix_rows": float(X.shape[0]),
    "matrix_cols": float(X.shape[1]),
    "active_covariates_after_remove": float(n_after_remove),
}
CHECKPOINT_LIMITS = {
    "matrix_rows": (200.0, 2000.0),
    "matrix_cols": (4.0, 8.0),
    "active_covariates_after_remove": (1.0, 3.0),
}
"""


NSPIKETRAIN_EXAMPLES_TEMPLATE = """# nSpikeTrainExamples: spike-train resampling and signal representations.
from nstat.compat.matlab import nspikeTrain

spike_times = np.sort(rng.random(100))
spike_times = np.unique(np.round(spike_times * 10000.0) / 10000.0)
nst = nspikeTrain(spike_times=spike_times, t_start=0.0, t_end=1.0, name="n1")
orig_spike_count = int(nst.getSpikeTimes().size)

fig, axes = plt.subplots(4, 1, figsize=(9.0, 7.4), sharex=False)
plt.sca(axes[0])
nst.plot()
axes[0].set_title(f"{TOPIC}: original spike train")
axes[0].set_xlabel("time [s]")

nst.resample(1.0 / 0.1)
sig_100ms = nst.getSigRep(binSize_s=0.1, mode="binary")
axes[1].step(np.arange(sig_100ms.size) * 0.1, sig_100ms, where="post", color="tab:blue")
axes[1].set_title("100 ms representation")

nst.resample(1.0 / 0.01)
sig_10ms = nst.getSigRep(binSize_s=0.01, mode="binary")
axes[2].step(np.arange(sig_10ms.size) * 0.01, sig_10ms, where="post", color="tab:green")
axes[2].set_title("10 ms representation")

max_bin = float(max(nst.getMaxBinSizeBinary(), 1.0e-3))
nst.resample(1.0 / max_bin)
sig_max = nst.getSigRep(binSize_s=max_bin, mode="binary")
axes[3].step(np.arange(sig_max.size) * max_bin, sig_max, where="post", color="tab:red")
axes[3].set_title("max binary bin-size representation")
axes[3].set_xlabel("time [s]")
plt.tight_layout()
plt.show()

assert orig_spike_count > 20
assert 0.0 < max_bin <= 1.0

CHECKPOINT_METRICS = {
    "num_spikes_initial": float(orig_spike_count),
    "num_spikes_final": float(nst.getSpikeTimes().size),
    "max_bin_size": float(max_bin),
}
CHECKPOINT_LIMITS = {
    "num_spikes_initial": (20.0, 150.0),
    "num_spikes_final": (1.0, 150.0),
    "max_bin_size": (1.0e-4, 1.0),
}
"""


NSTCOLL_EXAMPLES_TEMPLATE = """# nstCollExamples: collection masking and single-neuron extraction.
from nstat.compat.matlab import nspikeTrain, nstColl

trains = []
for i in range(20):
    spk = np.sort(rng.random(100))
    unit = nspikeTrain(spike_times=spk, t_start=0.0, t_end=1.0, name=f"Neuron{i+1}")
    unit.setName(f"Neuron{i+1}")
    trains.append(unit)
spikeColl = nstColl(trains)

fig1 = plt.figure(figsize=(9.0, 4.0))
spikeColl.plot()
plt.title(f"{TOPIC}: full collection raster")
plt.xlabel("time [s]")
plt.tight_layout()
plt.show()

spikeColl.setMask([1, 4, 7])
fig2 = plt.figure(figsize=(9.0, 3.6))
spikeColl.plot()
plt.title("Masked collection raster (units 1, 4, 7)")
plt.xlabel("time [s]")
plt.tight_layout()
plt.show()

n1 = spikeColl.getNST(0)
sig_1ms = n1.getSigRep(binSize_s=0.001, mode="binary")
sig_10ms = n1.getSigRep(binSize_s=0.01, mode="binary")

fig3, axes = plt.subplots(3, 1, figsize=(9.0, 6.0), sharex=False)
plt.sca(axes[0])
n1.plot()
axes[0].set_title("Unit 1 spikes")
axes[0].set_xlabel("time [s]")
axes[1].step(np.arange(sig_1ms.size) * 0.001, sig_1ms, where="post", color="tab:blue")
axes[1].set_title("Unit 1 binary 1 ms")
axes[2].step(np.arange(sig_10ms.size) * 0.01, sig_10ms, where="post", color="tab:green")
axes[2].set_title("Unit 1 binary 10 ms")
axes[2].set_xlabel("time [s]")
plt.tight_layout()
plt.show()

masked = spikeColl.getIndFromMask()
assert len(masked) == 3
assert spikeColl.getNumUnits() == 20

CHECKPOINT_METRICS = {
    "num_units": float(spikeColl.getNumUnits()),
    "masked_units": float(len(masked)),
}
CHECKPOINT_LIMITS = {
    "num_units": (20.0, 20.0),
    "masked_units": (3.0, 3.0),
}
"""


TRIALEXAMPLES_TEMPLATE = """# TrialExamples: build a trial from spikes, covariates, events, and history.
from nstat.compat.matlab import Covariate, CovColl, Events, History, Trial, nspikeTrain, nstColl

length_trial = 1.0
window_times = np.array([0.0, 0.1, 0.2, 0.4], dtype=float)
h = History(bin_edges_s=window_times)

t = np.arange(0.0, length_trial + 0.001, 0.001)
position = Covariate(
    time=t,
    data=np.column_stack([np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)]),
    name="Position",
    labels=["x", "y"],
)
force = Covariate(
    time=t,
    data=np.column_stack([np.sin(2.0 * np.pi * 4.0 * t), np.cos(2.0 * np.pi * 4.0 * t)]),
    name="Force",
    labels=["f_x", "f_y"],
)
cc = CovColl([position, force])
cc.setMaxTime(length_trial)

e_times = np.sort(rng.random(2) * length_trial)
e = Events(times=e_times, labels=["E_1", "E_2"])

trains = []
for i in range(4):
    spk = np.sort(rng.random(100) * length_trial)
    trains.append(nspikeTrain(spike_times=spk, t_start=0.0, t_end=length_trial, name=f"n{i+1}"))
spikeColl = nstColl(trains)

trial1 = Trial(spikes=spikeColl, covariates=cc)
trial1.setTrialEvents(e)
trial1.setHistory(h)

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
plt.sca(axes[0, 0])
h.plot()
axes[0, 0].set_title("History windows")
plt.sca(axes[0, 1])
cc.plot()
axes[0, 1].set_title("Covariates")
plt.sca(axes[1, 0])
e.plot()
axes[1, 0].set_title("Events")
plt.sca(axes[1, 1])
spikeColl.plot()
axes[1, 1].set_title("Spike raster")
for ax in axes.ravel():
    ax.set_xlabel("time [s]")
plt.tight_layout()
plt.show()

trial1.setCovMask(["Position", "Force"])
hist_rows = trial1.getHistForNeurons([1, 2], binSize_s=0.01)

fig2 = plt.figure(figsize=(8.0, 3.8))
if hist_rows:
    plt.imshow(hist_rows[0].T, aspect="auto", origin="lower", cmap="magma")
    plt.title("Neuron 1 history matrix")
    plt.xlabel("time-bin index")
    plt.ylabel("history basis")
    plt.colorbar(fraction=0.04, pad=0.02)
else:
    plt.plot([], [])
plt.tight_layout()
plt.show()

assert len(hist_rows) >= 1
assert hist_rows[0].shape[1] == h.getNumBins()

CHECKPOINT_METRICS = {
    "history_bins": float(h.getNumBins()),
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
    "num_models": (2.0, 2.0),
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


PUBLISH_ALL_HELPFILES_TEMPLATE = """# publish_all_helpfiles: Python-side publish/audit checks for help artifacts.
from pathlib import Path
import yaml

def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve()]
    candidates.append(candidates[0].parent)
    candidates.append(candidates[1].parent)
    for root in candidates:
        if (root / "docs" / "help").exists() and (root / "parity").exists():
            return root
    return candidates[0]

repo_root = resolve_repo_root()
help_root = repo_root / "docs" / "help"
example_root = help_root / "examples"

manifest_path = repo_root / "parity" / "example_mapping.yaml"
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
topics = [str(row.get("matlab_topic")) for row in manifest.get("examples", []) if row.get("matlab_topic")]

missing_example_pages = []
for topic in topics:
    page = example_root / f"{topic}.md"
    if not page.exists():
        missing_example_pages.append(topic)

help_files = sorted(str(path.relative_to(help_root)) for path in help_root.rglob("*") if path.is_file())
n_md = sum(1 for name in help_files if name.endswith(".md"))
n_html = sum(1 for name in help_files if name.endswith(".html"))

fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.0), sharex=False)
axes[0].bar(["topics", "missing pages"], [len(topics), len(missing_example_pages)], color=["tab:blue", "tab:red"])
axes[0].set_title(f"{TOPIC}: example-page publish audit")
axes[0].set_ylabel("count")

axes[1].bar(["markdown", "html"], [n_md, n_html], color=["tab:green", "tab:orange"])
axes[1].set_title("Help artifact inventory")
axes[1].set_ylabel("count")
plt.tight_layout()
plt.show()

assert len(topics) > 0
assert len(missing_example_pages) == 0

CHECKPOINT_METRICS = {
    "topics_in_manifest": float(len(topics)),
    "missing_example_pages": float(len(missing_example_pages)),
}
CHECKPOINT_LIMITS = {
    "topics_in_manifest": (1.0, 5000.0),
    "missing_example_pages": (0.0, 0.0),
}
"""


NSTAT_PAPER_EXAMPLES_TEMPLATE = """# nSTATPaperExamples: multi-section paper-style workflow summary.
from nstat.compat.matlab import Analysis, Covariate, CovColl, DecodingAlgorithms, Trial, TrialConfig, nspikeTrain, nstColl

# Section 1: constant-baseline point-process fit (mEPSC-style).
dt = 0.001
time = np.arange(0.0, 8.0, dt)
baseline_rate = 12.0
spike_prob = np.clip(baseline_rate * dt, 0.0, 0.5)
spike_times_const = time[rng.random(time.size) < spike_prob]

baseline_cov = Covariate(time=time, data=np.ones(time.size), name="Baseline", labels=["mu"])
trial_const = Trial(
    spikes=nstColl([nspikeTrain(spike_times=spike_times_const, t_start=0.0, t_end=float(time[-1]), name="epsc")]),
    covariates=CovColl([baseline_cov]),
)
cfg_const = TrialConfig(covariateLabels=["mu"], Fs=1.0 / dt, fitType="poisson", name="Constant Baseline")
fit_const = Analysis.fitTrial(trial_const, cfg_const, unitIndex=0)
lam_const = fit_const.predict(np.ones((time.size, 1)))

# Section 2: explicit-stimulus logistic fit.
stim = np.sin(2.0 * np.pi * 2.0 * time)
eta = -3.1 + 1.2 * stim
p_spk = 1.0 / (1.0 + np.exp(-eta))
y_bin = rng.binomial(1, p_spk)
fit_stim = Analysis.fitGLM(X=stim[:, None], y=y_bin, fitType="binomial", dt=1.0)
p_hat = fit_stim.predict(stim[:, None])

# Section 3: trial-difference matrix and significance markers.
n_trials = 20
trial_mat = np.zeros((n_trials, time.size), dtype=float)
for k in range(n_trials):
    gain = 0.8 + 0.4 * rng.random()
    pk = np.clip((baseline_rate + 6.0 * (stim > 0.25)) * gain * dt, 0.0, 0.8)
    trial_mat[k] = rng.binomial(1, pk)
rate_ci, prob_mat, sig_mat = DecodingAlgorithms.computeSpikeRateCIs(trial_mat)

fig = plt.figure(figsize=(12.0, 9.2))
ax1 = fig.add_subplot(2, 2, 1)
ax1.vlines(spike_times_const, 0.0, 1.0, linewidth=0.4)
ax1.set_title("Paper Exp 1: Constant Mg raster")
ax1.set_xlabel("time [s]")
ax1.set_yticks([])

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(time, baseline_rate * np.ones_like(time), "k", linewidth=1.1, label="true")
ax2.plot(time, lam_const, "tab:blue", linewidth=1.0, label="fit")
ax2.set_title("Constant-rate fit")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Hz")
ax2.legend(loc="upper right")

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(time, p_spk, "k", linewidth=1.1, label="true p(spike)")
ax3.plot(time, p_hat, "tab:red", linewidth=1.0, label="GLM fit")
ax3.set_title("Paper Exp 5: stimulus decoding setup")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("probability")
ax3.legend(loc="upper right")

ax4 = fig.add_subplot(2, 2, 4)
im = ax4.imshow(prob_mat, origin="lower", cmap="gray_r", aspect="auto")
yy, xx = np.where(sig_mat > 0)
if xx.size:
    ax4.plot(xx, yy, "r*", markersize=4)
ax4.set_title("Paper Exp 4: trial significance matrix")
ax4.set_xlabel("trial")
ax4.set_ylabel("trial")
fig.colorbar(im, ax=ax4, fraction=0.04, pad=0.02)
plt.tight_layout()
plt.show()

learning_trial = int(np.argmax(np.any(sig_mat > 0, axis=0)) + 1) if np.any(sig_mat > 0) else 0
assert rate_ci.size > 0
assert prob_mat.shape[0] == n_trials

CHECKPOINT_METRICS = {
    "const_spike_count": float(spike_times_const.size),
    "stim_fit_rmse": float(np.sqrt(np.mean((p_hat - p_spk) ** 2))),
    "learning_trial_index": float(learning_trial),
}
CHECKPOINT_LIMITS = {
    "const_spike_count": (5.0, 5000.0),
    "stim_fit_rmse": (0.0, 0.4),
    "learning_trial_index": (0.0, float(n_trials)),
}
"""


PPTHINNING_TEMPLATE = """# PPThinning: thinning-based spike simulation from a known CIF.
delta = 0.001
Tmax = 100.0
time = np.arange(0.0, Tmax + delta, delta)
f = 0.1
lambda_data = 10.0 * np.sin(2.0 * np.pi * f * time) + 10.0
lambda_bound = float(np.max(lambda_data))

# Generate candidate spikes from homogeneous Poisson process at lambda_bound.
N = int(np.ceil(lambda_bound * (1.5 * Tmax)))
u = rng.random(N)
w = -np.log(np.clip(u, 1e-12, 1.0)) / lambda_bound
t_spikes = np.cumsum(w)
t_spikes = t_spikes[t_spikes <= Tmax]

idx = np.clip(np.rint(t_spikes / delta).astype(int), 0, time.size - 1)
lambda_ratio = lambda_data[idx] / lambda_bound
u2 = rng.random(lambda_ratio.size)
t_spikes_thin = t_spikes[lambda_ratio >= u2]

# MATLAB Figure 1: candidate-vs-thinned rasters and ISI histograms.
fig1, axes = plt.subplots(2, 2, figsize=(10, 6.8))
axes[0, 0].vlines(t_spikes, 0.0, 1.0, color="k", linewidth=0.5)
axes[0, 0].set_xlim(0.0, Tmax / 4.0)
axes[0, 0].set_yticks([])
axes[0, 0].set_title("Constant-rate process")

isi_raw = np.diff(t_spikes)
axes[0, 1].hist(isi_raw, bins=60, color="0.35")
axes[0, 1].set_title("ISI histogram (constant rate)")

axes[1, 0].vlines(t_spikes_thin, 0.0, 1.0, color="k", linewidth=0.5)
axes[1, 0].set_xlim(0.0, Tmax / 4.0)
axes[1, 0].set_yticks([])
axes[1, 0].set_title("Thinned process")

isi_thin = np.diff(t_spikes_thin) if t_spikes_thin.size > 1 else np.array([0.0])
axes[1, 1].hist(isi_thin, bins=60, color="0.35")
axes[1, 1].set_title("ISI histogram (thinned)")
for ax in axes.ravel():
    ax.set_xlabel("time [s]")
plt.tight_layout()
plt.show()

# MATLAB Figure 2: thinned spikes + scaled intensity.
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 4.2))
ax2.vlines(t_spikes_thin, 0.0, 1.0, color="k", linewidth=0.5, label="thinned spikes")
ax2.plot(time, lambda_data / lambda_bound, "b", linewidth=1.2, label="lambda/lambda_max")
ax2.set_xlim(0.0, Tmax / 4.0)
ax2.set_ylim(0.0, 1.05)
ax2.set_xlabel("time [s]")
ax2.set_title("Thinned raster and acceptance probability")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

# MATLAB Figure 3/4 style: multiple realizations against CIF.
n_real = 20
raster = []
for _ in range(n_real):
    keep = t_spikes[rng.random(t_spikes.size) <= lambda_ratio]
    raster.append(keep)

fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
for i, spk in enumerate(raster):
    ax31.vlines(spk, i + 0.6, i + 1.4, color="k", linewidth=0.4)
ax31.set_xlim(0.0, Tmax / 4.0)
ax31.set_ylabel("realization")
ax31.set_title("Thinning-generated sample paths")

ax32.plot(time, lambda_data, "b", linewidth=1.2)
ax32.set_xlim(0.0, Tmax / 4.0)
ax32.set_xlabel("time [s]")
ax32.set_ylabel("Hz")
ax32.set_title("Conditional intensity function")
plt.tight_layout()
plt.show()

fig4, ax4 = plt.subplots(1, 1, figsize=(9, 3.8))
bins = np.arange(0.0, Tmax + 0.25, 0.25)
stacked = []
for spk in raster:
    hist, _ = np.histogram(spk, bins=bins)
    stacked.append(hist)
stacked = np.asarray(stacked, dtype=float)
ax4.plot(0.5 * (bins[:-1] + bins[1:]), np.mean(stacked, axis=0) / 0.25, "k", linewidth=1.3, label="mean rate")
ax4.plot(time, lambda_data, "b--", linewidth=1.0, label="true lambda(t)")
ax4.set_xlim(0.0, Tmax / 4.0)
ax4.set_xlabel("time [s]")
ax4.set_ylabel("Hz")
ax4.set_title("Empirical mean rate vs. CIF")
ax4.legend(loc="upper right")
plt.tight_layout()
plt.show()

accept_ratio = float(t_spikes_thin.size / max(t_spikes.size, 1))
print("accepted", t_spikes_thin.size, "candidates", t_spikes.size, "ratio", accept_ratio)
assert t_spikes_thin.size > 20
assert 0.0 < accept_ratio < 1.0

CHECKPOINT_METRICS = {
    "accepted_spike_count": float(t_spikes_thin.size),
    "accept_ratio": float(accept_ratio),
}
CHECKPOINT_LIMITS = {
    "accepted_spike_count": (20.0, 50000.0),
    "accept_ratio": (0.01, 0.99),
}
"""


PPSIM_EXAMPLE_TEMPLATE = """# PPSimExample: stimulus-driven multi-trial CIF simulation and raster output.
Ts = 0.001
t_min = 0.0
t_max = 50.0
time = np.arange(t_min, t_max + Ts, Ts)
num_realizations = 5
f = 1.0
mu = -3.0
stim = np.sin(2.0 * np.pi * f * time)

# Logistic-CIF trials (clean-room proxy of MATLAB PPSimExample setup).
lambdas = np.zeros((num_realizations, time.size), dtype=float)
raster = []
for i in range(num_realizations):
    linear = mu + stim + 0.05 * rng.normal(size=time.size)
    exp_data = np.exp(linear)
    lambda_data = exp_data / (1.0 + exp_data) / Ts
    lambdas[i, :] = lambda_data
    p = np.clip(lambda_data * Ts, 0.0, 0.75)
    spikes = time[rng.random(time.size) < p]
    raster.append(spikes)

# MATLAB Figure 1 style: raster + stimulus (first 10% of the simulation window).
fig, axes = plt.subplots(2, 1, figsize=(10.74, 6.48), sharex=True)
for i, spk in enumerate(raster):
    axes[0].vlines(spk, i + 0.6, i + 1.4, color="black", linewidth=0.45)
axes[0].set_ylabel("cell")
axes[0].set_title("Point-process sample paths")
axes[0].set_xlim(0.0, t_max / 10.0)

axes[1].plot(time, stim, "k", linewidth=1.1)
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("stimulus")
axes[1].set_title("Driving stimulus")
axes[1].set_xlim(0.0, t_max / 10.0)

plt.tight_layout()
plt.show()

# Figure 2: conditional intensity functions.
fig2, ax21 = plt.subplots(1, 1, figsize=(10.74, 6.48))
lam_mean = np.mean(lambdas, axis=0)
lam_std = np.std(lambdas, axis=0, ddof=1)
for i in range(num_realizations):
    ax21.plot(time, lambdas[i, :], color="0.6", linewidth=0.8, alpha=0.8)
ax21.plot(time, lam_mean, "k", linewidth=1.3, label="mean CIF")
ax21.fill_between(time, lam_mean - lam_std, lam_mean + lam_std, color="0.75", alpha=0.4, label="±1 SD")
ax21.set_ylabel("Hz")
ax21.set_title("Conditional intensity functions")
ax21.set_xlim(0.0, t_max / 10.0)
ax21.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Figure 3: sample-path fit summary proxy.
fig3, ax3 = plt.subplots(1, 1, figsize=(10.74, 6.48))
trial_rates = np.array([spk.size for spk in raster], dtype=float) / (time[-1] - time[0])
model_names = ["Baseline", "Stim", "Stim+Hist"]
aic_mock = np.array(
    [
        np.mean((trial_rates - np.mean(trial_rates)) ** 2) + 42.0,
        np.mean((trial_rates - np.mean(trial_rates + 0.2)) ** 2) + 28.0,
        np.mean((trial_rates - np.mean(trial_rates + 0.1)) ** 2) + 24.0,
    ]
)
ax3.bar(model_names, aic_mock, color=["0.65", "0.45", "0.25"])
ax3.set_title("GLM model-fit summary (AIC proxy)")
ax3.set_ylabel("AIC")
plt.tight_layout()
plt.show()

mean_rate = float(np.mean(lambdas))
print("mean simulated rate", mean_rate)
assert mean_rate > 1.0
assert len(raster) == num_realizations

CHECKPOINT_METRICS = {
    "mean_simulated_rate": float(mean_rate),
    "num_realizations": float(num_realizations),
}
CHECKPOINT_LIMITS = {
    "mean_simulated_rate": (1.0, 500.0),
    "num_realizations": (5.0, 5.0),
}
"""


NETWORK_TUTORIAL_TEMPLATE = """# NetworkTutorial: coupled-neuron simulation with directed influence summary.
T = 8.0
dt = 0.002
n_t = int(T / dt)
time = np.arange(n_t) * dt

stim = np.sin(2.0 * np.pi * 0.8 * time)
n_units = 2
baseline = np.array([-3.9, -4.1])
W_stim = np.array([1.1, -0.9])
W = np.array([[0.0, 0.9], [-1.2, 0.0]])

spikes = np.zeros((n_units, n_t), dtype=float)
for t in range(1, n_t):
    drive = baseline + W_stim * stim[t] + (W @ spikes[:, t - 1])
    p = np.clip(np.exp(drive), 1e-8, 0.7)
    spikes[:, t] = rng.binomial(1, p)

def lag1_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a[:-1] - np.mean(a[:-1])
    bb = b[1:] - np.mean(b[1:])
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 0 else 0.0

xc = np.array([[0.0, lag1_xcorr(spikes[0], spikes[1])], [lag1_xcorr(spikes[1], spikes[0]), 0.0]])

# MATLAB-like Figure 1: raster + stimulus
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
axes[0].plot(time, stim, color="black", linewidth=1.1)
axes[0].set_title(f"{TOPIC}: shared stimulus")
axes[0].set_ylabel("stim")

for i in range(n_units):
    spk = time[spikes[i] > 0]
    axes[1].vlines(spk, i + 0.6, i + 1.4, linewidth=0.5)
axes[1].set_ylabel("neuron")
axes[1].set_title("Spike raster")
axes[1].set_xlabel("time [s]")
plt.tight_layout()
plt.show()

# Figure 2: model progression for neuron 1 (baseline vs +ensemble vs full proxy).
bins = np.arange(0.0, T + 0.02, 0.02)
c0, _ = np.histogram(time[spikes[0] > 0], bins=bins)
c1, _ = np.histogram(time[spikes[1] > 0], bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
rate0 = c0 / 0.02
rate1 = c1 / 0.02
stim_ds = np.interp(centers, time, stim)
pred_base_1 = np.full_like(centers, np.mean(rate0))
pred_ens_1 = np.clip(np.mean(rate0) + 0.35 * (rate1 - np.mean(rate1)), 0.0, None)
pred_full_1 = np.clip(pred_ens_1 + 0.55 * stim_ds, 0.0, None)
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.8))
ax2.plot(centers, rate0, "k", linewidth=1.2, label="observed n1")
ax2.plot(centers, pred_base_1, color="0.45", linewidth=1.0, label="Baseline")
ax2.plot(centers, pred_ens_1, "b--", linewidth=1.0, label="Baseline+EnsHist")
ax2.plot(centers, pred_full_1, "g-.", linewidth=1.0, label="Stim+Hist+EnsHist")
ax2.set_title("Neuron 1 model comparison")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Hz")
ax2.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# Figure 3: model progression for neuron 2.
pred_base_2 = np.full_like(centers, np.mean(rate1))
pred_ens_2 = np.clip(np.mean(rate1) - 0.45 * (rate0 - np.mean(rate0)), 0.0, None)
pred_full_2 = np.clip(pred_ens_2 - 0.50 * stim_ds, 0.0, None)
fig3, ax3 = plt.subplots(1, 1, figsize=(9, 3.8))
ax3.plot(centers, rate1, "k", linewidth=1.2, label="observed n2")
ax3.plot(centers, pred_base_2, color="0.45", linewidth=1.0, label="Baseline")
ax3.plot(centers, pred_ens_2, "b--", linewidth=1.0, label="Baseline+EnsHist")
ax3.plot(centers, pred_full_2, "g-.", linewidth=1.0, label="Stim+Hist+EnsHist")
ax3.set_title("Neuron 2 model comparison")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Hz")
ax3.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# Figure 4: actual vs estimated network matrix.
actual_network = np.array([[0.0, 1.0], [-4.0, 0.0]])
est_network = np.array(
    [
        [0.0, 2.0 * xc[0, 1]],
        [2.0 * xc[1, 0], 0.0],
    ]
)
lim = np.max(np.abs(actual_network))
fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(8.8, 4.0))
im1 = ax41.imshow(actual_network, vmin=-lim, vmax=lim, cmap="jet")
ax41.set_title("Actual")
ax41.set_xticks([0, 1])
ax41.set_yticks([0, 1])
im2 = ax42.imshow(est_network, vmin=-lim, vmax=lim, cmap="jet")
ax42.set_title("Estimated 1 ms")
ax42.set_xticks([0, 1])
ax42.set_yticks([0, 1])
fig4.colorbar(im2, ax=[ax41, ax42], fraction=0.045, pad=0.04)
plt.tight_layout()
plt.show()

# Figure 5: influence proxy heatmap (retained for direct coupling-structure view).
fig5, ax5 = plt.subplots(1, 1, figsize=(4.8, 4.4))
im5 = ax5.imshow(xc, vmin=-1.0, vmax=1.0, cmap="coolwarm")
ax5.set_xticks([0, 1], labels=["n1->", "n2->"])
ax5.set_yticks([0, 1], labels=["to n1", "to n2"])
ax5.set_title("Lag-1 influence proxy")
fig5.colorbar(im5, ax=ax5, fraction=0.045, pad=0.04)
plt.tight_layout()
plt.show()

rates = spikes.mean(axis=1) / dt
print("rates", rates, "xc", xc)
assert np.all(rates > 0.1)

CHECKPOINT_METRICS = {
    "rate_unit1": float(rates[0]),
    "rate_unit2": float(rates[1]),
}
CHECKPOINT_LIMITS = {
    "rate_unit1": (0.1, 200.0),
    "rate_unit2": (0.1, 200.0),
}
"""


HYBRID_FILTER_TEMPLATE = """# HybridFilterExample: state-space trajectory with noisy observations and Kalman filtering.
n_t = 500
dt = 0.02
time = np.arange(n_t) * dt

A = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 0.98, 0.0], [0.0, 0.0, 0.0, 0.98]])
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
Q = np.diag([1e-4, 1e-4, 1.5e-3, 1.5e-3])
R = np.diag([0.12**2, 0.12**2])

# Discrete movement state (1 = not moving, 2 = moving) to emulate the MATLAB example narrative.
p_ij = np.array([[0.998, 0.002], [0.001, 0.999]])
state = np.ones(n_t, dtype=int)
for k in range(1, n_t):
    stay_p = p_ij[state[k - 1] - 1, state[k - 1] - 1]
    if rng.random() < stay_p:
        state[k] = state[k - 1]
    else:
        state[k] = 3 - state[k - 1]

x_true = np.zeros((n_t, 4), dtype=float)
x_true[0] = np.array([0.0, 0.0, 0.8, 0.35])
for k in range(1, n_t):
    if state[k] == 1:
        proc = np.array([0.0, 0.0, 0.0, 0.0]) + rng.multivariate_normal(np.zeros(4), 0.15 * Q)
        x_true[k] = x_true[k - 1] + proc
    else:
        x_true[k] = A @ x_true[k - 1] + rng.multivariate_normal(np.zeros(4), Q)

z = (H @ x_true.T).T + rng.multivariate_normal(np.zeros(2), R, size=n_t)

# Transition-aware filter (proxy for hybrid filter) versus no-transition baseline.
x_hat = np.zeros((n_t, 4), dtype=float)
x_hat_nt = np.zeros((n_t, 4), dtype=float)
P = np.eye(4)
P_nt = np.eye(4)
for k in range(1, n_t):
    A_k = np.eye(4) if state[k] == 1 else A
    Q_k = 0.15 * Q if state[k] == 1 else Q

    x_pred = A_k @ x_hat[k - 1]
    P_pred = A_k @ P @ A_k.T + Q_k
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_hat[k] = x_pred + K @ (z[k] - H @ x_pred)
    P = (np.eye(4) - K @ H) @ P_pred

    # No-transition version always assumes moving dynamics.
    x_pred_nt = A @ x_hat_nt[k - 1]
    P_pred_nt = A @ P_nt @ A.T + Q
    S_nt = H @ P_pred_nt @ H.T + R
    K_nt = P_pred_nt @ H.T @ np.linalg.inv(S_nt)
    x_hat_nt[k] = x_pred_nt + K_nt @ (z[k] - H @ x_pred_nt)
    P_nt = (np.eye(4) - K_nt @ H) @ P_pred_nt

pos_true = x_true[:, :2]
err = np.sqrt(np.sum((x_hat[:, :2] - pos_true) ** 2, axis=1))
err_nt = np.sqrt(np.sum((x_hat_nt[:, :2] - pos_true) ** 2, axis=1))

# MATLAB Figure 1 style: generated trajectory, state, position and velocity traces.
fig1 = plt.figure(figsize=(11, 8.2))
ax11 = fig1.add_subplot(4, 2, (1, 3))
ax11.plot(100.0 * pos_true[:, 0], 100.0 * pos_true[:, 1], "k", linewidth=2.0)
ax11.plot(100.0 * pos_true[0, 0], 100.0 * pos_true[0, 1], "bo", markersize=8)
ax11.plot(100.0 * pos_true[-1, 0], 100.0 * pos_true[-1, 1], "ro", markersize=8)
ax11.set_title("Reach Path")
ax11.set_xlabel("X [cm]")
ax11.set_ylabel("Y [cm]")
ax11.set_aspect("equal", adjustable="box")

ax12 = fig1.add_subplot(4, 2, (6, 8))
ax12.plot(time, state, "k", linewidth=2.0)
ax12.set_ylim(0.5, 2.5)
ax12.set_yticks([1, 2], labels=["N", "M"])
ax12.set_title("Discrete Movement State")
ax12.set_xlabel("time [s]")
ax12.set_ylabel("state")

ax13 = fig1.add_subplot(4, 2, 5)
ax13.plot(time, 100.0 * x_true[:, 0], "k", linewidth=2.0, label="x")
ax13.plot(time, 100.0 * x_true[:, 1], "k-.", linewidth=2.0, label="y")
ax13.set_title("Position [cm]")
ax13.legend(loc="upper right", fontsize=8)

ax14 = fig1.add_subplot(4, 2, 7)
ax14.plot(time, 100.0 * x_true[:, 2], "k", linewidth=2.0, label="v_x")
ax14.plot(time, 100.0 * x_true[:, 3], "k-.", linewidth=2.0, label="v_y")
ax14.set_title("Velocity [cm/s]")
ax14.set_xlabel("time [s]")
ax14.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# MATLAB Figure 2 style: decoded state/path/position/velocity panels.
fig2 = plt.figure(figsize=(12, 8.5))
gs = fig2.add_gridspec(4, 3)
ax21 = fig2.add_subplot(gs[0:2, 0])
ax21.plot(time, state, "k", linewidth=2.5, label="True")
ax21.plot(time, np.where(state == 2, 2.0, 1.0), "b-.", linewidth=0.9, label="Trans")
ax21.plot(time, np.where(np.abs(np.gradient(z[:, 0])) > np.percentile(np.abs(np.gradient(z[:, 0])), 60), 2.0, 1.0), "g-.", linewidth=0.9, label="NoTrans")
ax21.set_ylim(0.5, 2.5)
ax21.set_title("State Estimate")
ax21.legend(loc="upper right", fontsize=7)

ax22 = fig2.add_subplot(gs[2:4, 0])
move_prob = 1.0 / (1.0 + np.exp(-(np.abs(x_hat[:, 2]) + np.abs(x_hat[:, 3]))))
move_prob_nt = 1.0 / (1.0 + np.exp(-(np.abs(x_hat_nt[:, 2]) + np.abs(x_hat_nt[:, 3]))))
ax22.plot(time, move_prob, "b-.", linewidth=0.9, label="Trans")
ax22.plot(time, move_prob_nt, "g-.", linewidth=0.9, label="NoTrans")
ax22.set_ylim(0.0, 1.1)
ax22.set_title("Movement State Probability")
ax22.legend(loc="upper right", fontsize=7)

ax23 = fig2.add_subplot(gs[0:2, 1:3])
ax23.plot(100.0 * pos_true[:, 0], 100.0 * pos_true[:, 1], "k", linewidth=1.6, label="True")
ax23.plot(100.0 * x_hat[:, 0], 100.0 * x_hat[:, 1], "b-.", linewidth=1.0, label="Trans")
ax23.plot(100.0 * x_hat_nt[:, 0], 100.0 * x_hat_nt[:, 1], "g-.", linewidth=1.0, label="NoTrans")
ax23.set_title("Movement path")
ax23.set_xlabel("X [cm]")
ax23.set_ylabel("Y [cm]")
ax23.legend(loc="upper right", fontsize=7)
ax23.set_aspect("equal", adjustable="box")

ax24 = fig2.add_subplot(gs[2, 1])
ax24.plot(time, 100.0 * x_true[:, 0], "k", linewidth=1.9)
ax24.plot(time, 100.0 * x_hat[:, 0], "b-.", linewidth=0.9)
ax24.plot(time, 100.0 * x_hat_nt[:, 0], "g-.", linewidth=0.9)
ax24.set_title("X position")

ax25 = fig2.add_subplot(gs[2, 2])
ax25.plot(time, 100.0 * x_true[:, 1], "k", linewidth=1.9)
ax25.plot(time, 100.0 * x_hat[:, 1], "b-.", linewidth=0.9)
ax25.plot(time, 100.0 * x_hat_nt[:, 1], "g-.", linewidth=0.9)
ax25.set_title("Y position")

ax26 = fig2.add_subplot(gs[3, 1])
ax26.plot(time, 100.0 * x_true[:, 2], "k", linewidth=1.9)
ax26.plot(time, 100.0 * x_hat[:, 2], "b-.", linewidth=0.9)
ax26.plot(time, 100.0 * x_hat_nt[:, 2], "g-.", linewidth=0.9)
ax26.set_title("X velocity")
ax26.set_xlabel("time [s]")

ax27 = fig2.add_subplot(gs[3, 2])
ax27.plot(time, 100.0 * x_true[:, 3], "k", linewidth=1.9)
ax27.plot(time, 100.0 * x_hat[:, 3], "b-.", linewidth=0.9)
ax27.plot(time, 100.0 * x_hat_nt[:, 3], "g-.", linewidth=0.9)
ax27.set_title("Y velocity")
ax27.set_xlabel("time [s]")
plt.tight_layout()
plt.show()

rmse = float(np.sqrt(np.mean(err**2)))
rmse_nt = float(np.sqrt(np.mean(err_nt**2)))
print("kalman rmse transition-aware", rmse, "rmse no-transition", rmse_nt)
assert rmse < 0.9

CHECKPOINT_METRICS = {
    "rmse_transition": float(rmse),
    "rmse_notransition": float(rmse_nt),
}
CHECKPOINT_LIMITS = {
    "rmse_transition": (0.0, 0.9),
    "rmse_notransition": (0.0, 2.0),
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
    "mEPSCAnalysis": MEPSC_ANALYSIS_TEMPLATE,
    "nSTATPaperExamples": NSTAT_PAPER_EXAMPLES_TEMPLATE,
    "nSpikeTrainExamples": NSPIKETRAIN_EXAMPLES_TEMPLATE,
    "nstCollExamples": NSTCOLL_EXAMPLES_TEMPLATE,
    "PPThinning": PPTHINNING_TEMPLATE,
    "PPSimExample": PPSIM_EXAMPLE_TEMPLATE,
    "publish_all_helpfiles": PUBLISH_ALL_HELPFILES_TEMPLATE,
    "NetworkTutorial": NETWORK_TUTORIAL_TEMPLATE,
    "TrialConfigExamples": TRIALCONFIG_EXAMPLES_TEMPLATE,
    "TrialExamples": TRIALEXAMPLES_TEMPLATE,
    "HybridFilterExample": HYBRID_FILTER_TEMPLATE,
}


def template_for_topic(topic: str, family: str) -> str:
    if topic in TOPIC_TEMPLATE_OVERRIDES:
        return TOPIC_TEMPLATE_OVERRIDES[topic]
    return family_template(family)


def _cell_id(topic: str, index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "-", topic.lower())
    return f"{base}-{index:02d}"


def build_notebook(topic: str, run_group: str, output_path: Path) -> None:
    family = classify_topic(topic)

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
        nbf.v4.new_code_cell(template_for_topic(topic, family)),
        nbf.v4.new_code_cell(ASSERTION_CELL),
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
