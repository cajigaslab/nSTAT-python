#!/usr/bin/env python3
"""Export MATLAB-gold fixtures for canonical parity workflows.

This script runs MATLAB in batch mode to generate deterministic fixture files
for parity-critical workflow families:
- PPSimExample
- DecodingExampleWithHist
- HippocampalPlaceCellExample
- SpikeRateDiffCIs
- PSTHEstimation
- nstCollExamples
- TrialExamples
- CovCollExamples
- EventsExamples
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import yaml


MATLAB_SCRIPT_TEMPLATE = r"""
rng(2026,'twister');

out_dir = '{out_dir}';
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

% ------------------------------
% Fixture 1: PPSimExample (GLM)
% ------------------------------
n = 2500;
dt = 0.01;
X = randn(n,1);
true_intercept = log(10.0);
true_beta = 0.45;
lambda = exp(true_intercept + X*true_beta);
y = poissrnd(lambda*dt);

offset = log(dt) * ones(n,1);
b = glmfit(X, y, 'poisson', 'constant', 'on', 'offset', offset);
expected_rate = exp(b(1) + X*b(2));

save(fullfile(out_dir, 'PPSimExample_gold.mat'), ...
    'X', 'y', 'dt', 'true_intercept', 'true_beta', 'b', 'expected_rate', '-v7');

% ------------------------------------------
% Fixture 2: DecodingExampleWithHist (Bayes)
% ------------------------------------------
n_units = 12;
n_states = 18;
n_time = 180;
centers = linspace(0, n_states-1, n_units)';
widths = 2.0 * ones(n_units,1);
states = 0:(n_states-1);
tuning = zeros(n_units, n_states);
for i=1:n_units
    tuning(i,:) = 0.05 + 0.35*exp(-0.5*((states-centers(i))./widths(i)).^2);
end

transition = zeros(n_states, n_states);
for i=1:n_states
    if i>1
        transition(i,i-1) = 0.2;
    end
    transition(i,i) = 0.6;
    if i<n_states
        transition(i,i+1) = 0.2;
    end
    transition(i,:) = transition(i,:)/sum(transition(i,:));
end

latent = zeros(1, n_time);
latent(1) = floor(n_states/2) + 1;
for t=2:n_time
    cdf = cumsum(transition(latent(t-1),:));
    r = rand();
    latent(t) = find(r <= cdf, 1, 'first');
end

spike_counts = zeros(n_units, n_time);
for t=1:n_time
    spike_counts(:,t) = poissrnd(tuning(:, latent(t)));
end

log_emit = zeros(n_states, n_time);
for s=1:n_states
    r = tuning(:,s);
    log_emit(s,:) = sum(spike_counts .* log(r) - r - gammaln(spike_counts + 1), 1);
end
log_prior = log((1.0/n_states) * ones(n_states,1));

log_post = zeros(n_states, n_time);
log_post(:,1) = log_prior + log_emit(:,1);
log_post(:,1) = log_post(:,1) - log(sum(exp(log_post(:,1))));

for t=2:n_time
    pred = zeros(n_states,1);
    for s_next=1:n_states
        vals = log_post(:,t-1) + log(transition(:,s_next));
        maxv = max(vals);
        pred(s_next) = maxv + log(sum(exp(vals - maxv)));
    end
    log_post(:,t) = pred + log_emit(:,t);
    maxv = max(log_post(:,t));
    log_post(:,t) = log_post(:,t) - (maxv + log(sum(exp(log_post(:,t)-maxv))));
end

expected_posterior = exp(log_post);
[~, idx] = max(expected_posterior, [], 1);
expected_decoded = idx - 1; % zero-based to match python implementation

save(fullfile(out_dir, 'DecodingExampleWithHist_gold.mat'), ...
    'spike_counts', 'tuning', 'transition', 'expected_posterior', 'expected_decoded', '-v7');

% --------------------------------------------------
% Fixture 3: HippocampalPlaceCellExample (Weighted)
% --------------------------------------------------
n_units_pc = 25;
n_states_pc = 40;
n_time_pc = 200;
state_axis = 0:(n_states_pc-1);

centers_pc = linspace(0, n_states_pc-1, n_units_pc)';
widths_pc = 3.0 + rand(n_units_pc,1);
tuning_curves = zeros(n_units_pc, n_states_pc);
for i=1:n_units_pc
    tuning_curves(i,:) = 0.1 + 1.2*exp(-0.5*((state_axis-centers_pc(i))./widths_pc(i)).^2);
end

latent_pc = zeros(1, n_time_pc);
latent_pc(1) = floor(n_states_pc/2) + 1;
for t=2:n_time_pc
    latent_pc(t) = min(max(latent_pc(t-1) + randi([-1,1]), 1), n_states_pc);
end

spike_counts_pc = zeros(n_units_pc, n_time_pc);
for t=1:n_time_pc
    spike_counts_pc(:,t) = poissrnd(tuning_curves(:, latent_pc(t)));
end

expected_decoded_weighted = zeros(1, n_time_pc);
for t=1:n_time_pc
    weights = spike_counts_pc(:,t) .* tuning_curves;
    post = sum(weights,1);
    post = post / (sum(post) + 1e-12);
    expected_decoded_weighted(t) = sum(post .* state_axis);
end

save(fullfile(out_dir, 'HippocampalPlaceCellExample_gold.mat'), ...
    'spike_counts_pc', 'tuning_curves', 'expected_decoded_weighted', '-v7');

% ------------------------------------------------------
% Fixture 4: Spike rate-difference confidence intervals
% ------------------------------------------------------
n_trials_diff = 20;
n_bins_diff = 300;
alpha_diff = 0.05;
p_a = 0.08;
p_b = 0.06;
spike_matrix_a = binornd(1, p_a, n_trials_diff, n_bins_diff);
spike_matrix_b = binornd(1, p_b, n_trials_diff, n_bins_diff);

rate_a = sum(spike_matrix_a, 2) / n_bins_diff;
rate_b = sum(spike_matrix_b, 2) / n_bins_diff;
expected_diff = rate_a - rate_b;
var_term = (rate_a .* (1-rate_a) + rate_b .* (1-rate_b)) / n_bins_diff;
var_term = max(var_term, 1e-12);
z = 1.959963984540054;
half = z * sqrt(var_term);
expected_lo = expected_diff - half;
expected_hi = expected_diff + half;

save(fullfile(out_dir, 'SpikeRateDiffCIs_gold.mat'), ...
    'spike_matrix_a', 'spike_matrix_b', 'alpha_diff', ...
    'expected_diff', 'expected_lo', 'expected_hi', '-v7');

% ----------------------------------------
% Fixture 5: PSTHEstimation (rate + FDR)
% ----------------------------------------
n_trials_psth = 16;
n_bins_psth = 240;
alpha_psth = 0.05;
t = linspace(0, 1, n_bins_psth);
base_p = 0.03 + 0.02*(t > 0.35) + 0.01*sin(2*pi*2*t);
base_p = min(max(base_p, 0.001), 0.75);

spike_matrix_psth = zeros(n_trials_psth, n_bins_psth);
for k=1:n_trials_psth
    scale = 0.65 + 0.04*k;
    p = min(max(base_p * scale, 0.001), 0.85);
    spike_matrix_psth(k,:) = binornd(1, p);
end

counts_psth = sum(spike_matrix_psth, 2);
expected_rate_psth = counts_psth / n_bins_psth;
expected_prob_psth = ones(n_trials_psth, n_trials_psth);
upper_idx = zeros(n_trials_psth*(n_trials_psth-1)/2, 2);
upper_p = zeros(n_trials_psth*(n_trials_psth-1)/2, 1);
idx = 1;
for i=1:n_trials_psth
    for j=i+1:n_trials_psth
        p1 = expected_rate_psth(i);
        p2 = expected_rate_psth(j);
        pooled = (counts_psth(i) + counts_psth(j)) / (2.0 * n_bins_psth);
        se = sqrt(max(pooled * (1.0 - pooled) * (2.0 / n_bins_psth), 0.0));
        if se <= 0.0
            if abs(p1-p2) <= 1e-12
                pval = 1.0;
            else
                pval = 0.0;
            end
        else
            zstat = (p1 - p2) / se;
            pval = 2.0 * (1.0 - normcdf(abs(zstat), 0, 1));
        end
        expected_prob_psth(i,j) = pval;
        expected_prob_psth(j,i) = pval;
        upper_idx(idx,:) = [i j];
        upper_p(idx) = pval;
        idx = idx + 1;
    end
end

expected_sig_psth = zeros(n_trials_psth, n_trials_psth);
[sorted_p, order] = sort(upper_p, 'ascend');
m = numel(sorted_p);
thresholds = alpha_psth * ((1:m)' / m);
pass = find(sorted_p <= thresholds);
if ~isempty(pass)
    cutoff = sorted_p(max(pass));
    selected = upper_p <= cutoff;
    for q=1:numel(selected)
        if selected(q)
            i = upper_idx(q,1);
            j = upper_idx(q,2);
            expected_sig_psth(i,j) = 1;
            expected_sig_psth(j,i) = 1;
        end
    end
end
expected_prob_psth(1:n_trials_psth+1:end) = 1.0;
expected_sig_psth(1:n_trials_psth+1:end) = 0.0;

save(fullfile(out_dir, 'PSTHEstimation_gold.mat'), ...
    'spike_matrix_psth', 'alpha_psth', ...
    'expected_rate_psth', 'expected_prob_psth', 'expected_sig_psth', '-v7');

% ---------------------------------------------------------
% Fixture 6: nstCollExamples (binned/count collection ops)
% ---------------------------------------------------------
spike_times_1 = [0.02 0.08 0.30 0.45 0.75]';
spike_times_2 = [0.01 0.10 0.40 0.41 0.95]';
t_start_coll = 0.0;
t_end_coll = 1.0;
bin_size_coll = 0.1;
edges_coll = t_start_coll:bin_size_coll:t_end_coll;

c1 = histcounts(spike_times_1, edges_coll);
c2 = histcounts(spike_times_2, edges_coll);
expected_count_matrix = [c1; c2];
expected_binary_matrix = double(expected_count_matrix > 0);
expected_centers = (edges_coll(1:end-1) + edges_coll(2:end))/2;
expected_first_spike = min([spike_times_1; spike_times_2]);
expected_last_spike = max([spike_times_1; spike_times_2]);
expected_merged_spikes = sort([spike_times_1; spike_times_2]);

save(fullfile(out_dir, 'nstCollExamples_gold.mat'), ...
    'spike_times_1', 'spike_times_2', 't_start_coll', 't_end_coll', 'bin_size_coll', ...
    'expected_count_matrix', 'expected_binary_matrix', 'expected_centers', ...
    'expected_first_spike', 'expected_last_spike', 'expected_merged_spikes', '-v7');

% ------------------------------------------------------
% Fixture 7: CovCollExamples (design matrix + selectors)
% ------------------------------------------------------
time_cov = (0:0.01:1.0)';
cov_stim = sin(2*pi*1.0*time_cov);
cov_ctx = [cos(2*pi*1.0*time_cov), time_cov];
expected_design_cov = [cov_stim, cov_ctx];
expected_ctx_only = cov_ctx;
expected_stim_only = cov_stim;

save(fullfile(out_dir, 'CovCollExamples_gold.mat'), ...
    'time_cov', 'cov_stim', 'cov_ctx', 'expected_design_cov', ...
    'expected_ctx_only', 'expected_stim_only', '-v7');

% --------------------------------------------------------------
% Fixture 8: TrialExamples (aligned binned observations + X map)
% --------------------------------------------------------------
spike_times_trial = [0.02 0.08 0.13 0.21 0.32 0.55 0.72 0.91]';
bin_size_trial = 0.05;
edges_trial = t_start_coll:bin_size_trial:t_end_coll;
expected_t_bins_trial = (edges_trial(1:end-1) + edges_trial(2:end))/2;
expected_y_trial = histcounts(spike_times_trial, edges_trial)';

idx_trial = zeros(numel(expected_t_bins_trial), 1);
for k=1:numel(expected_t_bins_trial)
    idx = find(time_cov >= expected_t_bins_trial(k), 1, 'first');
    if isempty(idx)
        idx = numel(time_cov);
    end
    idx_trial(k) = idx;
end
expected_X_trial = expected_design_cov(idx_trial, :);

save(fullfile(out_dir, 'TrialExamples_gold.mat'), ...
    'time_cov', 'cov_stim', 'cov_ctx', 'spike_times_trial', 'bin_size_trial', ...
    'expected_t_bins_trial', 'expected_y_trial', 'expected_X_trial', '-v7');

% ----------------------------------------------
% Fixture 9: EventsExamples (subset extraction)
% ----------------------------------------------
event_times = [0.079 0.579 0.997]';
subset_start = 0.1;
subset_end = 0.7;
mask = event_times >= subset_start & event_times <= subset_end;
expected_subset_times = event_times(mask);

save(fullfile(out_dir, 'EventsExamples_gold.mat'), ...
    'event_times', 'subset_start', 'subset_end', 'expected_subset_times', '-v7');

fprintf('MATLAB gold fixtures exported to %s\n', out_dir);
"""


FIXTURE_FILES = [
    "PPSimExample_gold.mat",
    "DecodingExampleWithHist_gold.mat",
    "HippocampalPlaceCellExample_gold.mat",
    "SpikeRateDiffCIs_gold.mat",
    "PSTHEstimation_gold.mat",
    "nstCollExamples_gold.mat",
    "TrialExamples_gold.mat",
    "CovCollExamples_gold.mat",
    "EventsExamples_gold.mat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold"),
        help="Directory for exported MATLAB fixtures",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold/manifest.yml"),
        help="Output manifest path",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script_content = MATLAB_SCRIPT_TEMPLATE.format(out_dir=str(out_dir).replace("'", "''"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_content)
        tmp_path = Path(tmp.name)

    try:
        escaped_tmp = str(tmp_path).replace("'", "''")
        cmd = ["matlab", "-batch", f"run('{escaped_tmp}')"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError("MATLAB fixture export failed")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    fixtures = []
    for file_name in FIXTURE_FILES:
        path = out_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"expected fixture missing: {path}")
        fixtures.append(
            {
                "name": file_name.replace("_gold.mat", ""),
                "path": str(path.relative_to(repo_root).as_posix()),
                "sha256": _sha256(path),
                "source": "matlab_batch_export",
            }
        )

    manifest = {
        "version": 1,
        "fixtures": fixtures,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
