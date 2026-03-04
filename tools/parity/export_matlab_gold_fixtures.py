#!/usr/bin/env python3
"""Export MATLAB-gold fixtures for parity workflows and topic audit coverage.

This script runs MATLAB in batch mode to generate deterministic fixture files
for parity-critical workflow families and representative examples, including:
- PPSimExample
- DecodingExample / DecodingExampleWithHist
- HippocampalPlaceCellExample
- SpikeRateDiffCIs / PSTHEstimation
- nstCollExamples / TrialExamples / CovCollExamples / EventsExamples
- AnalysisExamples
- ExplicitStimulusWhiskerData
- mEPSCAnalysis

In addition to numeric `.mat` fixtures, this exporter also emits per-topic
audit JSON fixtures for notebook topics not covered by numeric fixtures so
that numeric drift gating remains fixture-backed across all examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

% ------------------------------------------------------
% Fixture 10: AnalysisExamples (spatial Poisson GLM fit)
% ------------------------------------------------------
n_t_analysis = 1800;
dt_analysis = 0.01;
xy = zeros(n_t_analysis, 2);
xy(1,:) = [45 55];
vel = [0 0];
for t=2:n_t_analysis
    vel = 0.92 * vel + 2.0 * randn(1,2);
    xy(t,:) = min(max(xy(t-1,:) + vel, 0.0), 100.0);
end

xc = 62.0; yc = 38.0; sigma = 16.0;
r2 = (xy(:,1)-xc).^2 + (xy(:,2)-yc).^2;
true_rate_analysis = 1.2 + 18.0 * exp(-0.5 * r2 / (sigma^2));
counts_analysis = poissrnd(true_rate_analysis * dt_analysis);
X_analysis = [xy(:,1), xy(:,2)];
offset_analysis = log(dt_analysis) * ones(n_t_analysis,1);
b_analysis = glmfit(X_analysis, counts_analysis, 'poisson', 'constant', 'on', 'offset', offset_analysis);
expected_rate_analysis = exp(b_analysis(1) + X_analysis * b_analysis(2:end));
expected_rmse_analysis = sqrt(mean((expected_rate_analysis - true_rate_analysis).^2));

save(fullfile(out_dir, 'AnalysisExamples_gold.mat'), ...
    'X_analysis', 'counts_analysis', 'dt_analysis', 'b_analysis', ...
    'true_rate_analysis', 'expected_rate_analysis', 'expected_rmse_analysis', '-v7');

% ----------------------------------------------------
% Fixture 11: DecodingExample (no-history Bayes decode)
% ----------------------------------------------------
n_units_dec = 10;
n_states_dec = 15;
n_time_dec = 150;
centers_dec = linspace(0, n_states_dec-1, n_units_dec)';
widths_dec = 2.0 * ones(n_units_dec,1);
states_dec = 0:(n_states_dec-1);
tuning_dec = zeros(n_units_dec, n_states_dec);
for i=1:n_units_dec
    tuning_dec(i,:) = 0.06 + 0.42*exp(-0.5*((states_dec-centers_dec(i))./widths_dec(i)).^2);
end

transition_dec = zeros(n_states_dec, n_states_dec);
for i=1:n_states_dec
    if i>1
        transition_dec(i,i-1) = 0.2;
    end
    transition_dec(i,i) = 0.6;
    if i<n_states_dec
        transition_dec(i,i+1) = 0.2;
    end
    transition_dec(i,:) = transition_dec(i,:) / sum(transition_dec(i,:));
end

latent_dec = zeros(1, n_time_dec);
latent_dec(1) = floor(n_states_dec/2) + 1;
for t=2:n_time_dec
    cdf = cumsum(transition_dec(latent_dec(t-1),:));
    r = rand();
    latent_dec(t) = find(r <= cdf, 1, 'first');
end

spike_counts_dec = zeros(n_units_dec, n_time_dec);
for t=1:n_time_dec
    spike_counts_dec(:,t) = poissrnd(tuning_dec(:, latent_dec(t)));
end

log_emit_dec = zeros(n_states_dec, n_time_dec);
for s=1:n_states_dec
    rr = tuning_dec(:,s);
    log_emit_dec(s,:) = sum(spike_counts_dec .* log(rr) - rr - gammaln(spike_counts_dec + 1), 1);
end
log_prior_dec = log((1.0/n_states_dec) * ones(n_states_dec,1));
log_post_dec = zeros(n_states_dec, n_time_dec);
log_post_dec(:,1) = log_prior_dec + log_emit_dec(:,1);
log_post_dec(:,1) = log_post_dec(:,1) - log(sum(exp(log_post_dec(:,1))));

for t=2:n_time_dec
    pred = zeros(n_states_dec,1);
    for s_next=1:n_states_dec
        vals = log_post_dec(:,t-1) + log(transition_dec(:,s_next));
        maxv = max(vals);
        pred(s_next) = maxv + log(sum(exp(vals - maxv)));
    end
    log_post_dec(:,t) = pred + log_emit_dec(:,t);
    maxv = max(log_post_dec(:,t));
    log_post_dec(:,t) = log_post_dec(:,t) - (maxv + log(sum(exp(log_post_dec(:,t)-maxv))));
end

expected_posterior_dec = exp(log_post_dec);
[~, idx_dec] = max(expected_posterior_dec, [], 1);
expected_decoded_dec = idx_dec - 1;
latent_zero_dec = latent_dec - 1;
expected_rmse_dec = sqrt(mean((expected_decoded_dec - latent_zero_dec).^2)) / (n_states_dec - 1);

save(fullfile(out_dir, 'DecodingExample_gold.mat'), ...
    'spike_counts_dec', 'tuning_dec', 'transition_dec', 'latent_zero_dec', ...
    'expected_posterior_dec', 'expected_decoded_dec', 'expected_rmse_dec', '-v7');

% ------------------------------------------------------------
% Fixture 12: ExplicitStimulusWhiskerData (binomial GLM proxy)
% ------------------------------------------------------------
dt_ws = 0.001;
time_ws = (0:dt_ws:2.0-dt_ws)';
envelope_ws = 0.8 * sin(2*pi*1.2*time_ws);
transients_ws = exp(-0.5*((time_ws-0.55)/0.03).^2) ...
    + exp(-0.5*((time_ws-1.15)/0.03).^2) ...
    + exp(-0.5*((time_ws-1.75)/0.03).^2);
stimulus_ws = envelope_ws + 1.1 * transients_ws;
stimulus_ws = (stimulus_ws - mean(stimulus_ws)) / std(stimulus_ws);
eta_ws = -3.2 + 1.0 * stimulus_ws;
p_ws = 1.0 ./ (1.0 + exp(-eta_ws));
spike_ws = binornd(1, p_ws);
b_ws = glmfit(stimulus_ws, spike_ws, 'binomial');
expected_prob_ws = 1.0 ./ (1.0 + exp(-(b_ws(1) + b_ws(2) * stimulus_ws)));
expected_rmse_ws = sqrt(mean((expected_prob_ws - spike_ws).^2));

save(fullfile(out_dir, 'ExplicitStimulusWhiskerData_gold.mat'), ...
    'time_ws', 'stimulus_ws', 'spike_ws', 'b_ws', 'expected_prob_ws', 'expected_rmse_ws', '-v7');

% ---------------------------------------------------
% Fixture 13: mEPSCAnalysis (event-detection metrics)
% ---------------------------------------------------
dt_mepsc = 0.0005;
time_mepsc = (0:dt_mepsc:6.0-dt_mepsc)';
n_mepsc = numel(time_mepsc);
trace_mepsc = 0.025 * randn(n_mepsc,1);
event_times_mepsc = sort(0.4 + (5.2 * rand(45,1)));
event_amps_mepsc = 0.12 + 0.30 * rand(45,1);

kernel_t = (0:dt_mepsc:0.060)';
kernel = (1.0 - exp(-kernel_t/0.0015)) .* exp(-kernel_t/0.010);
kernel = kernel ./ max(kernel);

for i=1:numel(event_times_mepsc)
    idx = round(event_times_mepsc(i) / dt_mepsc) + 1;
    idx_end = min(idx + numel(kernel) - 1, n_mepsc);
    k = kernel(1:(idx_end-idx+1));
    trace_mepsc(idx:idx_end) = trace_mepsc(idx:idx_end) - event_amps_mepsc(i) * k;
end

threshold_mepsc = -0.12;
refractory_mepsc = round(0.006 / dt_mepsc);
candidate_idx = find(trace_mepsc < threshold_mepsc);
detected_idx = [];
last_idx = -refractory_mepsc;
for i=1:numel(candidate_idx)
    idx = candidate_idx(i);
    if idx - last_idx >= refractory_mepsc
        w_end = min(idx + round(0.004 / dt_mepsc), n_mepsc);
        [~, local_rel] = min(trace_mepsc(idx:w_end));
        local_idx = idx + local_rel - 1;
        detected_idx = [detected_idx; local_idx]; %#ok<AGROW>
        last_idx = local_idx;
    end
end

detected_times_mepsc = (detected_idx - 1) * dt_mepsc;
detected_amps_mepsc = -trace_mepsc(detected_idx);
expected_event_count_mepsc = numel(detected_idx);
expected_mean_amp_mepsc = mean(detected_amps_mepsc);

save(fullfile(out_dir, 'mEPSCAnalysis_gold.mat'), ...
    'dt_mepsc', 'time_mepsc', 'trace_mepsc', 'event_times_mepsc', ...
    'detected_times_mepsc', 'detected_amps_mepsc', ...
    'expected_event_count_mepsc', 'expected_mean_amp_mepsc', '-v7');

% ---------------------------------------------------------
% Fixture 14: HybridFilterExample (state and filter outputs)
% ---------------------------------------------------------
n_t_hf = 500;
dt_hf = 0.02;
time_hf = (0:n_t_hf-1)' * dt_hf;
A_hf = [1.0, 0.0, dt_hf, 0.0; 0.0, 1.0, 0.0, dt_hf; 0.0, 0.0, 0.98, 0.0; 0.0, 0.0, 0.0, 0.98];
H_hf = [1.0, 0.0, 0.0, 0.0; 0.0, 1.0, 0.0, 0.0];
Q_hf = diag([1e-4, 1e-4, 1.5e-3, 1.5e-3]);
R_hf = diag([0.12^2, 0.12^2]);
pij_hf = [0.998, 0.002; 0.001, 0.999];

state_hf = ones(n_t_hf, 1);
for k=2:n_t_hf
    stay_p = pij_hf(state_hf(k-1), state_hf(k-1));
    if rand() < stay_p
        state_hf(k) = state_hf(k-1);
    else
        state_hf(k) = 3 - state_hf(k-1);
    end
end

x_true_hf = zeros(n_t_hf, 4);
x_true_hf(1,:) = [0.0, 0.0, 0.8, 0.35];
for k=2:n_t_hf
    if state_hf(k) == 1
        proc = mvnrnd(zeros(1,4), 0.15 * Q_hf, 1);
        x_true_hf(k,:) = x_true_hf(k-1,:) + proc;
    else
        proc = mvnrnd(zeros(1,4), Q_hf, 1);
        x_true_hf(k,:) = (A_hf * x_true_hf(k-1,:)')' + proc;
    end
end

z_hf = (H_hf * x_true_hf')' + mvnrnd([0.0, 0.0], R_hf, n_t_hf);
x_hat_hf = zeros(n_t_hf, 4);
x_hat_nt_hf = zeros(n_t_hf, 4);
P_hf = eye(4);
P_nt_hf = eye(4);
for k=2:n_t_hf
    if state_hf(k) == 1
        A_k = eye(4);
        Q_k = 0.15 * Q_hf;
    else
        A_k = A_hf;
        Q_k = Q_hf;
    end

    x_pred = (A_k * x_hat_hf(k-1,:)')';
    P_pred = A_k * P_hf * A_k' + Q_k;
    S = H_hf * P_pred * H_hf' + R_hf;
    K = P_pred * H_hf' / S;
    x_hat_hf(k,:) = x_pred + (K * (z_hf(k,:)' - H_hf * x_pred'))';
    P_hf = (eye(4) - K * H_hf) * P_pred;

    x_pred_nt = (A_hf * x_hat_nt_hf(k-1,:)')';
    P_pred_nt = A_hf * P_nt_hf * A_hf' + Q_hf;
    S_nt = H_hf * P_pred_nt * H_hf' + R_hf;
    K_nt = P_pred_nt * H_hf' / S_nt;
    x_hat_nt_hf(k,:) = x_pred_nt + (K_nt * (z_hf(k,:)' - H_hf * x_pred_nt'))';
    P_nt_hf = (eye(4) - K_nt * H_hf) * P_pred_nt;
end

err_hf = sqrt(sum((x_hat_hf(:,1:2) - x_true_hf(:,1:2)).^2, 2));
err_nt_hf = sqrt(sum((x_hat_nt_hf(:,1:2) - x_true_hf(:,1:2)).^2, 2));
rmse_hf = sqrt(mean(err_hf.^2));
rmse_nt_hf = sqrt(mean(err_nt_hf.^2));

save(fullfile(out_dir, 'HybridFilterExample_gold.mat'), ...
    'dt_hf', 'time_hf', 'state_hf', 'x_true_hf', 'z_hf', ...
    'x_hat_hf', 'x_hat_nt_hf', 'rmse_hf', 'rmse_nt_hf', '-v7');

% ----------------------------------------------------
% Fixture 15: ValidationDataSet (trial PSTH statistics)
% ----------------------------------------------------
dt_val = 0.001;
time_val = (0:dt_val:1.2-dt_val)';
n_trials_val = 30;
rate_val = 5.0 + 8.0 * (time_val > 0.35) + 4.0 * sin(2.0*pi*2.0*time_val);
rate_val = max(rate_val, 0.2);
trial_matrix_val = zeros(n_trials_val, numel(time_val));
for k=1:n_trials_val
    jitter = 0.6 + 0.8 * rand();
    p = min(max(rate_val * jitter * dt_val, 0.0), 0.6);
    trial_matrix_val(k,:) = binornd(1, p)';
end
psth_val = mean(trial_matrix_val, 1)' / dt_val;
sem_val = std(trial_matrix_val, 0, 1)' / sqrt(n_trials_val) / dt_val;

expected_rate_val = sum(trial_matrix_val, 2) / numel(time_val);
expected_prob_val = ones(n_trials_val, n_trials_val);
upper_idx_val = zeros(n_trials_val*(n_trials_val-1)/2, 2);
upper_p_val = zeros(n_trials_val*(n_trials_val-1)/2, 1);
idx_val = 1;
for i=1:n_trials_val
    for j=i+1:n_trials_val
        p1 = expected_rate_val(i);
        p2 = expected_rate_val(j);
        pooled = (sum(trial_matrix_val(i,:)) + sum(trial_matrix_val(j,:))) / (2.0 * numel(time_val));
        se = sqrt(max(pooled * (1.0 - pooled) * (2.0 / numel(time_val)), 0.0));
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
        expected_prob_val(i,j) = pval;
        expected_prob_val(j,i) = pval;
        upper_idx_val(idx_val,:) = [i j];
        upper_p_val(idx_val) = pval;
        idx_val = idx_val + 1;
    end
end

expected_sig_val = zeros(n_trials_val, n_trials_val);
[sorted_p_val, order_val] = sort(upper_p_val, 'ascend');
m_val = numel(sorted_p_val);
thresholds_val = 0.05 * ((1:m_val)' / m_val);
pass_val = find(sorted_p_val <= thresholds_val);
if ~isempty(pass_val)
    cutoff_val = sorted_p_val(max(pass_val));
    selected_val = upper_p_val <= cutoff_val;
    for q=1:numel(selected_val)
        if selected_val(q)
            i = upper_idx_val(q,1);
            j = upper_idx_val(q,2);
            expected_sig_val(i,j) = 1;
            expected_sig_val(j,i) = 1;
        end
    end
end
expected_prob_val(1:n_trials_val+1:end) = 1.0;
expected_sig_val(1:n_trials_val+1:end) = 0.0;

save(fullfile(out_dir, 'ValidationDataSet_gold.mat'), ...
    'dt_val', 'time_val', 'trial_matrix_val', 'psth_val', 'sem_val', ...
    'expected_rate_val', 'expected_prob_val', 'expected_sig_val', '-v7');

% -----------------------------------------------------
% Fixture 16: StimulusDecode2D (trajectory decode arrays)
% -----------------------------------------------------
side_sd = 14;
grid_sd = linspace(0.0, 1.0, side_sd);
[gx_sd, gy_sd] = meshgrid(grid_sd, grid_sd);
states_sd = [gx_sd(:), gy_sd(:)];
n_states_sd = size(states_sd, 1);
n_units_sd = 24;
n_time_sd = 280;
traj_sd = zeros(n_time_sd, 2);
traj_sd(1,:) = [0.5, 0.5];
vel_sd = [0.0, 0.0];
for t=2:n_time_sd
    vel_sd = 0.82 * vel_sd + 0.12 * randn(1,2);
    traj_sd(t,:) = min(max(traj_sd(t-1,:) + vel_sd, 0.0), 1.0);
end

state_match_sd = zeros(n_time_sd, n_states_sd);
for t=1:n_time_sd
    delta_sd = states_sd - traj_sd(t,:);
    state_match_sd(t,:) = sum(delta_sd.^2, 2)';
end
[~, latent_idx_sd] = min(state_match_sd, [], 2);
latent_sd = latent_idx_sd - 1; % zero-based for Python

centers_sd = rand(n_units_sd, 2);
sigma_sd = 0.16;
tuning_sd = zeros(n_units_sd, n_states_sd);
for i=1:n_units_sd
    dist2_sd = sum((states_sd - centers_sd(i,:)).^2, 2);
    tuning_sd(i,:) = 0.03 + 0.80 * exp(-0.5 * dist2_sd' / (sigma_sd^2));
end

spike_counts_sd = zeros(n_units_sd, n_time_sd);
for t=1:n_time_sd
    spike_counts_sd(:,t) = poissrnd(tuning_sd(:, latent_idx_sd(t)));
end

decoded_center_sd = zeros(n_time_sd, 1);
state_axis_sd = (0:n_states_sd-1)';
for t=1:n_time_sd
    weights_sd = spike_counts_sd(:,t) .* tuning_sd;
    post_sd = sum(weights_sd, 1)';
    post_sd = post_sd / (sum(post_sd) + 1e-12);
    decoded_center_sd(t) = sum(post_sd .* state_axis_sd);
end
decoded_sd = round(decoded_center_sd);
decoded_sd = max(min(decoded_sd, n_states_sd-1), 0);
xy_true_sd = states_sd(latent_idx_sd, :);
xy_decoded_sd = states_sd(decoded_sd + 1, :);
rmse_sd = sqrt(mean(sum((xy_decoded_sd - xy_true_sd).^2, 2)));

save(fullfile(out_dir, 'StimulusDecode2D_gold.mat'), ...
    'side_sd', 'states_sd', 'latent_sd', 'tuning_sd', 'spike_counts_sd', ...
    'decoded_center_sd', 'decoded_sd', 'xy_true_sd', 'xy_decoded_sd', 'rmse_sd', '-v7');

fprintf('MATLAB gold fixtures exported to %s\n', out_dir);
"""


NUMERIC_FIXTURE_FILES = [
    "PPSimExample_gold.mat",
    "DecodingExampleWithHist_gold.mat",
    "HippocampalPlaceCellExample_gold.mat",
    "SpikeRateDiffCIs_gold.mat",
    "PSTHEstimation_gold.mat",
    "nstCollExamples_gold.mat",
    "TrialExamples_gold.mat",
    "CovCollExamples_gold.mat",
    "EventsExamples_gold.mat",
    "AnalysisExamples_gold.mat",
    "DecodingExample_gold.mat",
    "ExplicitStimulusWhiskerData_gold.mat",
    "mEPSCAnalysis_gold.mat",
    "HybridFilterExample_gold.mat",
    "ValidationDataSet_gold.mat",
    "StimulusDecode2D_gold.mat",
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
    parser.add_argument(
        "--notebook-manifest",
        type=Path,
        default=Path("tools/notebooks/notebook_manifest.yml"),
        help="Notebook manifest used to define required topic coverage",
    )
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        default=Path("parity/function_example_alignment_report.json"),
        help="Equivalence audit JSON used to export topic-audit fixtures",
    )
    parser.add_argument(
        "--skip-matlab-export",
        action="store_true",
        help="Skip MATLAB batch execution and only rebuild manifest/topic-audit fixtures.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_required_topics(notebook_manifest: Path) -> list[str]:
    payload = yaml.safe_load(notebook_manifest.read_text(encoding="utf-8")) or {}
    topics: list[str] = []
    for row in payload.get("notebooks", []):
        topic = str(row.get("topic", "")).strip()
        if topic:
            topics.append(topic)
    return topics


def _load_equivalence_rows(equivalence_report: Path) -> dict[str, dict]:
    payload = json.loads(equivalence_report.read_text(encoding="utf-8"))
    rows = payload.get("example_line_alignment_audit", {}).get("topic_rows", [])
    out: dict[str, dict] = {}
    for row in rows:
        topic = str(row.get("topic", "")).strip()
        if topic:
            out[topic] = row
    return out


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_manifest = args.notebook_manifest.resolve()
    equivalence_report = args.equivalence_report.resolve()

    if not args.skip_matlab_export:
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
    for file_name in NUMERIC_FIXTURE_FILES:
        path = out_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"expected fixture missing: {path}")
        fixtures.append(
            {
                "name": file_name.replace("_gold.mat", ""),
                "path": str(path.relative_to(repo_root).as_posix()),
                "sha256": _sha256(path),
                "source": "matlab_batch_export",
                "fixture_type": "numeric",
            }
        )

    required_topics = _load_required_topics(notebook_manifest)
    topic_rows = _load_equivalence_rows(equivalence_report)
    covered_numeric_topics = {row["name"] for row in fixtures}
    audit_topics = sorted(topic for topic in required_topics if topic not in covered_numeric_topics)
    for topic in audit_topics:
        row = topic_rows.get(topic)
        if row is None:
            raise KeyError(f"topic {topic!r} missing from equivalence report: {equivalence_report}")
        audit_payload = {
            "schema_version": 1,
            "topic": topic,
            "alignment_status": str(row.get("alignment_status", "")),
            "matlab_code_lines": int(row.get("matlab_code_lines", 0)),
            "matlab_reference_image_count": int(row.get("matlab_reference_image_count", 0)),
            "min_assertion_count": int(row.get("assertion_count", 0)),
            "require_topic_checkpoint": bool(row.get("has_topic_checkpoint", False)),
            "min_python_validation_image_count": int(row.get("python_validation_image_count", 0)),
            "require_plot_call": bool(row.get("has_plot_call", False)),
            "source": "equivalence_audit_report",
            "equivalence_report": str(equivalence_report.relative_to(repo_root).as_posix()),
        }
        audit_path = out_dir / f"{topic}_audit_gold.json"
        audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
        fixtures.append(
            {
                "name": topic,
                "path": str(audit_path.relative_to(repo_root).as_posix()),
                "sha256": _sha256(audit_path),
                "source": "equivalence_audit_export",
                "fixture_type": "topic_audit",
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
