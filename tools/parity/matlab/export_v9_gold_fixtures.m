function export_v9_gold_fixtures(repoRoot, matlabRepoRoot)
% export_v9_gold_fixtures
% =========================================================================
% Recipes for the 22 v9_* MATLAB gold fixtures in
% tests/parity/fixtures/matlab_gold/ that were originally captured ad-hoc by
% v9 iter ~39-40 via direct `/opt/homebrew/bin/matlab -batch` snippets that
% were never committed to the repo.
%
% Fixtures covered (22):
%
%   analysis surface:
%     v9_RunAnalysisForNeuron.mat
%     v9_computeKSStats_full.mat
%     v9_computeHistLag.mat
%     v9_computeFitResidual.mat
%   decoding_PPAF (point-process adaptive filter):
%     v9_PPDecode_predict.mat
%     v9_PPDecode_update.mat
%   decoding_KF (Kalman):
%     v9_kalman_smoother.mat
%     v9_kalman_fixedIntervalSmoother.mat
%   decoding_PPSS (point-process state-space EM):
%     v9_PPSS_EStep.mat
%     v9_PPSS_MStep.mat
%     v9_PPSS_EM.mat
%   decoding_PPHF (point-process hybrid):
%     v9_PPHybridFilter.mat
%     v9_PPHybridFilterLinear.mat
%   cif:
%     v9_simulateCIFByThinning.mat
%     v9_simulateCIFByThinningFromLambda.mat
%   history_fit_core:
%     v9_raisedCosine.mat
%   FitResult helpers:
%     v9_fitresult_KSPlot_data.mat
%     v9_fitresult_invGausTrans_data.mat
%     v9_fitresult_seqCorrCoeff.mat
%   SignalObj:
%     v9_signalobj_resample.mat
%     v9_signalobj_derivative.mat
%     v9_signalobj_integral.mat
%
% USAGE
% -----
%   export_v9_gold_fixtures('/Users/iahncajigas/projects/nstat-python', ...
%                           '/Users/iahncajigas/projects/nstat');
%
% Each fixture's per-function header lists the saved field set expected by
% the Python recipe in `tools/parity/numerical_drift.py` and a recipe
% sketch. Functions are NO-OPs by default (early `return`) so this script
% does NOT overwrite the committed .mat files — remove the early return on
% a per-fixture basis after verifying the recipe in MATLAB.
%
% v9 CONVENTIONS (from memory):
%   * `rng(42)` at the start of every fixture function (deterministic).
%   * Small canonical dims: 2-state SSM, 2-cell ensembles, 10-step
%     observation windows, sampleRate = 10 Hz.
%
% TOLERANCE STATUS (after v10 iter 47 tightening): see
% `parity/numerical_drift_spec.yml` for current rtol/atol per entry. Several
% deterministic entries are at float64 round-off (1e-12 / 1e-14); a few
% remain Case-C-relaxed due to MC sampling / RNG divergence.

if nargin < 1 || isempty(repoRoot)
    error('repoRoot is required');
end
if nargin < 2 || isempty(matlabRepoRoot)
    matlabRepoRoot = fullfile(fileparts(repoRoot), 'nSTAT');
end

repoRoot = char(repoRoot);
matlabRepoRoot = char(matlabRepoRoot);

addpath(matlabRepoRoot);
addpath(genpath(fullfile(matlabRepoRoot, 'libraries')));

fixtureRoot = fullfile(repoRoot, 'tests', 'parity', 'fixtures', 'matlab_gold');
if ~exist(fixtureRoot, 'dir')
    mkdir(fixtureRoot);
end

% Failure isolation: dispatch each capture under a try/catch so one upstream
% bug does not block the rest (mirrors export_pplfp_gold_fixtures iter 49).
captures = { ...
    @export_v9_RunAnalysisForNeuron_fixture,             'RunAnalysisForNeuron'; ...
    @export_v9_computeKSStats_full_fixture,              'computeKSStats_full'; ...
    @export_v9_computeHistLag_fixture,                   'computeHistLag'; ...
    @export_v9_computeFitResidual_fixture,               'computeFitResidual'; ...
    @export_v9_PPDecode_predict_fixture,                 'PPDecode_predict'; ...
    @export_v9_PPDecode_update_fixture,                  'PPDecode_update'; ...
    @export_v9_kalman_smoother_fixture,                  'kalman_smoother'; ...
    @export_v9_kalman_fixedIntervalSmoother_fixture,     'kalman_fixedIntervalSmoother'; ...
    @export_v9_PPSS_EStep_fixture,                       'PPSS_EStep'; ...
    @export_v9_PPSS_MStep_fixture,                       'PPSS_MStep'; ...
    @export_v9_PPSS_EM_fixture,                          'PPSS_EM'; ...
    @export_v9_PPHybridFilter_fixture,                   'PPHybridFilter'; ...
    @export_v9_PPHybridFilterLinear_fixture,             'PPHybridFilterLinear'; ...
    @export_v9_simulateCIFByThinning_fixture,            'simulateCIFByThinning'; ...
    @export_v9_simulateCIFByThinningFromLambda_fixture,  'simulateCIFByThinningFromLambda'; ...
    @export_v9_raisedCosine_fixture,                     'raisedCosine'; ...
    @export_v9_fitresult_KSPlot_data_fixture,            'fitresult_KSPlot_data'; ...
    @export_v9_fitresult_invGausTrans_data_fixture,      'fitresult_invGausTrans_data'; ...
    @export_v9_fitresult_seqCorrCoeff_fixture,           'fitresult_seqCorrCoeff'; ...
    @export_v9_signalobj_resample_fixture,               'signalobj_resample'; ...
    @export_v9_signalobj_derivative_fixture,             'signalobj_derivative'; ...
    @export_v9_signalobj_integral_fixture,               'signalobj_integral'};
for ci = 1:size(captures, 1)
    fn = captures{ci, 1};
    label = captures{ci, 2};
    try
        fn(fixtureRoot);
    catch ME
        fprintf('  [%s] FAILED: %s — keeping committed .mat untouched.\n', ...
            label, ME.message);
    end
end

end


% =========================================================================
% Per-fixture stubs — each documents the saved field set expected by the
% Python recipe in tools/parity/numerical_drift.py. The early `return`
% prevents accidental overwrite of the committed fixture; remove it after
% verifying the recipe in MATLAB.
% =========================================================================


function export_v9_RunAnalysisForNeuron_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50B: re-load committed inputs and re-run Analysis.RunAnalysisForNeuron.
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_sample_rate, input_neuron_number, coeffs, lambda_data,
%   AIC, BIC, logLL, Z, U, ks_stat.
% NOTE: fit.lambda is a single Covariate (not a cell array) when the
% ConfigColl has one config — index via local_get_lambda helper.
rng(42);
fixturePath = fullfile(fixtureRoot, 'v9_RunAnalysisForNeuron.mat');
in = load(fixturePath);
input_time = double(in.input_time(:));
input_stim_data = double(in.input_stim_data(:));
input_spike_times = double(in.input_spike_times(:)');
input_sample_rate = double(in.input_sample_rate);
input_neuron_number = double(in.input_neuron_number);

stim = Covariate(input_time, input_stim_data, 'Stimulus', 'time', 's', '', {'stim'});
st = nspikeTrain(input_spike_times, '1', input_sample_rate, ...
                 input_time(1), input_time(end), 'time', 's', '', '', -1);
trial = Trial(nstColl({st}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus','stim'}}, input_sample_rate, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, input_neuron_number, ...
                                    ConfigColl({cfg}), 0);
coeffs = fit.getCoeffs(1);
lam0 = local_v50B_get_lambda(fit, 1);
lambda_data = lam0.data;
AIC = fit.AIC;
BIC = fit.BIC;
logLL = fit.logLL;
[Z, U, ~, ~, ks_stat] = Analysis.computeKSStats(st, lam0, 1);

save(fixturePath, 'input_time', 'input_stim_data', 'input_spike_times', ...
     'input_sample_rate', 'input_neuron_number', 'coeffs', 'lambda_data', ...
     'AIC', 'BIC', 'logLL', 'Z', 'U', 'ks_stat', '-v7');
fprintf('  [OK]   export_v9_RunAnalysisForNeuron_fixture\n');
end %#ok<DEFNU>


function export_v9_computeKSStats_full_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50B: re-load committed inputs and re-run Analysis.computeKSStats.
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_DTCorrection, lambda_time, lambda_data, lambda_sample_rate,
%   Z, U, xAxis, KSSorted, ks_stat.
rng(42);
fixturePath = fullfile(fixtureRoot, 'v9_computeKSStats_full.mat');
in = load(fixturePath);
input_time = double(in.input_time(:));
input_stim_data = double(in.input_stim_data(:));
input_spike_times = double(in.input_spike_times(:)');
input_DTCorrection = double(in.input_DTCorrection);
lambda_time = double(in.lambda_time(:));
lambda_data = double(in.lambda_data(:));
lambda_sample_rate = double(in.lambda_sample_rate);

lam = Covariate(lambda_time, lambda_data, '\lambda(t)', 'time', 's', 'Hz', {'\lambda_{1}'});
st = nspikeTrain(input_spike_times, '1', lambda_sample_rate, ...
                 input_time(1), input_time(end), 'time', 's', '', '', -1);
[Z, U, xAxis, KSSorted, ks_stat] = Analysis.computeKSStats(st, lam, input_DTCorrection);

save(fixturePath, 'input_time', 'input_stim_data', 'input_spike_times', ...
     'input_DTCorrection', 'lambda_time', 'lambda_data', 'lambda_sample_rate', ...
     'Z', 'U', 'xAxis', 'KSSorted', 'ks_stat', '-v7');
fprintf('  [OK]   export_v9_computeKSStats_full_fixture\n');
end %#ok<DEFNU>


function export_v9_computeHistLag_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50B: re-load committed inputs and re-run Analysis.computeHistLag.
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_windowTimes, input_neuronNum, input_sampleRate, num_fits,
%   fit0_coeffs, fit_AIC, fit_BIC, fit_logLL, lambda_data_cfg1, lambda_data_cfg2.
% NOTE: Analysis.computeHistLag returns a single FitResult (with multiple
% configurations packed inside its AIC/BIC/logLL vectors), not a cell of
% per-window FitResults — extract per-config metrics from the vector fields.
rng(42);
fixturePath = fullfile(fixtureRoot, 'v9_computeHistLag.mat');
in = load(fixturePath);
input_time = double(in.input_time(:));
input_stim_data = double(in.input_stim_data(:));
input_spike_times = double(in.input_spike_times(:)');
input_windowTimes = double(in.input_windowTimes(:)');
input_neuronNum = double(in.input_neuronNum);
input_sampleRate = double(in.input_sampleRate);

stim = Covariate(input_time, input_stim_data, 'Stimulus', 'time', 's', '', {'stim'});
st = nspikeTrain(input_spike_times, '1', input_sampleRate, ...
                 input_time(1), input_time(end), 'time', 's', '', '', -1);
trial = Trial(nstColl({st}), CovColl({stim}));
% Signature: computeHistLag(tObj,neuronNum,windowTimes,CovLabels,Algorithm,
%   batchMode,sampleRate,makePlot)
fitResults = Analysis.computeHistLag(trial, input_neuronNum, input_windowTimes, ...
    {{'Stimulus','stim'}}, [], [], input_sampleRate, 0);

% computeHistLag returns a single FitResult; AIC/BIC/logLL are vectors over
% the per-window configurations.
fit_AIC = reshape(double(fitResults.AIC), 1, []);
fit_BIC = reshape(double(fitResults.BIC), 1, []);
fit_logLL = reshape(double(fitResults.logLL), 1, []);
num_fits = numel(fit_AIC);
fit0_coeffs = fitResults.getCoeffs(1);
lam1 = local_v50B_get_lambda(fitResults, 1);
lambda_data_cfg1 = lam1.data;
if num_fits >= 2
    lam2 = local_v50B_get_lambda(fitResults, 2);
    lambda_data_cfg2 = lam2.data;
else
    lambda_data_cfg2 = lambda_data_cfg1;
end

save(fixturePath, 'input_time', 'input_stim_data', 'input_spike_times', ...
     'input_windowTimes', 'input_neuronNum', 'input_sampleRate', 'num_fits', ...
     'fit0_coeffs', 'fit_AIC', 'fit_BIC', 'fit_logLL', ...
     'lambda_data_cfg1', 'lambda_data_cfg2', '-v7');
fprintf('  [OK]   export_v9_computeHistLag_fixture\n');
end %#ok<DEFNU>


function export_v9_computeFitResidual_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50B: re-load committed inputs and re-run Analysis.computeFitResidual.
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_windowSize, lambda_time, lambda_data, lambda_sample_rate,
%   M_time, M_data.
% NOTE: Case C — Python bins at windowSize, MATLAB at lambda sample rate;
% recipe in numerical_drift.py compares sum-of-squared-residuals.
rng(42);
fixturePath = fullfile(fixtureRoot, 'v9_computeFitResidual.mat');
in = load(fixturePath);
input_time = double(in.input_time(:));
input_stim_data = double(in.input_stim_data(:));
input_spike_times = double(in.input_spike_times(:)');
input_windowSize = double(in.input_windowSize);
lambda_time = double(in.lambda_time(:));
lambda_data = double(in.lambda_data(:));
lambda_sample_rate = double(in.lambda_sample_rate);

st = nspikeTrain(input_spike_times, '1', lambda_sample_rate, ...
                 input_time(1), input_time(end), 'time', 's', '', '', -1);
lam = Covariate(lambda_time, lambda_data, '\lambda(t)', 'time', 's', 'Hz', {'\lambda_{1}'});
M = Analysis.computeFitResidual(st, lam, input_windowSize);
M_time = M.time(:);
M_data = M.data;

save(fixturePath, 'input_time', 'input_stim_data', 'input_spike_times', ...
     'input_windowSize', 'lambda_time', 'lambda_data', 'lambda_sample_rate', ...
     'M_time', 'M_data', '-v7');
fprintf('  [OK]   export_v9_computeFitResidual_fixture\n');
end %#ok<DEFNU>


function export_v9_PPDecode_predict_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs (x_u, W_u, A, Q) and re-run
% nstat.decoding.PPAF.PPDecode_predict. Fully deterministic — byte-equiv
% reproduction expected.
fixturePath = fullfile(fixtureRoot, 'v9_PPDecode_predict.mat');
fprintf('  [PPDecode_predict] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

x_u = double(data.x_u);
W_u = double(data.W_u);
A = double(data.A);
Q = double(data.Q);

rng(42); %#ok<RNG>
[x_p, W_p] = nstat.decoding.PPAF.PPDecode_predict(x_u, W_u, A, Q);

save(fixturePath, 'x_u', 'W_u', 'A', 'Q', 'x_p', 'W_p', '-v7');
fprintf('  [PPDecode_predict] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPDecode_update_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs (x_p, W_p, dN, beta, binwidth,
% time_index) and re-run nstat.decoding.PPAF.PPDecode_update.
%
% MATLAB CIF constructor: `lambdaDelta = exp(beta*varIn)` — beta is a 1xN
% row vector aligned with Xnames (intercept first); varIn is the Nx1
% symbolic vector. The fixture already stores beta as 1x2.
fixturePath = fullfile(fixtureRoot, 'v9_PPDecode_update.mat');
fprintf('  [PPDecode_update] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

x_p = double(data.x_p);
W_p = double(data.W_p);
dN = double(data.dN);
beta_row = double(data.beta);
binwidth = double(data.binwidth);
time_index = double(data.time_index);

% Xnames intercept must be a valid MATLAB symbolic variable — use 'one'
% (per CIF.m line 252 caller note: 'one' not '1').
cif = CIF(beta_row, {'one', 'x1'}, {'x1'}, 'poisson');

rng(42); %#ok<RNG>
[x_u, W_u, lambdaDeltaMat] = nstat.decoding.PPAF.PPDecode_update( ...
    x_p, W_p, dN, {cif}, binwidth, time_index);

beta = beta_row; %#ok<NASGU> preserve original 1x2 shape on resave
save(fixturePath, 'x_p', 'W_p', 'dN', 'beta', 'binwidth', 'time_index', ...
    'x_u', 'W_u', 'lambdaDeltaMat', '-v7');
fprintf('  [PPDecode_update] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_kalman_smoother_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs (A, C, Pv, Pw, Px0, x0, y) and
% re-run nstat.decoding.KalmanFilter.kalman_smoother. Fully deterministic.
fixturePath = fullfile(fixtureRoot, 'v9_kalman_smoother.mat');
fprintf('  [kalman_smoother] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
C = double(data.C);
Pv = double(data.Pv);
Pw = double(data.Pw);
Px0 = double(data.Px0);
x0 = double(data.x0);
y = double(data.y);

rng(42); %#ok<RNG>
[x_N, P_N, Ln, x_p, Pe_p, x_u, Pe_u] = ...
    nstat.decoding.KalmanFilter.kalman_smoother(A, C, Pv, Pw, Px0, x0, y);

save(fixturePath, 'A', 'C', 'Pv', 'Pw', 'Px0', 'x0', 'y', ...
    'x_N', 'P_N', 'Ln', 'x_p', 'Pe_p', 'x_u', 'Pe_u', '-v7');
fprintf('  [kalman_smoother] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_kalman_fixedIntervalSmoother_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs and re-run
% nstat.decoding.KalmanFilter.kalman_fixedIntervalSmoother. Deterministic.
fixturePath = fullfile(fixtureRoot, 'v9_kalman_fixedIntervalSmoother.mat');
fprintf('  [kalman_fixedIntervalSmoother] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
C = double(data.C);
Pv = double(data.Pv);
Pw = double(data.Pw);
Px0 = double(data.Px0);
x0 = double(data.x0);
y = double(data.y);
lags = double(data.lags);

rng(42); %#ok<RNG>
[x_pLag, Pe_pLag, x_uLag, Pe_uLag] = ...
    nstat.decoding.KalmanFilter.kalman_fixedIntervalSmoother( ...
        A, C, Pv, Pw, Px0, x0, y, lags);

save(fixturePath, 'A', 'C', 'Pv', 'Pw', 'Px0', 'x0', 'y', 'lags', ...
    'x_pLag', 'Pe_pLag', 'x_uLag', 'Pe_uLag', '-v7');
fprintf('  [kalman_fixedIntervalSmoother] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPSS_EStep_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs and re-run
% nstat.decoding.SSGLM.PPSS_EStep. Deterministic forward-backward smoother.
%
% HkAll is a 1xnumCells cell array of TxnumHistTerms history matrices.
fixturePath = fullfile(fixtureRoot, 'v9_PPSS_EStep.mat');
fprintf('  [PPSS_EStep] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
Q = double(data.Q);
x0 = double(data.x0);
dN = double(data.dN);
HkAll = data.HkAll;          % cell array — keep as-is
fitType = char(string(data.fitType));
delta = double(data.delta);
gamma = double(data.gamma);
numBasis = double(data.numBasis);

rng(42); %#ok<RNG>
[x_K, W_K, Wku, logll, sumXkTerms, sumPPll] = ...
    nstat.decoding.SSGLM.PPSS_EStep(A, Q, x0, dN, HkAll, fitType, delta, ...
        gamma, numBasis);

save(fixturePath, 'A', 'Q', 'x0', 'dN', 'HkAll', 'fitType', 'delta', ...
    'gamma', 'numBasis', ...
    'x_K', 'W_K', 'Wku', 'logll', 'sumXkTerms', 'sumPPll', '-v7');
fprintf('  [PPSS_EStep] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPSS_MStep_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs (including the EStep-derived
% x_K/W_K/sumXkTerms) and re-run nstat.decoding.SSGLM.PPSS_MStep.
fixturePath = fullfile(fixtureRoot, 'v9_PPSS_MStep.mat');
fprintf('  [PPSS_MStep] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

dN = double(data.dN);
HkAll = data.HkAll;
fitType = char(string(data.fitType));
x_K = double(data.x_K);
W_K = double(data.W_K);
gamma = double(data.gamma);
delta = double(data.delta);
sumXkTerms = double(data.sumXkTerms);
windowTimes = double(data.windowTimes);

rng(42); %#ok<RNG>
[Qhat, gamma_new] = nstat.decoding.SSGLM.PPSS_MStep(dN, HkAll, fitType, ...
    x_K, W_K, gamma, delta, sumXkTerms, windowTimes);

save(fixturePath, 'dN', 'HkAll', 'fitType', 'x_K', 'W_K', 'gamma', ...
    'delta', 'sumXkTerms', 'windowTimes', ...
    'Qhat', 'gamma_new', '-v7');
fprintf('  [PPSS_MStep] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPSS_EM_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs and re-run
% nstat.decoding.SSGLM.PPSS_EM. NOTE: iteration-sensitive; rng(42).
fixturePath = fullfile(fixtureRoot, 'v9_PPSS_EM.mat');
fprintf('  [PPSS_EM] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
Q0 = double(data.Q0);
x0 = double(data.x0);
dN = double(data.dN);
fitType = char(string(data.fitType));
delta = double(data.delta);
gamma0 = double(data.gamma0);
windowTimes = double(data.windowTimes);
numBasis = double(data.numBasis);
HkAll = data.HkAll;

rng(42); %#ok<RNG>
[xKFinal, WKFinal, WkuFinal, Qhat, gammahat, logll, QhatAll, gammahatAll, ...
    nIter, negLL] = nstat.decoding.SSGLM.PPSS_EM(A, Q0, x0, dN, fitType, ...
        delta, gamma0, windowTimes, numBasis, HkAll);

save(fixturePath, 'A', 'Q0', 'x0', 'dN', 'fitType', 'delta', 'gamma0', ...
    'windowTimes', 'numBasis', 'HkAll', ...
    'xKFinal', 'WKFinal', 'WkuFinal', 'Qhat', 'gammahat', 'logll', ...
    'QhatAll', 'gammahatAll', 'nIter', 'negLL', '-v7');
fprintf('  [PPSS_EM] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPHybridFilter_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs and re-run
% nstat.decoding.PPHF.PPHybridFilter. The fixture stores beta1/beta2 as
% scalars — synthesise per-regime CIF objects with intercept 0 and the
% stored slope (matching the Python recipe convention).
fixturePath = fullfile(fixtureRoot, 'v9_PPHybridFilter.mat');
fprintf('  [PPHybridFilter] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A1 = double(data.A1);
A2 = double(data.A2);
Q1 = double(data.Q1);
Q2 = double(data.Q2);
p_ij = double(data.p_ij);
Mu0 = double(data.Mu0);
dN = double(data.dN);
b1 = double(data.beta1);
b2 = double(data.beta2);
binwidth = double(data.binwidth);

% CIF needs beta as a 1xN row vector, and Xnames intercept must be a valid
% MATLAB symbolic variable ('one' not '1') — see PPDecode_update note.
cif1 = CIF([0, b1], {'one', 'x1'}, {'x1'}, 'poisson');
cif2 = CIF([0, b2], {'one', 'x1'}, {'x1'}, 'poisson');

% Upstream MATLAB bug (cajigaslab/nSTAT#91) was fixed in commit 49a84d6:
% the 4th declared output was renamed MU_s -> MU_u to match the function
% body (which only ever assigned MU_u). The full 7-output signature now
% is [S_est, X, W, MU_u, X_s, W_s, pNGivenS] — identical to the sister
% PPHybridFilterLinear API. v12 iter 57 expands the capture to all 7
% outputs; X_s and W_s are 1xnmodels cell arrays (per-regime estimates).
%
% The saved field name ``MU_s`` is retained for backward compatibility
% with downstream Python recipes that referenced the original header
% name; its value is whatever upstream now assigns (i.e. MU_u).
rng(42); %#ok<RNG>
[S_est, X, W, MU_u, X_s, W_s, pNGivenS] = nstat.decoding.PPHF.PPHybridFilter( ...
    {A1, A2}, {Q1, Q2}, p_ij, Mu0, dN, {cif1, cif2}, binwidth);

% Persist the 4 additional outputs. X_s and W_s are cell arrays per
% regime; break them into X_s_1/X_s_2 and W_s_1/W_s_2 to mirror the
% PPHybridFilterLinear capture convention (v9 iter 50A).
if iscell(X_s)
    X_s_1 = X_s{1};
    X_s_2 = X_s{2};
else
    X_s_1 = X_s(:, :, :, 1);
    X_s_2 = X_s(:, :, :, 2);
end
if iscell(W_s)
    W_s_1 = W_s{1};
    W_s_2 = W_s{2};
else
    W_s_1 = W_s(:, :, :, 1);
    W_s_2 = W_s(:, :, :, 2);
end
MU_s = MU_u; %#ok<NASGU> backward-compat alias
beta1 = b1; %#ok<NASGU>
beta2 = b2; %#ok<NASGU>
save(fixturePath, 'A1', 'A2', 'Q1', 'Q2', 'p_ij', 'Mu0', 'dN', ...
    'beta1', 'beta2', 'binwidth', ...
    'S_est', 'X', 'W', 'MU_s', 'MU_u', ...
    'X_s_1', 'X_s_2', 'W_s_1', 'W_s_2', 'pNGivenS', '-v7');
fprintf('  [PPHybridFilter] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_PPHybridFilterLinear_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50A: re-load committed inputs and re-run
% nstat.decoding.PPHF.PPHybridFilterLinear. Deterministic; tolerance is
% already tightened to round-off-ish.
fixturePath = fullfile(fixtureRoot, 'v9_PPHybridFilterLinear.mat');
fprintf('  [PPHybridFilterLinear] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A1 = double(data.A1);
A2 = double(data.A2);
Q1 = double(data.Q1);
Q2 = double(data.Q2);
p_ij = double(data.p_ij);
Mu0 = double(data.Mu0);
dN = double(data.dN);
mu = double(data.mu);
beta = double(data.beta);
binwidth = double(data.binwidth);
fitType = char(string(data.fitType));

rng(42); %#ok<RNG>
[S_est, X, W, MU_u, X_s, W_s, pNGivenS] = ...
    nstat.decoding.PPHF.PPHybridFilterLinear({A1, A2}, {Q1, Q2}, p_ij, ...
        Mu0, dN, mu, beta, fitType, binwidth);

% Match the committed fixture's expanded field set: X_s/W_s are 1xMx10
% tensors per regime (X_s_1, X_s_2 and W_s_1, W_s_2). PPHybridFilterLinear
% returns X_s and W_s as cell arrays {regime1, regime2}.
if iscell(X_s)
    X_s_1 = X_s{1};
    X_s_2 = X_s{2};
else
    X_s_1 = X_s(:, :, :, 1);
    X_s_2 = X_s(:, :, :, 2);
end
if iscell(W_s)
    W_s_1 = W_s{1};
    W_s_2 = W_s{2};
else
    W_s_1 = W_s(:, :, :, 1);
    W_s_2 = W_s(:, :, :, 2);
end

save(fixturePath, 'A1', 'A2', 'Q1', 'Q2', 'p_ij', 'Mu0', 'dN', 'mu', ...
    'beta', 'binwidth', 'fitType', ...
    'S_est', 'X', 'W', 'MU_u', 'X_s_1', 'X_s_2', 'W_s_1', 'W_s_2', ...
    'pNGivenS', '-v7');
fprintf('  [PPHybridFilterLinear] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_simulateCIFByThinning_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: lambda_time, lambda_data, stim_data, ens_data, mu_val,
%   Ts_val, nReal, spikeTimes_r1.
% NOTE: Case C — thinning is RNG-sensitive.  The Python recipe in
% ``tools/parity/numerical_drift.py`` (`_recipe_v9_simulate_cif_thinning`)
% does NOT consume the realized thinning spike times — it only compares
% its own ``CIF.simulateCIF(..., return_lambda=True)`` lambda trace
% (computed with hist=[-1.0], stim=[1.0], ens=[0.0]) against the saved
% ``lambda_data``.  Python's _simulateCIF_python applies the stim kernel
% as a one-step lag and feeds back bernoulli-realized spikes via the
% history coefficient, so the realized lambda carries a small RNG-driven
% residual on top of the deterministic ``exp(mu + sd)`` envelope.
%
% v11 iter 50C rebaseline: the previously committed ``lambda_data`` was
% pre-C4-audit ``rate_hz = lambda_delta/dt`` (values ~50-120 with mu=-3
% and Ts=1e-5).  That convention was removed in audit C4 (`cif.py` lines
% 1294-1307).  Rebaseline to the post-C4 deterministic kernel
% ``exp(mu + 1.0*stim)``.  Residual drift is tracked in
% ``parity/matlab_defects.yml`` (id v11-iter50C-simulateCIFByThinning-rebaseline)
% and the spec tolerance is tightened in numerical_drift_spec.yml.
rng(42); %#ok<RNG>
T = 0.05;
sr = 1000;
lambda_time = (0:1/sr:T)';
stim_data = sin(2*pi*5*lambda_time);
ens_data = zeros(size(lambda_time));
mu_val = -3.0;
% Deterministic baseline lambda = exp(mu + 1.0*stim) — matches the python
% recipe's coefficient triple (hist=-1, stim=1, ens=0) in the absence of
% history feedback.
lambda_data = exp(mu_val + stim_data);
Ts_val = 1/sr;
nReal = 1;
% Capture one MATLAB thinning realization for posterity (not consumed by
% the Python comparison).
lambdaBound = max(lambda_data);
N = ceil(lambdaBound * (1.5 * T));
N = max(N, 1);
u = rand(1, N);
w = -log(u) ./ lambdaBound;
tSpikes = cumsum(w);
tSpikes = tSpikes(tSpikes <= T);
if isempty(tSpikes)
    spikeTimes_r1 = zeros(0, 1);
else
    lambdaRatio = interp1(lambda_time, lambda_data, tSpikes(:)) ./ lambdaBound;
    u2 = rand(numel(lambdaRatio), 1);
    spikeTimes_r1 = tSpikes(lambdaRatio >= u2);
    spikeTimes_r1 = spikeTimes_r1(:);
end
out = fullfile(fixtureRoot, 'v9_simulateCIFByThinning.mat');
save(out, 'lambda_time', 'lambda_data', 'stim_data', 'ens_data', ...
    'mu_val', 'Ts_val', 'nReal', 'spikeTimes_r1', '-v7');
fprintf('  [simulateCIFByThinning] saved %s\n', out);
end %#ok<DEFNU>


function export_v9_simulateCIFByThinningFromLambda_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: lambda_time, lambda_data, lambdaBound (baseline scalar),
%   numRealizations, minTime, maxTime, maxTimeRes, nRealizations_out,
%   spikeTimes_r1, spikeTimes_r2.
% NOTE: The Python recipe in ``tools/parity/numerical_drift.py``
% (`_recipe_v9_simulate_cif_lambda`) only reads ``lambda_data`` and
% ``lambdaBound`` and asserts ``max(ld) == lambdaBound`` — a pure
% deterministic round-off check.  Rebaseline keeps the existing committed
% lambda envelope shape (range 5-15, lambdaBound = 15) so downstream
% callers that load the fixture for spike-time references are unaffected.
%
% v11 iter 50C: regenerated deterministically.  Tolerance stays at
% rtol=1e-6/atol=1e-8 (already passes at 0/0).  Spike-time fields are
% captured via the canonical
% ``CIF.simulateCIFByThinningFromLambda`` for posterity but are not
% consumed by the Python comparison (RNG stream divergence — Case C).
rng(42); %#ok<RNG>
maxTime = 1.0;
minTime = 0.0;
sr = 100;
lambda_time = (0:1/sr:maxTime)';
% Match the committed envelope: ld = 10 + 5*sin(2*pi*lt) -> range 5..15.
lambda_data = 10 + 5*sin(2*pi*lambda_time);
lambdaBound = max(lambda_data);
numRealizations = 2;
maxTimeRes = 1/sr;
% Run the canonical MATLAB simulator to capture sample spike trains
% (informational; not consumed by drift comparison).
lambdaCov = Covariate(lambda_time, lambda_data, ...
    '\lambda', 'time', 's', 'Hz', {'\lambda'});
lambdaCov.setMinTime(minTime);
lambdaCov.setMaxTime(maxTime);
spikeTrainColl = CIF.simulateCIFByThinningFromLambda( ...
    lambdaCov, numRealizations, maxTimeRes);
nRealizations_out = spikeTrainColl.numSpikeTrains;
spikeTimes_r1 = zeros(0, 1);
spikeTimes_r2 = zeros(0, 1);
if nRealizations_out >= 1
    st = spikeTrainColl.getNST(1);
    spikeTimes_r1 = st.spikeTimes(:);
end
if nRealizations_out >= 2
    st = spikeTrainColl.getNST(2);
    spikeTimes_r2 = st.spikeTimes(:);
end
out = fullfile(fixtureRoot, 'v9_simulateCIFByThinningFromLambda.mat');
save(out, 'lambda_time', 'lambda_data', 'lambdaBound', ...
    'numRealizations', 'minTime', 'maxTime', 'maxTimeRes', ...
    'nRealizations_out', 'spikeTimes_r1', 'spikeTimes_r2', '-v7');
fprintf('  [simulateCIFByThinningFromLambda] saved %s\n', out);
end %#ok<DEFNU>


function export_v9_raisedCosine_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed inputs (K, tMin, tMax) and re-run
% History.raisedCosine. Fully deterministic — byte-equivalent reproduction
% expected.
fixturePath = fullfile(fixtureRoot, 'v9_raisedCosine.mat');
fprintf('  [raisedCosine] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

K = double(data.K);
tMin = double(data.tMin);
tMax = double(data.tMax);

rng(42); %#ok<RNG>
h = History.raisedCosine(K, tMin, tMax);
windowTimes = h.windowTimes;

save(fixturePath, 'K', 'tMin', 'tMax', 'windowTimes', '-v7');
fprintf('  [raisedCosine] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_fitresult_KSPlot_data_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed inputs (spikeTimes, t, lambdaData,
% sampleRate, minTime, maxTime) and re-run Analysis.computeKSStats — there
% is no standalone FitResult.KSPlot_data method in MATLAB; KSStats are
% populated via Analysis.computeKSStats (FitResult.m lines 795+, Python
% nstat/analysis.py:876). The Python recipe routes through the same call.
%
% Case C — non-trivial drift remains (rtol/atol=1e-1) because the rescaled-
% time Z values diverge slightly between MATLAB's cumulative integration of
% lambda and the Python port's scheme. This is a pre-existing analysis-
% subsystem drift, not a fixture-recipe defect; reproduction here uses the
% MATLAB function verbatim so the baseline is canonical.
fixturePath = fullfile(fixtureRoot, 'v9_fitresult_KSPlot_data.mat');
fprintf('  [fitresult_KSPlot_data] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

spikeTimes = double(data.spikeTimes);
t = double(data.t);
lambdaData = double(data.lambdaData);
sampleRate = double(data.sampleRate);
minTime = double(data.minTime);
maxTime = double(data.maxTime);

st = nspikeTrain(spikeTimes, '1', sampleRate, minTime, maxTime, ...
    'time', 's', '', '', -1);
lam = Covariate(t, lambdaData, '\lambda(t)', 'time', 's', 'Hz', ...
    {'\lambda_{1}'});

rng(42); %#ok<RNG>
[Z, U, xAxis, KSSorted, ks_stat] = Analysis.computeKSStats(st, lam, 1);

save(fixturePath, 'spikeTimes', 't', 'lambdaData', 'sampleRate', ...
    'minTime', 'maxTime', 'Z', 'U', 'xAxis', 'KSSorted', 'ks_stat', '-v7');
fprintf('  [fitresult_KSPlot_data] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_fitresult_invGausTrans_data_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed input Z and recompute the inverse-Gaussian
% transform X = norminv(1 - exp(-Z)). Pure-math; tolerance is at float64
% round-off (rtol=1e-12, atol=1e-14).
%
% The fixture also stores auxiliary fields (conf_data, conf_time, rho_data,
% rho_time) from a co-captured ConfidenceInterval; preserve them verbatim.
fixturePath = fullfile(fixtureRoot, 'v9_fitresult_invGausTrans_data.mat');
fprintf('  [fitresult_invGausTrans_data] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

Z = double(data.Z);
X = norminv(1 - exp(-Z));

% Preserve auxiliary fields verbatim.
conf_data = data.conf_data; %#ok<NASGU>
conf_time = data.conf_time; %#ok<NASGU>
rho_data = data.rho_data; %#ok<NASGU>
rho_time = data.rho_time; %#ok<NASGU>

save(fixturePath, 'Z', 'X', 'conf_data', 'conf_time', 'rho_data', ...
    'rho_time', '-v7');
fprintf('  [fitresult_invGausTrans_data] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_fitresult_seqCorrCoeff_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed input U and recompute the sequential
% correlation coefficient rho = corrcoef(uj, uj1). Pure-math; tolerance is
% at float64 round-off (rtol=1e-12, atol=1e-14).
%
% The fixture also stores auxiliary uj, uj1, pval; re-derive uj/uj1 from U
% and capture pval from corrcoef's second output.
fixturePath = fullfile(fixtureRoot, 'v9_fitresult_seqCorrCoeff.mat');
fprintf('  [fitresult_seqCorrCoeff] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

U = double(data.U);
uj = U(1:end-1);
uj1 = U(2:end);
[R, P] = corrcoef(uj, uj1);
rho = R(1, 2);
pval = P(1, 2);

save(fixturePath, 'U', 'uj', 'uj1', 'rho', 'pval', '-v7');
fprintf('  [fitresult_seqCorrCoeff] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_signalobj_resample_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed inputs (time_in, data_in, newRate) and
% re-run SignalObj.resample. Deterministic; current drift is at float64
% round-off (rtol=1e-9, atol=1e-12 in spec).
fixturePath = fullfile(fixtureRoot, 'v9_signalobj_resample.mat');
fprintf('  [signalobj_resample] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

time_in = double(data.time_in);
data_in = double(data.data_in);
newRate = double(data.newRate);

rng(42); %#ok<RNG>
sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
sig2 = sig.resample(newRate);
data_out = sig2.data;
time_out = sig2.time;
sampleRate_out = sig2.sampleRate;

save(fixturePath, 'time_in', 'data_in', 'newRate', 'data_out', ...
    'time_out', 'sampleRate_out', '-v7');
fprintf('  [signalobj_resample] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_signalobj_derivative_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed inputs (time_in, data_in) and re-run
% SignalObj.derivative. Deterministic; tolerance at float64 round-off.
fixturePath = fullfile(fixtureRoot, 'v9_signalobj_derivative.mat');
fprintf('  [signalobj_derivative] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

time_in = double(data.time_in);
data_in = double(data.data_in);

rng(42); %#ok<RNG>
sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
sig2 = sig.derivative();
data_out = sig2.data;
time_out = sig2.time;
sampleRate_in = sig.sampleRate;

save(fixturePath, 'time_in', 'data_in', 'data_out', 'time_out', ...
    'sampleRate_in', '-v7');
fprintf('  [signalobj_derivative] saved %s\n', fixturePath);
end %#ok<DEFNU>


function export_v9_signalobj_integral_fixture(fixtureRoot) %#ok<DEFNU>
% v11 iter 50D: re-load committed inputs (time_in, data_in) and re-run
% SignalObj.integral. Deterministic; tolerance at float64 round-off.
fixturePath = fullfile(fixtureRoot, 'v9_signalobj_integral.mat');
fprintf('  [signalobj_integral] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

time_in = double(data.time_in);
data_in = double(data.data_in);

rng(42); %#ok<RNG>
sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
sig2 = sig.integral();
data_out = sig2.data;
time_out = sig2.time;
sampleRate_in = sig.sampleRate;

save(fixturePath, 'time_in', 'data_in', 'data_out', 'time_out', ...
    'sampleRate_in', '-v7');
fprintf('  [signalobj_integral] saved %s\n', fixturePath);
end %#ok<DEFNU>


function lam = local_v50B_get_lambda(fitObj, idx) %#ok<DEFNU>
% v11 iter 50B helper: extract the idx-th lambda Covariate from a FitResult.
% fit.lambda is a single Covariate when only one configuration was fit, and a
% cell array otherwise. Returns a Covariate regardless of input layout.
if iscell(fitObj.lambda)
    lam = fitObj.lambda{idx};
else
    lam = fitObj.lambda;
end
end %#ok<DEFNU>
