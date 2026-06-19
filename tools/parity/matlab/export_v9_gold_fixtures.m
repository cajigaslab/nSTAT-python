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

% --- analysis ---
export_v9_RunAnalysisForNeuron_fixture(fixtureRoot);
export_v9_computeKSStats_full_fixture(fixtureRoot);
export_v9_computeHistLag_fixture(fixtureRoot);
export_v9_computeFitResidual_fixture(fixtureRoot);
% --- decoding_PPAF ---
export_v9_PPDecode_predict_fixture(fixtureRoot);
export_v9_PPDecode_update_fixture(fixtureRoot);
% --- decoding_KF ---
export_v9_kalman_smoother_fixture(fixtureRoot);
export_v9_kalman_fixedIntervalSmoother_fixture(fixtureRoot);
% --- decoding_PPSS ---
export_v9_PPSS_EStep_fixture(fixtureRoot);
export_v9_PPSS_MStep_fixture(fixtureRoot);
export_v9_PPSS_EM_fixture(fixtureRoot);
% --- decoding_PPHF ---
export_v9_PPHybridFilter_fixture(fixtureRoot);
export_v9_PPHybridFilterLinear_fixture(fixtureRoot);
% --- cif ---
export_v9_simulateCIFByThinning_fixture(fixtureRoot);
export_v9_simulateCIFByThinningFromLambda_fixture(fixtureRoot);
% --- history_fit_core ---
export_v9_raisedCosine_fixture(fixtureRoot);
% --- FitResult helpers ---
export_v9_fitresult_KSPlot_data_fixture(fixtureRoot);
export_v9_fitresult_invGausTrans_data_fixture(fixtureRoot);
export_v9_fitresult_seqCorrCoeff_fixture(fixtureRoot);
% --- SignalObj ---
export_v9_signalobj_resample_fixture(fixtureRoot);
export_v9_signalobj_derivative_fixture(fixtureRoot);
export_v9_signalobj_integral_fixture(fixtureRoot);

end


% =========================================================================
% Per-fixture stubs — each documents the saved field set expected by the
% Python recipe in tools/parity/numerical_drift.py. The early `return`
% prevents accidental overwrite of the committed fixture; remove it after
% verifying the recipe in MATLAB.
% =========================================================================


function export_v9_RunAnalysisForNeuron_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_sample_rate, coeffs (baseline).
% Recipe sketch:
%   rng(42); T = 1.0; sr = 100; t = (0:1/sr:T)';
%   stim_data = sin(2*pi*5*t);
%   spk = t(rand(numel(t),1) < 0.05*stim_data + 0.1);
%   stim = Covariate(t, stim_data, 'Stimulus', 'time', 's', '', {'stim'});
%   st = nspikeTrain(spk, '1', sr, 0, T, 'time', 's', '', '', -1);
%   trial = Trial(nstColl({st}), CovColl({stim}));
%   cfg = TrialConfig({{'Stimulus','stim'}}, sr, [], []);
%   fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}), 0);
%   coeffs = fit.getCoeffs(1);
%   save(.., 'input_time', 'input_stim_data', 'input_spike_times', ...
%        'input_sample_rate', 'coeffs', '-v7');
fprintf('  [TODO] export_v9_RunAnalysisForNeuron_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_computeKSStats_full_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: input_time, input_spike_times, input_DTCorrection,
%   lambda_time, lambda_data, lambda_sample_rate, Z (baseline).
% Recipe sketch:
%   rng(42); T=1.0; sr=100; lt = (0:1/sr:T)';
%   ld = 10*ones(size(lt));
%   spk = lt(rand(numel(lt),1) < ld/sr);
%   lam = Covariate(lt, ld, '\lambda(t)', 'time', 's', 'Hz', {'\lambda_1'});
%   st = nspikeTrain(spk, '1', sr, 0, T, 'time', 's', '', '', -1);
%   [Z, U, xAxis, KSSorted, ks_stat] = Analysis.computeKSStats(st, lam, 1);
%   save(.., 'Z', .., '-v7');
fprintf('  [TODO] export_v9_computeKSStats_full_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_computeHistLag_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: input_time, input_stim_data, input_spike_times,
%   input_windowTimes, input_neuronNum, input_sampleRate, fit0_coeffs.
% Recipe sketch:
%   rng(42); T=1.0; sr=100; t=(0:1/sr:T)';
%   sd = sin(2*pi*5*t);
%   spk = t(rand(numel(t),1) < 0.05*sd + 0.1);
%   wt = [0 0.02 0.05];
%   stim = Covariate(t, sd, 'Stimulus', 'time', 's', '', {'stim'});
%   st = nspikeTrain(spk, '1', sr, 0, T, 'time', 's', '', '', -1);
%   trial = Trial(nstColl({st}), CovColl({stim}));
%   fits = Analysis.computeHistLag(trial, 1, wt, {{'Stimulus','stim'}}, sr, 0);
%   fit0_coeffs = fits(1).getCoeffs(1);
%   save(.., 'fit0_coeffs', .., '-v7');
fprintf('  [TODO] export_v9_computeHistLag_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_computeFitResidual_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: input_time, input_spike_times, input_windowSize,
%   lambda_time, lambda_data, lambda_sample_rate, M_data (baseline).
% NOTE: Case-C entry — Python bins at windowSize, MATLAB at lambda sample
% rate; recipe in numerical_drift.py compares sum-of-squared-residuals.
% Recipe sketch:
%   rng(42); T=1.0; sr=100; lt=(0:1/sr:T)'; ld = 10*ones(size(lt));
%   spk = lt(rand(numel(lt),1) < ld/sr);
%   ws = 0.05;
%   st = nspikeTrain(spk, '1', sr, 0, T, 'time', 's', '', '', -1);
%   lam = Covariate(lt, ld, '\lambda(t)', 'time', 's', 'Hz', {'\lambda_1'});
%   M = Analysis.computeFitResidual(st, lam, ws);
%   save(.., 'M_data', M.data, .., '-v7');
fprintf('  [TODO] export_v9_computeFitResidual_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPDecode_predict_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: x_u, W_u, A, Q, x_p (baseline).
% Recipe sketch:
%   rng(42); nx=2;
%   x_u = randn(nx,1); W_u = 0.1*eye(nx);
%   A = 0.95*eye(nx); Q = 0.01*eye(nx);
%   [x_p, W_p] = PPDecode_predict(x_u, W_u, A, Q);
%   save(.., 'x_p', .., '-v7');
fprintf('  [TODO] export_v9_PPDecode_predict_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPDecode_update_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: x_p, W_p, dN, beta, binwidth, time_index, x_u (baseline).
% Recipe sketch:
%   rng(42); nx=2; nc=1;
%   x_p = randn(nx,1); W_p = 0.1*eye(nx);
%   dN = double(rand(nc,1) < 0.1);
%   beta = 0.1*randn(2,nc); % [intercept; x1] convention
%   binwidth = 0.01; time_index = 1; % MATLAB 1-indexed
%   cif = CIF(beta, {'1','x1'}, {'x1'}, 'poisson');
%   [x_u, W_u, lam] = PPDecode_update(x_p, W_p, dN, {cif}, binwidth, time_index);
%   save(.., 'x_u', .., '-v7');
fprintf('  [TODO] export_v9_PPDecode_update_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_kalman_smoother_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A, C, Pv, Pw, Px0, x0, y, x_N (baseline).
% NOTE: tightened to atol=1e-2 in v10 iter 47 (was 1e-1).
% Recipe sketch:
%   rng(42); nx=2; ny=2; T=10;
%   A = 0.9*eye(nx); C = eye(ny,nx);
%   Pv = 0.01*eye(nx); Pw = 0.1*eye(ny);
%   x0 = zeros(nx,1); Px0 = 0.1*eye(nx);
%   y = randn(ny, T);
%   [x_N, ...] = kalman_smoother(A, C, Pv, Pw, Px0, x0, y);
fprintf('  [TODO] export_v9_kalman_smoother_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_kalman_fixedIntervalSmoother_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A, C, Pv, Pw, Px0, x0, y, lags, x_uLag (baseline).
% NOTE: tightened to STRICT (rtol=1e-6, atol=1e-10) in v10 iter 47 — now at
% float64 round-off after upstream rebaseline.
% Recipe sketch:
%   rng(42); nx=2; ny=2; T=10; lags=2;
%   A = 0.9*eye(nx); C = eye(ny,nx);
%   Pv = 0.01*eye(nx); Pw = 0.1*eye(ny);
%   x0 = zeros(nx,1); Px0 = 0.1*eye(nx);
%   y = randn(ny, T);
%   [x_pLag, P_pLag, x_uLag, P_uLag] = kalman_fixedIntervalSmoother( ...
%       A, C, Pv, Pw, Px0, x0, y, lags);
fprintf('  [TODO] export_v9_kalman_fixedIntervalSmoother_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPSS_EStep_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A, Q, x0, dN, HkAll, fitType, delta, gamma, numBasis, x_K.
% Recipe sketch:
%   rng(42); nx=3; nc=2; T=2; nb=3;
%   A = 0.95*eye(nx); Q = 0.01*ones(nx,1);
%   x0 = zeros(nx,1);
%   dN = double(rand(nc, T) < 0.1);
%   HkAll = cell(nc,1); for k=1:nc; HkAll{k}=zeros(T,nb); end
%   gamma = zeros(nc,nb);
%   [x_K, ...] = PPSS_EStep(A, Q, x0, dN, HkAll, 'poisson', 1.0, gamma, nb);
fprintf('  [TODO] export_v9_PPSS_EStep_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPSS_MStep_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: dN, HkAll, fitType, x_K, W_K, gamma, delta, sumXkTerms,
%   windowTimes, Qhat (baseline).
% Recipe sketch:
%   [run PPSS_EStep first, see above]
%   [pack EStep outputs as x_K, W_K, sumXkTerms]
%   wt = [0 0.02 0.05];
%   [Qhat, gamma_new] = PPSS_MStep(dN, HkAll, 'poisson', x_K, W_K, ...
%       gamma, 1.0, sumXkTerms, wt);
fprintf('  [TODO] export_v9_PPSS_MStep_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPSS_EM_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A, Q0, x0, dN, fitType, delta, gamma0, windowTimes,
%   numBasis, HkAll, xKFinal (baseline).
% NOTE: Case C — RNG-sensitive convergence; tolerance stays at 1e+1 / 1e+0.
% Recipe sketch:
%   rng(42); nx=3; nc=2; T=2; nb=3;
%   A = 0.95*eye(nx); Q0 = 0.01*ones(nx,1);
%   x0 = zeros(nx,1);
%   dN = double(rand(nc,T) < 0.1);
%   gamma0 = zeros(nc,nb); wt = [0 0.02 0.05];
%   HkAll = cell(nc,1); for k=1:nc; HkAll{k}=zeros(T,nb); end
%   [xKFinal, ...] = PPSS_EM(A, Q0, x0, dN, 'poisson', 1.0, gamma0, wt, nb, HkAll);
fprintf('  [TODO] export_v9_PPSS_EM_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPHybridFilter_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A1, A2, Q1, Q2, p_ij, Mu0, dN, beta1, beta2, binwidth,
%   X (baseline — shape depends on impl).
% Recipe sketch:
%   rng(42); T=10;
%   A1=0.95; A2=0.99; Q1=0.01; Q2=0.001;
%   p_ij = [0.99 0.01; 0.01 0.99]; Mu0 = [0.5; 0.5];
%   dN = double(rand(1,T) < 0.1);
%   beta1 = 0.1; beta2 = 0.2; binwidth = 0.01;
%   cif1 = CIF([0; beta1], {'1','x1'}, {'x1'}, 'poisson');
%   cif2 = CIF([0; beta2], {'1','x1'}, {'x1'}, 'poisson');
%   [..., X, ...] = PPHybridFilter({A1,A2}, {Q1,Q2}, p_ij, Mu0, dN, {cif1,cif2}, binwidth);
fprintf('  [TODO] export_v9_PPHybridFilter_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_PPHybridFilterLinear_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: A1, A2, Q1, Q2, p_ij, Mu0, dN, mu, beta, binwidth, fitType,
%   MU_u (baseline; 2 x T mixing posterior).
% NOTE: tightened to rtol=1e-4, atol=1e-5 in v10 iter 47 — deterministic
% given inputs, observed error ~6e-6.
% Recipe sketch:
%   rng(42); T=10;
%   A1=0.95; A2=0.99; Q1=0.01; Q2=0.001;
%   p_ij = [0.99 0.01; 0.01 0.99]; Mu0 = [0.5; 0.5];
%   dN = double(rand(1,T) < 0.1);
%   mu = -2.5; beta = [0.1, 0.2]; binwidth = 0.01;
%   [..., MU_u, ...] = PPHybridFilterLinear({A1,A2}, {Q1,Q2}, p_ij, Mu0, ...
%       dN, mu, beta, 'poisson', binwidth);
fprintf('  [TODO] export_v9_PPHybridFilterLinear_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_simulateCIFByThinning_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: lambda_time, lambda_data, stim_data, ens_data, mu_val,
%   Ts_val, nReal.
% NOTE: Case C — thinning is RNG-sensitive.
% Recipe sketch:
%   rng(42); T=0.5; sr=100; lt=(0:1/sr:T)';
%   sd = sin(2*pi*5*lt); ed = zeros(size(lt));
%   ld = exp(-3 + 0.1*sd); mu = -3.0; Ts = 1/sr; nReal = 1;
%   save(.., 'lambda_data', ld, .., '-v7');
fprintf('  [TODO] export_v9_simulateCIFByThinning_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_simulateCIFByThinningFromLambda_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: lambda_time, lambda_data, lambdaBound (baseline scalar).
% Recipe sketch:
%   rng(42); T=0.5; sr=100; lt=(0:1/sr:T)';
%   ld = exp(-3 + 0.5*sin(2*pi*5*lt));
%   lambdaBound = max(ld);
%   save(.., 'lambda_time','lambda_data','lambdaBound', '-v7');
fprintf('  [TODO] export_v9_simulateCIFByThinningFromLambda_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_raisedCosine_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: K, tMin, tMax, windowTimes (baseline).
% Recipe sketch:
%   K = 6; tMin = 0.0; tMax = 0.1;
%   h = History.raisedCosine(K, tMin, tMax);
%   windowTimes = h.windowTimes;
%   save(.., 'K','tMin','tMax','windowTimes', '-v7');
fprintf('  [TODO] export_v9_raisedCosine_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_fitresult_KSPlot_data_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: spikeTimes, t, lambdaData, sampleRate, minTime, maxTime,
%   KSSorted (baseline).
% NOTE: Case C — KSPlot_data helper isn't ported; Python routes through
% Analysis.computeKSStats. Tolerance kept at 1e-1.
% Recipe sketch:
%   rng(42); T=1.0; sr=100; t=(0:1/sr:T)';
%   lambdaData = 5*ones(size(t));
%   spikeTimes = t(rand(numel(t),1) < lambdaData/sr);
%   FR = FitResult(...); % construct minimally as needed
%   [Z, U, ks_stat, KSSorted] = FR.KSPlot_data(...);
fprintf('  [TODO] export_v9_fitresult_KSPlot_data_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_fitresult_invGausTrans_data_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: Z, X (baseline).
% Recipe (pure-math; tightened to round-off):
%   rng(42); Z = rand(4,1)*2;
%   X = norminv(1 - exp(-Z));
%   save(.., 'Z','X', '-v7');
fprintf('  [TODO] export_v9_fitresult_invGausTrans_data_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_fitresult_seqCorrCoeff_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: U, rho (baseline scalar).
% Recipe (pure-math; tightened to round-off):
%   rng(42); U = rand(10,1);
%   uj = U(1:end-1); uj1 = U(2:end);
%   R = corrcoef(uj, uj1); rho = R(1,2);
%   save(.., 'U','rho', '-v7');
fprintf('  [TODO] export_v9_fitresult_seqCorrCoeff_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_signalobj_resample_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: time_in, data_in, newRate, data_out (baseline).
% Recipe sketch:
%   rng(42); t = (0:0.01:0.1)'; d = randn(numel(t), 2);
%   sig = SignalObj(t, d, 'x', 't', 's', '', {'c0','c1'});
%   sig2 = sig.resample(50);
%   save(.., 'time_in', t, 'data_in', d, 'newRate', 50, ...
%        'data_out', sig2.data, '-v7');
fprintf('  [TODO] export_v9_signalobj_resample_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_signalobj_derivative_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: time_in, data_in, data_out (baseline).
% Recipe sketch:
%   rng(42); t = (0:0.01:0.09)'; d = randn(numel(t), 2);
%   sig = SignalObj(t, d, 'x', 't', 's', '', {'c0','c1'});
%   sig2 = sig.derivative();
%   save(.., 'time_in', t, 'data_in', d, 'data_out', sig2.data, '-v7');
fprintf('  [TODO] export_v9_signalobj_derivative_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_v9_signalobj_integral_fixture(fixtureRoot) %#ok<DEFNU>
% Saved fields: time_in, data_in, data_out (baseline).
% Recipe sketch:
%   rng(42); t = (0:0.01:0.09)'; d = randn(numel(t), 2);
%   sig = SignalObj(t, d, 'x', 't', 's', '', {'c0','c1'});
%   sig2 = sig.integral();
%   save(.., 'time_in', t, 'data_in', d, 'data_out', sig2.data, '-v7');
fprintf('  [TODO] export_v9_signalobj_integral_fixture: unverified; skipping\n');
return  %#ok<UNRCH>
end %#ok<DEFNU>
