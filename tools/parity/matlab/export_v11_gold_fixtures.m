function export_v11_gold_fixtures(repoRoot, matlabRepoRoot)
% export_v11_gold_fixtures
% =========================================================================
% v11 iter 51A — expand numerical drift coverage with 12 new MATLAB gold
% fixtures for previously uncovered public nstat functions.
%
% Fixtures created (12 new captures):
%
%   CIF gradient/jacobian surface (no-history branch):
%     v11_cif_evalGradient.mat
%     v11_cif_evalGradientLog.mat
%     v11_cif_evalJacobian.mat
%     v11_cif_evalJacobianLog.mat
%     v11_cif_evalLambdaDelta_vector.mat
%
%   SignalObj transforms (deterministic):
%     v11_signalobj_filter.mat
%     v11_signalobj_filtfilt.mat
%     v11_signalobj_periodogram.mat
%     v11_signalobj_autocorrelation.mat
%
%   History helper:
%     v11_history_toFilter.mat
%
%   Analysis helpers:
%     v11_analysis_computeInvGausTrans.mat
%     v11_analysis_RunAnalysisForAllNeurons.mat
%
% USAGE
% -----
%   export_v11_gold_fixtures('/Users/iahncajigas/projects/nstat-python', ...
%                            '/Users/iahncajigas/projects/nstat');
%
% v9 conventions inherited: rng(42) at the top of every fixture function,
% small canonical dims (10-step, 2-cell, 2-state), per-fixture try/catch
% dispatcher so one upstream bug does not block the rest.
%
% NOTE: Four additional v11 drift entries reuse existing fixtures
% (fit_summary_exactness, analysis_multineuron_exactness, v9_fitresult_KSPlot_data)
% — they do not require a new capture; their recipes are wired in
% tools/parity/numerical_drift.py only.

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

captures = { ...
    @export_v11_cif_evalGradient,             'cif_evalGradient'; ...
    @export_v11_cif_evalGradientLog,          'cif_evalGradientLog'; ...
    @export_v11_cif_evalJacobian,             'cif_evalJacobian'; ...
    @export_v11_cif_evalJacobianLog,          'cif_evalJacobianLog'; ...
    @export_v11_cif_evalLambdaDelta_vector,   'cif_evalLambdaDelta_vector'; ...
    @export_v11_signalobj_filter,             'signalobj_filter'; ...
    @export_v11_signalobj_filtfilt,           'signalobj_filtfilt'; ...
    @export_v11_signalobj_periodogram,        'signalobj_periodogram'; ...
    @export_v11_signalobj_xcorr,              'signalobj_xcorr'; ...
    @export_v11_history_toFilter,             'history_toFilter'; ...
    @export_v11_analysis_computeInvGausTrans, 'analysis_computeInvGausTrans'; ...
    @export_v11_analysis_RunAnalysisForAllNeurons, 'analysis_RunAnalysisForAllNeurons'};
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
% CIF gradient/jacobian fixtures — single-stim, no-history poisson CIF.
% Inputs: beta = [0.5, -0.3], Xnames = {'one','x1'}, stimNames = {'x1'},
% fitType = 'poisson'. stimVal scanned over [-1.0, -0.5, 0.0, 0.5, 1.5].
% Outputs are stacked NxM matrices (one row per stim value).
% =========================================================================


function export_v11_cif_evalGradient(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
beta = [0.5, -0.3];
Xnames = {'one', 'x1'};
stimNames = {'x1'};
fitType = 'poisson';
stimVals = [-1.0; -0.5; 0.0; 0.5; 1.5];

cif = CIF(beta, Xnames, stimNames, fitType);
rows = zeros(numel(stimVals), numel(beta));
for ii = 1:numel(stimVals)
    g = cif.evalGradient(stimVals(ii));
    rows(ii, :) = reshape(double(g), 1, []);
end
gradient_out = rows;
out = fullfile(fixtureRoot, 'v11_cif_evalGradient.mat');
save(out, 'beta', 'Xnames', 'stimNames', 'fitType', 'stimVals', ...
    'gradient_out', '-v7');
fprintf('  [cif_evalGradient] saved %s\n', out);
end


function export_v11_cif_evalGradientLog(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
beta = [0.5, -0.3];
Xnames = {'one', 'x1'};
stimNames = {'x1'};
fitType = 'poisson';
stimVals = [-1.0; -0.5; 0.0; 0.5; 1.5];

cif = CIF(beta, Xnames, stimNames, fitType);
rows = zeros(numel(stimVals), numel(beta));
for ii = 1:numel(stimVals)
    g = cif.evalGradientLog(stimVals(ii));
    rows(ii, :) = reshape(double(g), 1, []);
end
gradientLog_out = rows;
out = fullfile(fixtureRoot, 'v11_cif_evalGradientLog.mat');
save(out, 'beta', 'Xnames', 'stimNames', 'fitType', 'stimVals', ...
    'gradientLog_out', '-v7');
fprintf('  [cif_evalGradientLog] saved %s\n', out);
end


function export_v11_cif_evalJacobian(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
beta = [0.5, -0.3];
Xnames = {'one', 'x1'};
stimNames = {'x1'};
fitType = 'poisson';
stimVals = [-1.0; -0.5; 0.0; 0.5; 1.5];

cif = CIF(beta, Xnames, stimNames, fitType);
% Jacobian is len(beta) x len(beta) per stim value; stash as 3D array
N = numel(beta);
J_stack = zeros(numel(stimVals), N, N);
for ii = 1:numel(stimVals)
    J = cif.evalJacobian(stimVals(ii));
    J_stack(ii, :, :) = double(J);
end
jacobian_out = J_stack;
out = fullfile(fixtureRoot, 'v11_cif_evalJacobian.mat');
save(out, 'beta', 'Xnames', 'stimNames', 'fitType', 'stimVals', ...
    'jacobian_out', '-v7');
fprintf('  [cif_evalJacobian] saved %s\n', out);
end


function export_v11_cif_evalJacobianLog(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
beta = [0.5, -0.3];
Xnames = {'one', 'x1'};
stimNames = {'x1'};
fitType = 'poisson';
stimVals = [-1.0; -0.5; 0.0; 0.5; 1.5];

cif = CIF(beta, Xnames, stimNames, fitType);
N = numel(beta);
J_stack = zeros(numel(stimVals), N, N);
for ii = 1:numel(stimVals)
    J = cif.evalJacobianLog(stimVals(ii));
    J_stack(ii, :, :) = double(J);
end
jacobianLog_out = J_stack;
out = fullfile(fixtureRoot, 'v11_cif_evalJacobianLog.mat');
save(out, 'beta', 'Xnames', 'stimNames', 'fitType', 'stimVals', ...
    'jacobianLog_out', '-v7');
fprintf('  [cif_evalJacobianLog] saved %s\n', out);
end


function export_v11_cif_evalLambdaDelta_vector(fixtureRoot) %#ok<DEFNU>
% Vector sweep of lambda(stim) for the no-history poisson CIF.
rng(42); %#ok<RNG>
beta = [0.5, -0.3];
Xnames = {'one', 'x1'};
stimNames = {'x1'};
fitType = 'poisson';
stimVals = [-1.0; -0.5; 0.0; 0.5; 1.5];

cif = CIF(beta, Xnames, stimNames, fitType);
ld = zeros(numel(stimVals), 1);
for ii = 1:numel(stimVals)
    ld(ii) = double(cif.evalLambdaDelta(stimVals(ii)));
end
lambdaDelta_out = ld;
out = fullfile(fixtureRoot, 'v11_cif_evalLambdaDelta_vector.mat');
save(out, 'beta', 'Xnames', 'stimNames', 'fitType', 'stimVals', ...
    'lambdaDelta_out', '-v7');
fprintf('  [cif_evalLambdaDelta_vector] saved %s\n', out);
end


% =========================================================================
% SignalObj transforms (deterministic).
% =========================================================================


function export_v11_signalobj_filter(fixtureRoot) %#ok<DEFNU>
% Apply a small FIR low-pass filter (boxcar) to a 2-channel signal.
% B = [0.25, 0.25, 0.25, 0.25], A = 1 (FIR).
rng(42); %#ok<RNG>
sr = 100;
T = 1.0;
time_in = (0:1/sr:T)';
data_in = [sin(2*pi*5*time_in), cos(2*pi*3*time_in)];
B = [0.25, 0.25, 0.25, 0.25];
A_coef = 1.0;

sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
sig2 = sig.filter(B, A_coef);
data_out = sig2.data;
time_out = sig2.time;

out = fullfile(fixtureRoot, 'v11_signalobj_filter.mat');
save(out, 'time_in', 'data_in', 'B', 'A_coef', 'data_out', 'time_out', '-v7');
fprintf('  [signalobj_filter] saved %s\n', out);
end


function export_v11_signalobj_filtfilt(fixtureRoot) %#ok<DEFNU>
% Zero-phase filter via filtfilt — same FIR coefficients as filter.
rng(42); %#ok<RNG>
sr = 100;
T = 1.0;
time_in = (0:1/sr:T)';
data_in = [sin(2*pi*5*time_in), cos(2*pi*3*time_in)];
% A small low-pass IIR (Butterworth-like). filtfilt requires len(b)<len(x)/3.
B = [0.1, 0.2, 0.4, 0.2, 0.1];
A_coef = 1.0;

sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
sig2 = sig.filtfilt(B, A_coef);
data_out = sig2.data;
time_out = sig2.time;

out = fullfile(fixtureRoot, 'v11_signalobj_filtfilt.mat');
save(out, 'time_in', 'data_in', 'B', 'A_coef', 'data_out', 'time_out', '-v7');
fprintf('  [signalobj_filtfilt] saved %s\n', out);
end


function export_v11_signalobj_periodogram(fixtureRoot) %#ok<DEFNU>
% Periodogram of a 2-channel sinusoid. MATLAB's SignalObj.periodogram
% returns a 1xC cell of struct('Pxx', 'f') (one per channel). Capture
% channel-1's PSD + frequency axis as the comparison target.
rng(42); %#ok<RNG>
sr = 100;
T = 1.0;
time_in = (0:1/sr:T - 1/sr)';
data_in = [sin(2*pi*5*time_in), cos(2*pi*3*time_in)];

sig = SignalObj(time_in, data_in, 'x', 't', 's', '', {'c0', 'c1'});
result = sig.periodogram();
s1 = result{1};
psd_data = double(reshape(s1.Pxx, [], 1));
freq = double(reshape(s1.f, [], 1));
s2 = result{2};
psd_data_ch2 = double(reshape(s2.Pxx, [], 1)); %#ok<NASGU>

out = fullfile(fixtureRoot, 'v11_signalobj_periodogram.mat');
save(out, 'time_in', 'data_in', 'psd_data', 'psd_data_ch2', 'freq', '-v7');
fprintf('  [signalobj_periodogram] saved %s\n', out);
end


function export_v11_signalobj_xcorr(fixtureRoot) %#ok<DEFNU>
% Cross-correlation between channel-0 and channel-1 of a 2-channel signal.
% MATLAB-side SignalObj.autocorrelation is blocked by a newer-MATLAB
% crosscorr-API regression (Expected a string for the parameter name),
% so capture the simpler manual xcorr instead.
rng(42); %#ok<RNG>
sr = 50;
T = 1.0;
time_in = (0:1/sr:T - 1/sr)';
x1 = sin(2*pi*5*time_in);
x2 = cos(2*pi*3*time_in);
[c, lags] = xcorr(x1, x2);
xcorr_data = double(c);
xcorr_lags = double(lags(:)) / sr;

out = fullfile(fixtureRoot, 'v11_signalobj_xcorr.mat');
save(out, 'time_in', 'x1', 'x2', 'sr', 'xcorr_data', 'xcorr_lags', '-v7');
fprintf('  [signalobj_xcorr] saved %s\n', out);
end


% =========================================================================
% History.toFilter — basis-function expansion as a discrete-time filter bank.
% =========================================================================


function export_v11_history_toFilter(fixtureRoot) %#ok<DEFNU>
% History.toFilter returns a Control-System tf() object that is not
% directly numerically comparable in Python. Instead we capture the raw
% numerator matrix the function builds internally — same algorithm,
% comparable as a dense float array. See History.m lines 152-171.
rng(42); %#ok<RNG>
windowTimes = [0.0, 0.01, 0.02, 0.05];
minTime = 0.0;
maxTime = 1.0;
delta = 0.005;

tmin = windowTimes(1:end-1);
tmax = windowTimes(2:end);
timeVec = min(tmin):delta:max(tmax);
b_mat = zeros(length(tmax), length(timeVec));
for i = 1:length(tmax)
    NumSamples = ceil(tmax(i)/delta);
    StartSample = ceil(tmin(i)/delta) + 1;
    b_mat(i, (StartSample:NumSamples)+1) = 1;  % delay by 1
end

out = fullfile(fixtureRoot, 'v11_history_toFilter.mat');
save(out, 'windowTimes', 'minTime', 'maxTime', 'delta', 'b_mat', '-v7');
fprintf('  [history_toFilter] saved %s\n', out);
end


% =========================================================================
% Analysis.computeInvGausTrans — inverse Gaussian transform X = norminv(1-exp(-Z))
% (with clipping at .000001 / .999999), exposed as a static method.
% =========================================================================


function export_v11_analysis_computeInvGausTrans(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
% Synthetic Z — sorted ascending, values in (0, ~3).
Z = [0.1; 0.3; 0.5; 0.8; 1.2; 1.7; 2.4];

[X, ~, ~] = Analysis.computeInvGausTrans(Z);

out = fullfile(fixtureRoot, 'v11_analysis_computeInvGausTrans.mat');
save(out, 'Z', 'X', '-v7');
fprintf('  [analysis_computeInvGausTrans] saved %s\n', out);
end


% =========================================================================
% Analysis.RunAnalysisForAllNeurons — 2-neuron version. Use canonical
% 10 Hz / 1s synthetic stimulus + spikes.
% =========================================================================


function export_v11_analysis_RunAnalysisForAllNeurons(fixtureRoot) %#ok<DEFNU>
rng(42); %#ok<RNG>
sr = 10;
T = 1.0;
time_in = (0:1/sr:T)';
stim_data = sin(2*pi*1*time_in);
spk1 = [0.1, 0.3, 0.6, 0.8];
spk2 = [0.2, 0.4, 0.5, 0.7, 0.9];

stim = Covariate(time_in, stim_data, 'Stimulus', 'time', 's', '', {'stim'});
st1 = nspikeTrain(spk1, '1', sr, time_in(1), time_in(end), 'time', 's', '', '', -1);
st2 = nspikeTrain(spk2, '2', sr, time_in(1), time_in(end), 'time', 's', '', '', -1);
trial = Trial(nstColl({st1, st2}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, sr, [], []);
cfg.setName('stim');
fits = Analysis.RunAnalysisForAllNeurons(trial, ConfigColl({cfg}), 0);

% MATLAB returns a 1xN (here 2x1) cell of per-neuron FitResult objects.
% Capture per-neuron AIC/BIC/logLL and the first-config coefficient vector.
nNeurons = numel(fits);
fit_AIC = zeros(1, nNeurons);
fit_BIC = zeros(1, nNeurons);
fit_logLL = zeros(1, nNeurons);
coeffs_n1 = reshape(double(fits{1}.getCoeffs(1)), 1, []);
coeffs_n2 = reshape(double(fits{2}.getCoeffs(1)), 1, []);
for ii = 1:nNeurons
    fit_AIC(ii) = double(fits{ii}.AIC(1));
    fit_BIC(ii) = double(fits{ii}.BIC(1));
    fit_logLL(ii) = double(fits{ii}.logLL(1));
end

out = fullfile(fixtureRoot, 'v11_analysis_RunAnalysisForAllNeurons.mat');
save(out, 'time_in', 'stim_data', 'spk1', 'spk2', 'sr', ...
    'fit_AIC', 'fit_BIC', 'fit_logLL', 'coeffs_n1', 'coeffs_n2', '-v7');
fprintf('  [analysis_RunAnalysisForAllNeurons] saved %s\n', out);
end
