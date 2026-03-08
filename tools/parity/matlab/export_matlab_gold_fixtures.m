function export_matlab_gold_fixtures(repoRoot, matlabRepoRoot)
if nargin < 1 || isempty(repoRoot)
    error('repoRoot is required');
end
if nargin < 2 || isempty(matlabRepoRoot)
    matlabRepoRoot = fullfile(fileparts(repoRoot), 'nSTAT');
end

repoRoot = char(repoRoot);
matlabRepoRoot = char(matlabRepoRoot);

addpath(matlabRepoRoot);
addpath(fullfile(matlabRepoRoot, 'helpfiles'));
addpath(genpath(fullfile(matlabRepoRoot, 'libraries')));

fixtureRoot = fullfile(repoRoot, 'tests', 'parity', 'fixtures', 'matlab_gold');
if ~exist(fixtureRoot, 'dir')
    mkdir(fixtureRoot);
end

export_signalobj_fixture(fixtureRoot);
export_covariate_fixture(fixtureRoot);
export_nspiketrain_fixture(fixtureRoot);
export_nstcoll_fixture(fixtureRoot);
export_cif_fixture(fixtureRoot);
export_analysis_fixture(fixtureRoot);
export_point_process_fixture(fixtureRoot);
export_decoding_predict_fixture(fixtureRoot);
export_nonlinear_decode_fixture(fixtureRoot);
export_simulated_network_fixture(fixtureRoot);
end

function export_signalobj_fixture(fixtureRoot)
t = (0:0.1:0.4)';
data = [1 0; 2 1; 4 0; 8 -1; 16 0];
s = SignalObj(t, data, 'sig', 'time', 's', 'u', {'x1', 'x2'});
s1 = s.getSubSignal(1);
s2 = SignalObj((0.05:0.1:0.45)', [0; 1; 0; -1; 0], 'sig2', 'time', 's', 'u', {'x3'});

filtered = s.filter([0.25 0.5 0.25], 1);
resampled = s.resample(20);
derivative = s.derivative;
integral_sig = s.integral();
xc = xcorr(s.getSubSignal(1), s.getSubSignal(2), 2);
[s1c, s2c] = s1.makeCompatible(s2, 1);

payload = struct();
payload.time = s.time;
payload.data = s.data;
payload.filter_b = [0.25 0.5 0.25];
payload.filter_a = 1;
payload.filtered_data = filtered.data;
payload.derivative_data = derivative.data;
payload.integral_data = integral_sig.data;
payload.resample_rate = 20;
payload.resampled_time = resampled.time;
payload.resampled_data = resampled.data;
payload.xcorr_maxlag = 2;
payload.xcorr_time = xc.time;
payload.xcorr_data = xc.data;
payload.compat_time = s1c.time;
payload.compat_left_data = s1c.data;
payload.compat_right_data = s2c.data;

save(fullfile(fixtureRoot, 'signalobj_exactness.mat'), '-struct', 'payload');
end

function export_nspiketrain_fixture(fixtureRoot)
spikeTimes = [0.05; 0.10; 0.11; 0.30; 0.47];
binwidth = 0.05;
nst = nspikeTrain(spikeTimes, 'nst', binwidth, 0.0, 0.5, 'time', 's', 'spikes', 'spk', 0);
sig = nst.getSigRep(binwidth, 0.0, 0.5);
parts = nst.partitionNST([0.0 0.2 0.5]);
restoreTrain = nspikeTrain(spikeTimes, 'restore', 0.2, -0.1, 0.8, 'time', 's', 'spikes', 'spk', -1);
restoreTrain.setSigRep(0.1, -0.1, 0.8);
restoreTrain.setMinTime(-0.3);
restoreTrain.setMaxTime(1.1);
restoreTrain.restoreToOriginal();
burstTrain = nspikeTrain([0.0; 0.001; 0.002; 0.007; 0.507; 0.508; 0.509; 0.514], 'bursting', 0.001, 0.0, 0.6, 'time', 's', 'spikes', 'spk', 0);

payload = struct();
payload.spikeTimes = spikeTimes;
payload.binwidth = binwidth;
payload.minTime = 0.0;
payload.maxTime = 0.5;
payload.sig_time = sig.time;
payload.sig_data = sig.data;
payload.isis = nst.getISIs();
payload.avgFiringRate = nst.avgFiringRate;
payload.B = nst.B;
payload.An = nst.An;
payload.burstIndex = nst.burstIndex;
payload.numBursts = nst.numBursts;
payload.numSpikesPerBurst = nst.numSpikesPerBurst;
payload.part1_spikes = parts.getNST(1).spikeTimes;
payload.part2_spikes = parts.getNST(2).spikeTimes;
payload.restore_min_time = restoreTrain.minTime;
payload.restore_max_time = restoreTrain.maxTime;
payload.burst_avgSpikesPerBurst = burstTrain.avgSpikesPerBurst;
payload.burst_stdSpikesPerBurst = burstTrain.stdSpikesPerBurst;
payload.burst_numBursts = burstTrain.numBursts;
payload.burst_numSpikesPerBurst = burstTrain.numSpikesPerBurst;

save(fullfile(fixtureRoot, 'nspiketrain_exactness.mat'), '-struct', 'payload');
end

function export_covariate_fixture(fixtureRoot)
t = (0:0.1:0.4)';
replicates = [0.0 0.1 0.2 0.3; 0.2 0.3 0.4 0.5; 0.4 0.5 0.6 0.7; 0.6 0.7 0.8 0.9; 0.8 0.9 1.0 1.1];
cov = Covariate(t, replicates, 'Stimulus', 'time', 's', 'a.u.', {'r1','r2','r3','r4'});
meanCov = cov.computeMeanPlusCI(0.05);
ci = ConfidenceInterval(t, [mean(replicates,2)-0.1, mean(replicates,2)+0.1], 'CI', 'time', 's', 'a.u.');
covSingle = Covariate(t, mean(replicates,2), 'StimulusSingle', 'time', 's', 'a.u.', {'stim'});
covSingle.setConfInterval(ci);

payload = struct();
payload.time = t;
payload.replicates = replicates;
payload.mean_data = meanCov.data;
payload.mean_ci = meanCov.ci{1}.data;
payload.explicit_ci = covSingle.ci{1}.data;

save(fullfile(fixtureRoot, 'covariate_exactness.mat'), '-struct', 'payload');
end

function export_nstcoll_fixture(fixtureRoot)
n1 = nspikeTrain([0.1 0.3], '1', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
n2 = nspikeTrain([0.2], '2', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
coll = nstColl({n1, n2});
dataMat = coll.dataToMatrix([1 2], 0.1, 0.0, 0.5);

payload = struct();
payload.numSpikeTrains = coll.numSpikeTrains;
payload.firstName = coll.getNST(1).name;
payload.dataMatrix = dataMat;
payload.firstSpikeTimes = coll.getNST(1).spikeTimes;
payload.secondSpikeTimes = coll.getNST(2).spikeTimes;

save(fullfile(fixtureRoot, 'nstcoll_exactness.mat'), '-struct', 'payload');
end

function export_cif_fixture(fixtureRoot)
cif = CIF([0.1 0.5], {'stim1', 'stim2'}, {'stim1', 'stim2'}, 'binomial');
stimVal = [0.6; -0.2];
polyCif = build_polynomial_binomial_cif([-2.0 -0.5 0.3 -0.2 -0.1 0.05]);
polyStim = [0.2; -0.4];

payload = struct();
payload.beta = [0.1 0.5];
payload.stimVal = stimVal;
payload.lambda_delta = cif.evalLambdaDelta(stimVal);
payload.gradient = cif.evalGradient(stimVal);
payload.gradient_log = cif.evalGradientLog(stimVal);
payload.jacobian = cif.evalJacobian(stimVal);
payload.jacobian_log = cif.evalJacobianLog(stimVal);
payload.poly_beta = [-2.0 -0.5 0.3 -0.2 -0.1 0.05];
payload.poly_stimVal = polyStim;
payload.poly_lambda_delta = polyCif.evalLambdaDelta(polyStim);
payload.poly_gradient = polyCif.evalGradient(polyStim);
payload.poly_gradient_log = polyCif.evalGradientLog(polyStim);
payload.poly_jacobian = polyCif.evalJacobian(polyStim);
payload.poly_jacobian_log = polyCif.evalJacobianLog(polyStim);

save(fullfile(fixtureRoot, 'cif_exactness.mat'), '-struct', 'payload');
end

function export_analysis_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}));
summary = FitResSummary({fit});

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times = spikeTrain.spikeTimes;
payload.sample_rate = trial.sampleRate;
payload.coeffs = fit.getCoeffs(1);
payload.lambda_time = fit.lambda.time;
payload.lambda_data = fit.lambda.data(:,1);
payload.AIC = fit.AIC(1);
payload.BIC = fit.BIC(1);
payload.logLL = fit.logLL(1);
payload.distribution = fit.fitType{1};
payload.summaryAIC = summary.AIC(1);
payload.summaryBIC = summary.BIC(1);
payload.summarylogLL = summary.logLL(1);

save(fullfile(fixtureRoot, 'analysis_exactness.mat'), '-struct', 'payload');
end

function export_point_process_fixture(fixtureRoot)
rng(5);
Ts = .001;
t = (0:Ts:50)';
mu = -3;
H = tf([-1 -2 -4], [1], Ts, 'Variable', 'z^-1');
S = tf([1], 1, Ts, 'Variable', 'z^-1');
E = tf([0], 1, Ts, 'Variable', 'z^-1');
stim = Covariate(t, sin(2*pi*1*t), 'Stimulus', 'time', 's', 'Voltage', {'sin'});
ens = Covariate(t, zeros(length(t), 1), 'Ensemble', 'time', 's', 'Spikes', {'n1'});
[spikeColl, lambda] = CIF.simulateCIF(mu, H, S, E, stim, ens, 5, 'binomial');

spikeCounts = zeros(1, spikeColl.numSpikeTrains);
for i = 1:spikeColl.numSpikeTrains
    spikeCounts(i) = length(spikeColl.getNST(i).spikeTimes);
end

payload = struct();
payload.seed = 5;
payload.lambda_head = lambda.data(1:8, 1);
payload.spike_counts = spikeCounts;

save(fullfile(fixtureRoot, 'point_process_exactness.mat'), '-struct', 'payload');
end

function export_decoding_predict_fixture(fixtureRoot)
x_u = [0.1; -0.2];
W_u = [1.0 0.1; 0.1 2.0];
A = [1.0 0.2; 0.0 0.9];
Q = 0.05 * eye(2);
[x_p, W_p] = DecodingAlgorithms.PPDecode_predict(x_u, W_u, A, Q);

payload = struct();
payload.x_u = x_u;
payload.W_u = W_u;
payload.A = A;
payload.Q = Q;
payload.x_p = x_p;
payload.W_p = W_p;

save(fullfile(fixtureRoot, 'decoding_predict_exactness.mat'), '-struct', 'payload');
end

function export_nonlinear_decode_fixture(fixtureRoot)
A = eye(2);
Q = 0.01 * eye(2);
Px0 = 0.05 * eye(2);
delta = 0.1;
dN = [0 1 0 1;
      1 0 1 0];
lambdaCIF = {
    build_polynomial_binomial_cif([-2.0 -0.5 0.3 -0.2 -0.1 0.05]), ...
    build_polynomial_binomial_cif([-1.5 0.4 -0.2 0.15 -0.05 0.02])
};
[x_p, W_p, x_u, W_u] = DecodingAlgorithms.PPDecodeFilter(A, Q, Px0, dN, lambdaCIF, delta);

payload = struct();
payload.A = A;
payload.Q = Q;
payload.Px0 = Px0;
payload.delta = delta;
payload.dN = dN;
payload.beta1 = [-2.0 -0.5 0.3 -0.2 -0.1 0.05];
payload.beta2 = [-1.5 0.4 -0.2 0.15 -0.05 0.02];
payload.x_p = x_p;
payload.W_p = W_p;
payload.x_u = x_u;
payload.W_u = W_u;

save(fullfile(fixtureRoot, 'nonlinear_decode_exactness.mat'), '-struct', 'payload');
end

function export_simulated_network_fixture(fixtureRoot)
rng(4);
Ts = .001;
t = (0:Ts:50)';
mu{1} = -3; mu{2} = -3; %#ok<AGROW>
H{1} = tf([-4 -2 -1], [1], Ts, 'Variable', 'z^-1'); %#ok<AGROW>
H{2} = tf([-4 -2 -1], [1], Ts, 'Variable', 'z^-1'); %#ok<AGROW>
S{1} = tf([1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
S{2} = tf([-1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
E{1} = tf([1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
E{2} = tf([-4], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
stim = Covariate(t, sin(2*pi*1*t), 'Stimulus', 'time', 's', 'Voltage', {'sin'});
assignin('base', 'S1', S{1}); assignin('base', 'H1', H{1}); assignin('base', 'E1', E{1}); assignin('base', 'mu1', mu{1});
assignin('base', 'S2', S{2}); assignin('base', 'H2', H{2}); assignin('base', 'E2', E{2}); assignin('base', 'mu2', mu{2});
options = simget;
[~,~,yout] = sim('SimulatedNetwork2', [stim.minTime stim.maxTime], options, stim.dataToStructure);
[h1Num, ~] = tfdata(H{1}, 'v');
[h2Num, ~] = tfdata(H{2}, 'v');
[s1Num, ~] = tfdata(S{1}, 'v');
[s2Num, ~] = tfdata(S{2}, 'v');
[e1Num, ~] = tfdata(E{1}, 'v');
[e2Num, ~] = tfdata(E{2}, 'v');
stateMat = yout(:,1:2);
probMat = zeros(size(stateMat));
for n = 1:size(stateMat, 1)
    hist1 = 0; hist2 = 0;
    for lag = 1:length(h1Num)
        if n-lag >= 1
            hist1 = hist1 + h1Num(lag) * stateMat(n-lag,1);
            hist2 = hist2 + h2Num(lag) * stateMat(n-lag,2);
        end
    end
    ens1 = 0; ens2 = 0;
    if n > 1
        ens1 = e1Num(1) * stateMat(n-1,2);
        ens2 = e2Num(1) * stateMat(n-1,1);
    end
    eta1 = mu{1} + hist1 + s1Num(1) * stim.data(n) + ens1;
    eta2 = mu{2} + hist2 + s2Num(1) * stim.data(n) + ens2;
    probMat(n,1) = exp(eta1) / (1 + exp(eta1));
    probMat(n,2) = exp(eta2) / (1 + exp(eta2));
end

payload = struct();
payload.actual_network = [0 1; -4 0];
payload.prob_head = probMat(1:5,:);
payload.state_head = yout(1:5,1:2);
payload.spike_counts = [sum(yout(:,1) > .5), sum(yout(:,2) > .5)];

save(fullfile(fixtureRoot, 'simulated_network_exactness.mat'), '-struct', 'payload');
end

function cifObj = build_polynomial_binomial_cif(beta)
beta = beta(:)';
syms x y real
cifObj = CIF(beta(1:3), {'1', 'x', 'y'}, {'x', 'y'}, 'binomial');
cifObj.b = beta;
cifObj.varIn = [sym(1); x; y; x^2; y^2; x * y];
cifObj.stimVars = [x; y];
cifObj.fitType = 'binomial';
cifObj.history = [];
cifObj.histCoeffs = [];
cifObj.histVars = {};
cifObj.histCoeffVars = {};
cifObj.spikeTrain = [];
cifObj.historyMat = [];
cifObj.lambdaDelta = simplify(exp(beta * cifObj.varIn) ./ (1 + exp(beta * cifObj.varIn)));
cifObj.lambdaDeltaFunction = matlabFunction(cifObj.lambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.gradientLambdaDelta = simplify(jacobian(cifObj.lambdaDelta, cifObj.stimVars));
cifObj.gradientLogLambdaDelta = simplify(jacobian(log(cifObj.lambdaDelta), cifObj.stimVars));
cifObj.gradientFunction = matlabFunction(cifObj.gradientLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.gradientLogFunction = matlabFunction(cifObj.gradientLogLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.jacobianLambdaDelta = simplify(jacobian(cifObj.gradientLambdaDelta, cifObj.stimVars));
cifObj.jacobianFunction = matlabFunction(cifObj.jacobianLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.jacobianLogLambdaDelta = simplify(jacobian(cifObj.gradientLogLambdaDelta, cifObj.stimVars));
cifObj.jacobianLogFunction = matlabFunction(cifObj.jacobianLogLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.lambdaDeltaGamma = [];
cifObj.LogLambdaDeltaGamma = [];
cifObj.gradientLambdaDeltaGamma = [];
cifObj.gradientLogLambdaDeltaGamma = [];
cifObj.jacobianLambdaDeltaGamma = [];
cifObj.jacobianLogLambdaDeltaGamma = [];
cifObj.lambdaDeltaGammaFunction = [];
cifObj.LogLambdaDeltaGammaFunction = [];
cifObj.gradientFunctionGamma = [];
cifObj.gradientLogFunctionGamma = [];
cifObj.jacobianFunctionGamma = [];
cifObj.jacobianLogFunctionGamma = [];
cifObj.indepVars = symvar(cifObj.lambdaDelta);

vars = symvar(cifObj.varIn);
if length(vars) == 1
    cifObj.argstr = 'val';
else
    argstr = 'val(1)';
    for i = 2:length(vars)
        argstr = strcat(argstr, [',val(' num2str(i) ')']);
    end
    cifObj.argstr = argstr;
end
cifObj.argstrLDGamma = '';
end
