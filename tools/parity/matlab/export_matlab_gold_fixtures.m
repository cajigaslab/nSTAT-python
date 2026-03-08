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
export_nspiketrain_fixture(fixtureRoot);
export_cif_fixture(fixtureRoot);
export_point_process_fixture(fixtureRoot);
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

function export_cif_fixture(fixtureRoot)
cif = CIF([0.1 0.5], {'stim1', 'stim2'}, {'stim1', 'stim2'}, 'binomial');
stimVal = [0.6; -0.2];

payload = struct();
payload.beta = [0.1 0.5];
payload.stimVal = stimVal;
payload.lambda_delta = cif.evalLambdaDelta(stimVal);
payload.gradient = cif.evalGradient(stimVal);
payload.gradient_log = cif.evalGradientLog(stimVal);
payload.jacobian = cif.evalJacobian(stimVal);
payload.jacobian_log = cif.evalJacobianLog(stimVal);

save(fullfile(fixtureRoot, 'cif_exactness.mat'), '-struct', 'payload');
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
