function export_class_equivalence_fixtures(nstatRoot, outRoot)
%EXPORT_CLASS_EQUIVALENCE_FIXTURES Create MATLAB-gold fixtures for class contracts.
%   export_class_equivalence_fixtures(nstatRoot, outRoot)
%
% Inputs
%   nstatRoot - path to MATLAB nSTAT source-of-truth repo
%   outRoot   - output root where class fixture .mat files are written
%
% Notes
%   - This script is deterministic (rng fixed).
%   - It writes compact fixtures used by Python class-contract tests.

if nargin < 1 || isempty(nstatRoot)
    envRoot = getenv('NSTAT_MATLAB_ROOT');
    if isempty(envRoot)
        error(['nstatRoot not provided. Pass source root explicitly or set ', ...
               'NSTAT_MATLAB_ROOT environment variable.']);
    end
    nstatRoot = envRoot;
end

if nargin < 2 || isempty(outRoot)
    here = fileparts(mfilename('fullpath'));
    outRoot = fullfile(here, '..', '..', 'tests', 'parity', 'fixtures', 'matlab_gold', 'classes');
end

if exist(nstatRoot, 'dir') ~= 7
    error('nstatRoot does not exist: %s', nstatRoot);
end

addpath(nstatRoot);
rng(0, 'twister');

if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

write_confidence_interval_fixture(outRoot);
write_trialconfig_fixture(outRoot);
write_configcoll_fixture(outRoot);
write_cif_fixture(outRoot);
write_fitresult_and_summary_fixtures(outRoot);

fprintf('Wrote class fixtures to %s\n', outRoot);
end

function write_confidence_interval_fixture(outRoot)
classDir = fullfile(outRoot, 'ConfidenceInterval');
if exist(classDir, 'dir') ~= 7
    mkdir(classDir);
end

time = (0:0.25:1.0)';
ci_data = [time, time + 0.5];
ci = ConfidenceInterval(time, ci_data, 'CI', 'time', 's', 'a.u.', {'low', 'high'});
ci.setColor('r');
ci.setValue(0.90);

width = ci_data(:, 2) - ci_data(:, 1);
ci_color = ci.color;
ci_value = ci.value;

save(fullfile(classDir, 'basic.mat'), ...
    'time', 'ci_data', 'width', 'ci_color', 'ci_value', '-v7');
end

function write_trialconfig_fixture(outRoot)
classDir = fullfile(outRoot, 'TrialConfig');
if exist(classDir, 'dir') ~= 7
    mkdir(classDir);
end

tc = TrialConfig({'stim'}, 1000, [], [], [], 0, 'cfg');
tc_struct = tc.toStructure;
tc_roundtrip = TrialConfig.fromStructure(tc_struct);

tc_name = tc.getName;
tc_sample_rate = tc.sampleRate;
tc_cov_mask = tc.covMask;
tc_cov_lag = tc.covLag;
tc_roundtrip_name = tc_roundtrip.getName;
tc_roundtrip_sample_rate = tc_roundtrip.sampleRate;

save(fullfile(classDir, 'basic.mat'), ...
    'tc_name', 'tc_sample_rate', 'tc_cov_mask', 'tc_cov_lag', ...
    'tc_roundtrip_name', 'tc_roundtrip_sample_rate', '-v7');
end

function write_configcoll_fixture(outRoot)
classDir = fullfile(outRoot, 'ConfigColl');
if exist(classDir, 'dir') ~= 7
    mkdir(classDir);
end

tc1 = TrialConfig({'stim'}, 1000, [], [], [], 0, 'cfg1');
tc2 = TrialConfig({'stim2'}, 500, [], [], [], 0, 'cfg2');
cc = ConfigColl({tc1, tc2});
names = cc.getConfigNames;
subset = cc.getSubsetConfigs([1]);

num_configs = cc.numConfigs;
subset_num_configs = subset.numConfigs;

save(fullfile(classDir, 'basic.mat'), ...
    'num_configs', 'names', 'subset_num_configs', '-v7');
end

function write_cif_fixture(outRoot)
classDir = fullfile(outRoot, 'CIF');
if exist(classDir, 'dir') ~= 7
    mkdir(classDir);
end

dt = 0.001;
time = (0:dt:2.0)';
lambda_values = max(12 + 4 * sin(2 * pi * 1 * time), 0.2);
lambda_cov = Covariate(time, lambda_values, 'Lambda', 'time', 's', 'sp/s', {'lambda'});

rng(0, 'twister');
coll = CIF.simulateCIFByThinningFromLambda(lambda_cov, 3, dt);
spike_counts = zeros(3, 1);
first_five_spikes = cell(3, 1);
for i = 1:3
    st = coll.getNST(i);
    spike_counts(i) = length(st.spikeTimes);
    first_five_spikes{i} = st.spikeTimes(1:min(5, end));
end

save(fullfile(classDir, 'basic.mat'), ...
    'time', 'lambda_values', 'dt', 'spike_counts', 'first_five_spikes', '-v7');
end

function write_fitresult_and_summary_fixtures(outRoot)
fitDir = fullfile(outRoot, 'FitResult');
if exist(fitDir, 'dir') ~= 7
    mkdir(fitDir);
end
sumDir = fullfile(outRoot, 'FitResSummary');
if exist(sumDir, 'dir') ~= 7
    mkdir(sumDir);
end

time = (0:0.1:1.0)';
lambda_cov = Covariate(time, ones(size(time)), 'Lambda', 'time', 's', 'sp/s', {'lambda'});
spike_obj = nspikeTrain([0.2 0.4 0.8]', '1', 0.001, 0, 1);

tc = TrialConfig();
tc.setName('cfg1');
cc = ConfigColl(tc);

cov_labels = {{'Baseline'}};
num_hist = {0};
hist_objects = {[]};
ens_hist_obj = {[]};
b = {[0.1]};
dev = 1;
stats = {struct('se', 0.1, 'p', 0.5)};
AIC = 2;
BIC = 3;
logLL = -1;

fit_obj = FitResult( ...
    spike_obj, cov_labels, num_hist, hist_objects, ens_hist_obj, lambda_cov, ...
    b, dev, stats, AIC, BIC, logLL, cc, {}, {}, 'poisson');

Z = [0.1; 0.2; 0.3];
U = [0.05; 0.1; 0.2];
xAxis = [0.1; 0.2; 0.3];
ksSorted = [0.11; 0.21; 0.29];
ks_stat = 0.05;
fit_obj.setKSStats(Z, U, xAxis, ksSorted, ks_stat);

fit_num_results = fit_obj.numResults;
fit_aic = fit_obj.AIC;
fit_bic = fit_obj.BIC;
fit_logll = fit_obj.logLL;
fit_neuron_number = fit_obj.neuronNumber;
fit_ks_stat = fit_obj.KSStats.ks_stat;
fit_p_value = fit_obj.KSStats.pValue;

save(fullfile(fitDir, 'basic.mat'), ...
    'fit_num_results', 'fit_aic', 'fit_bic', 'fit_logll', ...
    'fit_neuron_number', 'fit_ks_stat', 'fit_p_value', '-v7');

frs = FitResSummary(fit_obj);
summary_num_neurons = frs.numNeurons;
summary_num_results = frs.numResults;
summary_aic = frs.AIC;
summary_bic = frs.BIC;
summary_fit_names = frs.fitNames;

save(fullfile(sumDir, 'basic.mat'), ...
    'summary_num_neurons', 'summary_num_results', ...
    'summary_aic', 'summary_bic', 'summary_fit_names', '-v7');
end
