function Trial_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for the Trial class.
%
% MATLAB reference: Trial.m constructor/core utilities/toStructure/fromStructure

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'Trial');
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end
    outputFile = fullfile(outputDir, 'basic.mat');
else
    outputDir = fileparts(outputFile);
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end
end

time_cov = (0:0.1:1)';
cov_stim = sin(2*pi*time_cov);
cov_ctx = cos(2*pi*time_cov);

spike_times_u1 = [0.11 0.34 0.72]';
spike_times_u2 = [0.22 0.41 0.83]';
bin_size = 0.1;

tr = make_trial(time_cov, cov_stim, cov_ctx, spike_times_u1, spike_times_u2, bin_size);

initial_minTime = tr.minTime;
initial_maxTime = tr.maxTime;
initial_sampleRate = tr.sampleRate;
initial_cov_labels = tr.getAllCovLabels();
initial_neuron_names = tr.getNeuronNames();
initial_design_matrix = tr.getDesignMatrix(1);

edges = initial_minTime:bin_size:initial_maxTime;
expected_t_bins = (edges(1:end-1) + edges(2:end))/2;
expected_y_u1 = histcounts(spike_times_u1, edges)';
expected_y_u2 = histcounts(spike_times_u2, edges)';

idx = zeros(numel(expected_t_bins), 1);
for k = 1:numel(expected_t_bins)
    thisIdx = find(time_cov >= expected_t_bins(k), 1, 'first');
    if isempty(thisIdx)
        thisIdx = numel(time_cov);
    end
    idx(k) = thisIdx;
end
expected_X = initial_design_matrix(idx, :);

cov_mask_trial = make_trial(time_cov, cov_stim, cov_ctx, spike_times_u1, spike_times_u2, bin_size);
cov_mask_trial.setCovMask({'sine', 'sine'});
cov_mask_labels = cov_mask_trial.getCovLabelsFromMask();
cov_mask_trial.resetCovMask();
cov_mask_reset_labels = cov_mask_trial.getCovLabelsFromMask();

neuron_mask_trial = make_trial(time_cov, cov_stim, cov_ctx, spike_times_u1, spike_times_u2, bin_size);
neuron_mask_trial.setNeuronMask(1);
neuron_mask_indices = neuron_mask_trial.getNeuronIndFromMask();
neuron_mask_trial.resetNeuronMask();
neuron_mask_reset_indices = neuron_mask_trial.getNeuronIndFromMask();

struct_payload = tr.toStructure();
tr_roundtrip = Trial.fromStructure(struct_payload);
roundtrip_minTime = tr_roundtrip.minTime;
roundtrip_maxTime = tr_roundtrip.maxTime;
roundtrip_sampleRate = tr_roundtrip.sampleRate;
roundtrip_cov_labels = tr_roundtrip.getAllCovLabels();
roundtrip_neuron_names = tr_roundtrip.getNeuronNames();
roundtrip_design_matrix = tr_roundtrip.getDesignMatrix(1);

shift_trial = make_trial(time_cov, cov_stim, cov_ctx, spike_times_u1, spike_times_u2, bin_size);
shift_trial.shiftCovariates(0.2);
shift_minTime = shift_trial.minTime;
shift_maxTime = shift_trial.maxTime;
shift_cov_time_start = shift_trial.getCov(1).time(1);
shift_cov_time_end = shift_trial.getCov(1).time(end);
shift_trial.restoreToOriginal();
restore_minTime = shift_trial.minTime;
restore_maxTime = shift_trial.maxTime;
restore_cov_time_start = shift_trial.getCov(1).time(1);
restore_cov_time_end = shift_trial.getCov(1).time(end);

save(outputFile, ...
    'time_cov', ...
    'cov_stim', ...
    'cov_ctx', ...
    'spike_times_u1', ...
    'spike_times_u2', ...
    'bin_size', ...
    'initial_minTime', ...
    'initial_maxTime', ...
    'initial_sampleRate', ...
    'initial_cov_labels', ...
    'initial_neuron_names', ...
    'initial_design_matrix', ...
    'expected_t_bins', ...
    'expected_y_u1', ...
    'expected_y_u2', ...
    'expected_X', ...
    'cov_mask_labels', ...
    'cov_mask_reset_labels', ...
    'neuron_mask_indices', ...
    'neuron_mask_reset_indices', ...
    'struct_payload', ...
    'roundtrip_minTime', ...
    'roundtrip_maxTime', ...
    'roundtrip_sampleRate', ...
    'roundtrip_cov_labels', ...
    'roundtrip_neuron_names', ...
    'roundtrip_design_matrix', ...
    'shift_minTime', ...
    'shift_maxTime', ...
    'shift_cov_time_start', ...
    'shift_cov_time_end', ...
    'restore_minTime', ...
    'restore_maxTime', ...
    'restore_cov_time_start', ...
    'restore_cov_time_end');

fprintf('Wrote Trial fixtures to %s\n', outputFile);
end


function tr = make_trial(time_cov, cov_stim, cov_ctx, spike_times_u1, spike_times_u2, bin_size)
cov1 = Covariate(time_cov, cov_stim, 'sine', 'time', 's', '', {'sine'});
cov2 = Covariate(time_cov, cov_ctx, 'ctx', 'time', 's', '', {'ctx'});
cc = CovColl({cov1, cov2});

nst1 = nspikeTrain(spike_times_u1', 'u1', bin_size, 0.0, 1.0, 'time', 's', '', '', -1);
nst2 = nspikeTrain(spike_times_u2', 'u2', bin_size, 0.0, 1.0, 'time', 's', '', '', -1);
sc = nstColl({nst1, nst2});

tr = Trial(sc, cc);
end
