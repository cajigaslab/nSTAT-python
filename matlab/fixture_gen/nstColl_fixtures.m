function nstColl_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for nstColl parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'nstColl');
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

n1 = nspikeTrain([0.10 0.20 0.25 0.90], 'u1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
n2 = nspikeTrain([0.15 0.40 0.80],      'u2', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
coll = nstColl({n1, n2});

first_spike = coll.getFirstSpikeTime;
last_spike = coll.getLastSpikeTime;

names = coll.getNSTnames;
indices_u2 = coll.getNSTIndicesFromName('u2');
name_ind2 = coll.getNSTnameFromInd(2);

data_mat = coll.dataToMatrix([1 2], 0.1, 0.0, 1.0);
is_binary = coll.isSigRepBinary;
binary_sig = coll.BinarySigRep;

min_isis = coll.getMinISIs;
isis_cell = coll.getISIs;
max_bin_size = coll.getMaxBinSizeBinary;
max_sample_rate = coll.findMaxSampleRate;

psth_signal = coll.psth(0.1);
psth_time = psth_signal.time;
psth_data = psth_signal.dataToMatrix;

merged = coll.toSpikeTrain;
merged_spike_times = merged.spikeTimes;

basis = nstColl.generateUnitImpulseBasis(0.2, 0.0, 1.0, 10.0);
basis_time = basis.time;
basis_data = basis.dataToMatrix;

coll_mask = nstColl({n1.nstCopy, n2.nstCopy});
coll_mask.setNeuronMaskFromInd([1]);
mask_indices = coll_mask.getIndFromMask;
mask_minus = coll_mask.getIndFromMaskMinusOne(1);
mask_is_set = coll_mask.isNeuronMaskSet;

coll_neigh = nstColl({n1.nstCopy, n2.nstCopy});
coll_neigh.setNeighbors([2;1]);
[neighbors_1, num_neighbors_1] = coll_neigh.getNeighbors(1);
are_neighbors_set = coll_neigh.areNeighborsSet;

var_est = coll.estimateVarianceAcrossTrials([], [], 3, 'poisson');

coll_struct = coll.toStructure;
roundtrip = nstColl.fromStructure(coll_struct);
roundtrip_data = roundtrip.dataToMatrix([1 2], 0.1, 0.0, 1.0);

save(outputFile, ...
    'first_spike', ...
    'last_spike', ...
    'names', ...
    'indices_u2', ...
    'name_ind2', ...
    'data_mat', ...
    'is_binary', ...
    'binary_sig', ...
    'min_isis', ...
    'isis_cell', ...
    'max_bin_size', ...
    'max_sample_rate', ...
    'psth_time', ...
    'psth_data', ...
    'merged_spike_times', ...
    'basis_time', ...
    'basis_data', ...
    'mask_indices', ...
    'mask_minus', ...
    'mask_is_set', ...
    'neighbors_1', ...
    'num_neighbors_1', ...
    'are_neighbors_set', ...
    'var_est', ...
    'coll_struct', ...
    'roundtrip_data');

fprintf('Wrote nstColl fixtures to %s\n', outputFile);
end
