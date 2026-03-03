function nspikeTrain_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for nspikeTrain parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'nspikeTrain');
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

spikeTimes = [0.10 0.20 0.25 0.90];
name = 'u1';
binwidth = 0.10;
minTime = 0.0;
maxTime = 1.0;

nst = nspikeTrain(spikeTimes, name, binwidth, minTime, maxTime, 'time', 's', '', '', -1);

sigRep = nst.getSigRep(binwidth, minTime, maxTime);
sig_time = sigRep.time;
sig_count = sigRep.dataToMatrix;
is_binary = nst.isSigRepBinary;

isis = nst.getISIs;
min_isi = nst.getMinISI;
max_bin_size = nst.getMaxBinSizeBinary;

firing_rate = length(spikeTimes) / (maxTime - minTime);
l_stat = nst.getLStatistic;

ncopy = nst.nstCopy;
copy_spike_times = ncopy.spikeTimes;

nset = nst.nstCopy;
nset.setMinTime(0.05);
nset.setMaxTime(0.95);
set_min_time = nset.minTime;
set_max_time = nset.maxTime;
set_spike_times = nset.spikeTimes;

nres = nst.nstCopy;
nres.resample(10.0);
resample_rate = nres.sampleRate;
resample_sig = nres.getSigRep(0.1, minTime, maxTime).dataToMatrix;

nclear = nst.nstCopy;
nclear.setSigRep(0.1, minTime, maxTime);
clear_before = isempty(nclear.sigRep);
nclear.clearSigRep;
clear_after = isempty(nclear.sigRep);

parts = nst.partitionNST([0.0 0.5 1.0], 0);
parts_num = parts.numSpikeTrains;
part1_spikes = parts.getNST(1).spikeTimes;
part2_spikes = parts.getNST(2).spikeTimes;

nst_struct = nst.toStructure;
roundtrip = nspikeTrain.fromStructure(nst_struct);
roundtrip_spike_times = roundtrip.spikeTimes;
roundtrip_sig = roundtrip.getSigRep(binwidth, minTime, maxTime).dataToMatrix;

save(outputFile, ...
    'spikeTimes', ...
    'sig_time', ...
    'sig_count', ...
    'is_binary', ...
    'isis', ...
    'min_isi', ...
    'max_bin_size', ...
    'firing_rate', ...
    'l_stat', ...
    'copy_spike_times', ...
    'set_min_time', ...
    'set_max_time', ...
    'set_spike_times', ...
    'resample_rate', ...
    'resample_sig', ...
    'clear_before', ...
    'clear_after', ...
    'parts_num', ...
    'part1_spikes', ...
    'part2_spikes', ...
    'nst_struct', ...
    'roundtrip_spike_times', ...
    'roundtrip_sig');

fprintf('Wrote nspikeTrain fixtures to %s\n', outputFile);
end
