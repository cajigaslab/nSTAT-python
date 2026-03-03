function SignalObj_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for SignalObj parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'SignalObj');
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

time = [0; 0.25; 0.50; 0.75; 1.00];
data = [1 2; 2 3; 4 5; 3 4; 2 3];
labels = {'ch1','ch2'};

sig = SignalObj(time, data, 'sig', 'time', 's', 'unit', labels);

base_data = sig.dataToMatrix();
base_time = sig.time;
base_sample_rate = sig.sampleRate;

deriv = sig.derivative();
deriv_data = deriv.dataToMatrix();

sub = sig.getSubSignal([2]);
sub_data = sub.dataToMatrix();
sub_time = sub.time;

other = SignalObj(time, [10; 20; 30; 40; 50], 'sig2', 'time', 's', 'unit', {'ch3'});
merged = sig.merge(other);
merged_data = merged.dataToMatrix();

resampled = sig.resample(8);
resampled_time = resampled.time;
resampled_data = resampled.dataToMatrix();
resampled_sample_rate = resampled.sampleRate;

shifted = sig.shift(0.1);
shifted_time = shifted.time;

aligned = sig.copySignal();
aligned.alignTime(0.5, 0.0);
aligned_time = aligned.time;

nearest_idx = sig.findNearestTimeIndex(0.63);
nearest_indices = sig.findNearestTimeIndices([0.00 0.38 0.99]);
value_at_05 = sig.getValueAt(0.5);

sig_struct = sig.dataToStructure();
roundtrip = SignalObj.signalFromStruct(sig_struct);
roundtrip_data = roundtrip.dataToMatrix();
roundtrip_time = roundtrip.time;

save(outputFile, ...
    'time', ...
    'data', ...
    'base_data', ...
    'base_time', ...
    'base_sample_rate', ...
    'deriv_data', ...
    'sub_data', ...
    'sub_time', ...
    'merged_data', ...
    'resampled_time', ...
    'resampled_data', ...
    'resampled_sample_rate', ...
    'shifted_time', ...
    'aligned_time', ...
    'nearest_idx', ...
    'nearest_indices', ...
    'value_at_05', ...
    'sig_struct', ...
    'roundtrip_data', ...
    'roundtrip_time');

fprintf('Wrote SignalObj fixtures to %s\n', outputFile);
end
