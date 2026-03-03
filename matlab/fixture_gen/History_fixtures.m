function History_fixtures(outputFile)
% Generate deterministic fixtures for History parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'History');
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

window_times = [0, 0.05, 0.10, 0.20];
min_time = 0.0;
max_time = 0.40;

hist_obj = History(window_times, min_time, max_time);
hist_struct = hist_obj.toStructure();

hist_roundtrip = History.fromStructure(hist_struct);
roundtrip_struct = hist_roundtrip.toStructure();

hist_set = History(window_times);
hist_set.setWindow([0, 0.10, 0.30]);
set_window_times = hist_set.windowTimes;

spike_times = [0.12, 0.28];
time_grid = [0.15, 0.25, 0.30, 0.40];

n_bins = length(window_times) - 1;
expected_design = zeros(length(time_grid), n_bins);
for i = 1:length(time_grid)
    lags = time_grid(i) - spike_times;
    for j = 1:n_bins
        lo = window_times(j);
        hi = window_times(j + 1);
        expected_design(i, j) = sum((lags > lo) & (lags <= hi));
    end
end

expected_filter = diff(window_times) ./ sum(diff(window_times));

delta = 0.05;
time_vec = min(window_times(1:end-1)):delta:max(window_times(2:end));
expected_filter_delta = zeros(length(window_times)-1, length(time_vec));
for i = 1:(length(window_times)-1)
    lo = window_times(i);
    hi = window_times(i+1);
    num_samples = ceil(hi/delta);
    start_sample = ceil(lo/delta) + 1;
    idx = (start_sample:num_samples) + 1;
    idx = idx(idx >= 1 & idx <= length(time_vec));
    expected_filter_delta(i, idx) = 1;
end

save(outputFile, ...
    'window_times', ...
    'min_time', ...
    'max_time', ...
    'hist_struct', ...
    'roundtrip_struct', ...
    'set_window_times', ...
    'spike_times', ...
    'time_grid', ...
    'expected_design', ...
    'expected_filter', ...
    'delta', ...
    'expected_filter_delta');

fprintf('Wrote History fixtures to %s\n', outputFile);
end
