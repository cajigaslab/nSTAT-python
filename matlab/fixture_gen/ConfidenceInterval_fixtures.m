function ConfidenceInterval_fixtures(outputFile)
% Generate deterministic fixtures for ConfidenceInterval parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'ConfidenceInterval');
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

time = linspace(0,1,6)';
lower = [0.10; 0.20; 0.30; 0.25; 0.20; 0.10];
upper = lower + 0.40;

ci = ConfidenceInterval(time, [lower upper]);
default_color = ci.color;
default_value = ci.value;

ci.setColor('g');
ci.setValue(0.90);
set_color = ci.color;
set_value = ci.value;

probe_values = (lower + upper) / 2;
contains_probe = (probe_values >= lower) & (probe_values <= upper);
width = upper - lower;

data_struct = ci.dataToStructure();
roundtrip = ConfidenceInterval.fromStructure(data_struct);
roundtrip_color = roundtrip.color;
roundtrip_value = roundtrip.value;
roundtrip_data = roundtrip.dataToMatrix();

fig = figure('Visible','off');
ax = axes('Parent',fig);
hold(ax,'on');
ci.plot('g',0.3,0);
line_handles = findall(ax, 'Type', 'line');
line_count = numel(line_handles);
line_x_data = cell(1, line_count);
line_y_data = cell(1, line_count);
line_means = zeros(1, line_count);
for i=1:line_count
    line_x_data{i} = get(line_handles(i), 'XData');
    line_y_data{i} = get(line_handles(i), 'YData');
    line_means(i) = mean(line_y_data{i});
end
[~, order] = sort(line_means, 'ascend');
line_x_data = line_x_data(order);
line_y_data = line_y_data(order);
close(fig);

fig2 = figure('Visible','off');
ax2 = axes('Parent',fig2);
hold(ax2,'on');
ci.plot('g',0.2,1);
patch_handles = findall(ax2, 'Type', 'patch');
patch_count = numel(patch_handles);
patch_x_data = cell(1, patch_count);
patch_y_data = cell(1, patch_count);
for i=1:patch_count
    patch_x_data{i} = get(patch_handles(i), 'XData');
    patch_y_data{i} = get(patch_handles(i), 'YData');
end
close(fig2);

save(outputFile, ...
    'time', ...
    'lower', ...
    'upper', ...
    'default_color', ...
    'default_value', ...
    'set_color', ...
    'set_value', ...
    'probe_values', ...
    'contains_probe', ...
    'width', ...
    'data_struct', ...
    'roundtrip_color', ...
    'roundtrip_value', ...
    'roundtrip_data', ...
    'line_count', ...
    'line_x_data', ...
    'line_y_data', ...
    'patch_count', ...
    'patch_x_data', ...
    'patch_y_data');

fprintf('Wrote ConfidenceInterval fixtures to %s\n', outputFile);
end
