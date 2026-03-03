function Events_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for the Events class.
%
% MATLAB reference: Events.m (constructor/plot/toStructure/fromStructure)

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'Events');
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

event_times = [0.1 0.4 0.9];
event_labels = {'E1', 'E2', 'E3'};

events_default = Events(event_times, event_labels);
events_custom = Events(event_times, event_labels, 'g');

default_struct = events_default.toStructure();
roundtrip = Events.fromStructure(default_struct);
roundtrip_struct = roundtrip.toStructure();
roundtrip_event_times = roundtrip.eventTimes;
roundtrip_event_labels = roundtrip.eventLabels;
roundtrip_event_color = roundtrip.eventColor;

plot_axis = [0 1 0 2];
fig = figure('Visible', 'off');
ax = axes('Parent', fig);
axis(ax, plot_axis);
hold(ax, 'on');

h_lines = events_default.plot(ax);
plot_line_count = numel(h_lines);
plot_x_data = cell(1, plot_line_count);
plot_y_data = cell(1, plot_line_count);
for i = 1:plot_line_count
    plot_x_data{i} = get(h_lines(i), 'XData');
    plot_y_data{i} = get(h_lines(i), 'YData');
end

text_handles = findall(ax, 'Type', 'text');
text_count = numel(text_handles);
text_strings = cell(1, text_count);
text_positions = cell(1, text_count);
text_x = zeros(1, text_count);
for i = 1:text_count
    text_strings{i} = get(text_handles(i), 'String');
    pos = get(text_handles(i), 'Position');
    text_positions{i} = pos;
    text_x(i) = pos(1);
end
[~, order] = sort(text_x);
text_strings = text_strings(order);
text_positions = text_positions(order);

close(fig);

save(outputFile, ...
    'event_times', ...
    'event_labels', ...
    'plot_axis', ...
    'plot_line_count', ...
    'plot_x_data', ...
    'plot_y_data', ...
    'text_strings', ...
    'text_positions', ...
    'default_struct', ...
    'roundtrip_struct', ...
    'roundtrip_event_times', ...
    'roundtrip_event_labels', ...
    'roundtrip_event_color');

fprintf('Wrote Events fixtures to %s\n', outputFile);
end
