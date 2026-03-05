function export_helpfile_figures(manifestJsonPath, outputRoot, reportJsonPath)
% Export helpfile figures in strict fig_### order for parity checks.
%
% Inputs:
%   manifestJsonPath - JSON with `topics` entries containing:
%       topic, source_path, source_type, expected_figure_count
%   outputRoot - destination root for <topic>/fig_###.png
%   reportJsonPath - output JSON report path

if nargin < 3
    error('export_helpfile_figures:InvalidArgs', ...
        'Usage: export_helpfile_figures(manifestJsonPath, outputRoot, reportJsonPath)');
end

manifestJsonPath = char(manifestJsonPath);
outputRoot = char(outputRoot);
reportJsonPath = char(reportJsonPath);

if ~exist(manifestJsonPath, 'file')
    error('export_helpfile_figures:MissingManifest', ...
        'Manifest JSON not found: %s', manifestJsonPath);
end

manifestPayload = jsondecode(fileread(manifestJsonPath));
topics = manifestPayload.topics;
if ~isstruct(topics)
    error('export_helpfile_figures:InvalidManifest', ...
        'Expected manifest.topics to be a struct array.');
end

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

% Add source roots to path so script-local dependencies resolve.
sourceRoots = {};
for i = 1:numel(topics)
    sourcePath = char(topics(i).source_path);
    helpDir = fileparts(sourcePath);
    toolboxRoot = fileparts(helpDir);
    if ~any(strcmp(sourceRoots, toolboxRoot))
        sourceRoots{end+1} = toolboxRoot; %#ok<AGROW>
    end
end
for i = 1:numel(sourceRoots)
    addpath(genpath(sourceRoots{i}));
end

reportRows = repmat(struct( ...
    'topic', '', ...
    'source_path', '', ...
    'source_type', '', ...
    'expected_figures', 0, ...
    'produced_figures', 0, ...
    'status', '', ...
    'error', ''), numel(topics), 1);

for i = 1:numel(topics)
    topic = char(topics(i).topic);
    sourcePath = char(topics(i).source_path);
    sourceType = char(topics(i).source_type);
    expectedFigures = double(topics(i).expected_figure_count);
    outDir = fullfile(outputRoot, topic);

    if exist(outDir, 'dir')
        rmdir(outDir, 's');
    end
    mkdir(outDir);

    close all force;
    drawnow;

    reportRows(i).topic = topic;
    reportRows(i).source_path = sourcePath;
    reportRows(i).source_type = sourceType;
    reportRows(i).expected_figures = expectedFigures;

    try
        if strcmpi(sourceType, 'mlx')
            convertedPath = fullfile(tempdir(), [topic '__converted__.m']);
            if exist(convertedPath, 'file')
                delete(convertedPath);
            end
            matlab.internal.liveeditor.openAndConvert(sourcePath, convertedPath);
            run_script_in_base(convertedPath);
        else
            run_script_in_base(sourcePath);
        end

        figs = findall(groot, 'Type', 'figure');
        if isempty(figs)
            produced = 0;
        else
            figNums = arrayfun(@(h) h.Number, figs);
            [~, order] = sort(figNums);
            figs = figs(order);
            produced = numel(figs);
            for j = 1:produced
                outFile = fullfile(outDir, sprintf('fig_%03d.png', j));
                try
                    exportgraphics(figs(j), outFile, 'Resolution', 180);
                catch
                    % Fallback for older MATLAB releases.
                    saveas(figs(j), outFile);
                end
            end
        end

        reportRows(i).produced_figures = produced;
        if produced == expectedFigures
            reportRows(i).status = 'ok';
        else
            reportRows(i).status = 'count_mismatch';
            reportRows(i).error = sprintf('produced=%d expected=%d', produced, expectedFigures);
        end
    catch ME
        reportRows(i).status = 'error';
        reportRows(i).produced_figures = 0;
        reportRows(i).error = ME.message;
    end

    close all force;
    evalin('base', 'clearvars');
    drawnow;
end

report = struct();
report.generated_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.output_root = outputRoot;
report.topic_count = numel(topics);
report.results = reportRows;

reportDir = fileparts(reportJsonPath);
if ~isempty(reportDir) && ~exist(reportDir, 'dir')
    mkdir(reportDir);
end
fid = fopen(reportJsonPath, 'w');
if fid < 0
    error('export_helpfile_figures:ReportWriteFailed', ...
        'Could not open report output path: %s', reportJsonPath);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(report, 'PrettyPrint', true));

end

function run_script_in_base(scriptPath)
scriptEscaped = strrep(scriptPath, '''', '''''');
evalin('base', sprintf('run(''%s'');', scriptEscaped));
end
