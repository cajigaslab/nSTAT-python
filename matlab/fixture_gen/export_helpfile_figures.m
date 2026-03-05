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
        scriptToRun = sourcePath;
        if strcmpi(sourceType, 'm') && is_classdef_source_file(sourcePath)
            produced = 0;
            reportRows(i).produced_figures = produced;
            if produced == expectedFigures
                reportRows(i).status = 'ok';
            else
                reportRows(i).status = 'count_mismatch';
                reportRows(i).error = sprintf('produced=%d expected=%d', produced, expectedFigures);
            end
            close all force;
            evalin('base', 'clearvars');
            if isappdata(groot, 'NSTAT_EXPORT_OUTDIR')
                rmappdata(groot, 'NSTAT_EXPORT_OUTDIR');
            end
            if isappdata(groot, 'NSTAT_EXPORT_COUNT')
                rmappdata(groot, 'NSTAT_EXPORT_COUNT');
            end
            drawnow;
            continue;
        end
        if strcmpi(sourceType, 'mlx')
            convertedPath = fullfile(tempdir(), [topic '__converted__.m']);
            if exist(convertedPath, 'file')
                delete(convertedPath);
            end
            matlab.internal.liveeditor.openAndConvert(sourcePath, convertedPath);
            scriptToRun = convertedPath;
        end

        instrumentedPath = fullfile(tempdir(), [topic '__instrumented__.m']);
        instrument_script_for_figure_capture(scriptToRun, instrumentedPath);

        setappdata(groot, 'NSTAT_EXPORT_OUTDIR', outDir);
        setappdata(groot, 'NSTAT_EXPORT_COUNT', 0);
        run_script_in_base(instrumentedPath);
        produced = getappdata(groot, 'NSTAT_EXPORT_COUNT');
        if isempty(produced)
            produced = 0;
        end
        produced = double(produced);

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
    if isappdata(groot, 'NSTAT_EXPORT_OUTDIR')
        rmappdata(groot, 'NSTAT_EXPORT_OUTDIR');
    end
    if isappdata(groot, 'NSTAT_EXPORT_COUNT')
        rmappdata(groot, 'NSTAT_EXPORT_COUNT');
    end
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

function tf = is_classdef_source_file(sourcePath)
tf = false;
if ~exist(sourcePath, 'file')
    return;
end
textPayload = fileread(sourcePath);
lines = splitlines(textPayload);
for iLine = 1:numel(lines)
    current = strtrim(char(lines(iLine)));
    if isempty(current) || startsWith(current, '%')
        continue;
    end
    tf = startsWith(lower(current), 'classdef ');
    return;
end
end

function instrument_script_for_figure_capture(sourcePath, outPath)
sourceText = fileread(sourcePath);
sourceLines = splitlines(sourceText);
captureBlock = get_capture_block();
outLines = cell(0, 1);

for iLine = 1:numel(sourceLines)
    lineText = char(sourceLines(iLine));
    trimmed = lower(strtrim(lineText));
    isCloseAll = startsWith(trimmed, 'close all');
    isClose = strcmp(trimmed, 'close') || startsWith(trimmed, 'close(');
    if isCloseAll || isClose
        outLines = [outLines; captureBlock]; %#ok<AGROW>
    end
    outLines{end+1,1} = lineText; %#ok<AGROW>
end
outLines = [outLines; captureBlock]; %#ok<AGROW>

fid = fopen(outPath, 'w');
if fid < 0
    error('export_helpfile_figures:InstrumentWriteFailed', ...
        'Could not write instrumented script: %s', outPath);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', outLines{:});
end

function block = get_capture_block()
block = {
    '% nSTAT export capture block'
    'if ~isappdata(groot, ''NSTAT_EXPORT_COUNT'')'
    '    setappdata(groot, ''NSTAT_EXPORT_COUNT'', 0);'
    'end'
    'if ~isappdata(groot, ''NSTAT_EXPORT_OUTDIR'')'
    '    setappdata(groot, ''NSTAT_EXPORT_OUTDIR'', pwd);'
    'end'
    'count_capture = getappdata(groot, ''NSTAT_EXPORT_COUNT'');'
    'outRoot_capture = getappdata(groot, ''NSTAT_EXPORT_OUTDIR'');'
    'figs_capture = findall(groot, ''Type'', ''figure'');'
    'if ~isempty(figs_capture)'
    '    figNums_capture = arrayfun(@(h) h.Number, figs_capture);'
    '    [~, ord_capture] = sort(figNums_capture);'
    '    figs_capture = figs_capture(ord_capture);'
    '    for idx_capture = 1:numel(figs_capture)'
    '        if ~isappdata(figs_capture(idx_capture), ''nstat_export_saved'')'
    '            count_capture = count_capture + 1;'
    '            outFile_capture = fullfile(outRoot_capture, sprintf(''fig_%03d.png'', count_capture));'
    '            try'
    '                exportgraphics(figs_capture(idx_capture), outFile_capture, ''Resolution'', 180);'
    '            catch'
    '                saveas(figs_capture(idx_capture), outFile_capture);'
    '            end'
    '            setappdata(figs_capture(idx_capture), ''nstat_export_saved'', true);'
    '        end'
    '    end'
    'end'
    'setappdata(groot, ''NSTAT_EXPORT_COUNT'', count_capture);'
    'clear figs_capture figNums_capture ord_capture idx_capture outFile_capture count_capture outRoot_capture;'
    };
end
