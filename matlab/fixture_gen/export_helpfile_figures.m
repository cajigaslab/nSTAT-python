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

envDataDir = getenv('NSTAT_DATA_DIR');
if ~isempty(envDataDir) && exist(envDataDir, 'dir') == 7
    addpath(genpath(envDataDir), '-begin');
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
        publishSourcePath = resolve_publish_source(topic, sourcePath, sourceType);
        [publishScriptPath, publishWorkDir] = prepare_publish_source( ...
            topic, publishSourcePath, fileparts(sourcePath));
        publishOutDir = fullfile(tempdir(), [topic '__publish_output__']);
        if exist(publishOutDir, 'dir')
            rmdir(publishOutDir, 's');
        end
        mkdir(publishOutDir);

        currentDir = pwd;
        restoreDir = onCleanup(@() cd(currentDir)); %#ok<NASGU>
        cd(publishWorkDir);
        publishOptions = struct( ...
            'format', 'html', ...
            'outputDir', publishOutDir, ...
            'showCode', false, ...
            'evalCode', true);
        publish(publishScriptPath, publishOptions);

        produced = collect_published_figures(topic, publishOutDir, outDir);

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

function publishSourcePath = resolve_publish_source(topic, sourcePath, sourceType)
if strcmpi(sourceType, 'm')
    publishSourcePath = sourcePath;
    return;
end

sourceDir = fileparts(sourcePath);
siblingMPath = fullfile(sourceDir, [topic '.m']);
if exist(siblingMPath, 'file')
    publishSourcePath = siblingMPath;
    return;
end

publishSourcePath = fullfile(tempdir(), [topic '__publish_source__.m']);
if exist(publishSourcePath, 'file')
    delete(publishSourcePath);
end
matlab.internal.liveeditor.openAndConvert(sourcePath, publishSourcePath);
end

function [publishScriptPath, publishWorkDir] = prepare_publish_source(topic, publishSourcePath, companionDir)
publishWorkDir = fullfile(tempdir(), [topic '__publish_work']);
if exist(publishWorkDir, 'dir')
    rmdir(publishWorkDir, 's');
end
mkdir(publishWorkDir);

publishScriptPath = fullfile(publishWorkDir, [topic '.m']);
copyfile(publishSourcePath, publishScriptPath);

assetPatterns = {'*.mat', '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'};
for iPattern = 1:numel(assetPatterns)
    assets = dir(fullfile(companionDir, assetPatterns{iPattern}));
    for iAsset = 1:numel(assets)
        copyfile(fullfile(companionDir, assets(iAsset).name), ...
            fullfile(publishWorkDir, assets(iAsset).name));
    end
end
end

function produced = collect_published_figures(topic, publishOutDir, outDir)
pngFiles = dir(fullfile(publishOutDir, '*.png'));
if isempty(pngFiles)
    produced = 0;
    return;
end

ordered = cell(0, 1);
numericEntries = struct('path', {}, 'index', {});
topicPrefix = [topic '_'];
for i = 1:numel(pngFiles)
    name = pngFiles(i).name;
    if contains(name, '_eq')
        continue;
    end
    if startsWith(name, topicPrefix)
        token = regexp(name, [regexptranslate('escape', topicPrefix) '(\d+)\.png$'], ...
            'tokens', 'once');
        if ~isempty(token)
            numericEntries(end+1).path = fullfile(publishOutDir, name); %#ok<AGROW>
            numericEntries(end).index = str2double(token{1}); %#ok<AGROW>
        end
    end
end
if ~isempty(numericEntries)
    [~, order] = sort([numericEntries.index]);
    ordered = {numericEntries(order).path}';
else
    plainPath = fullfile(publishOutDir, [topic '.png']);
    if exist(plainPath, 'file')
        ordered = {plainPath};
    end
end

for i = 1:numel(ordered)
    outFile = fullfile(outDir, sprintf('fig_%03d.png', i));
    copyfile(ordered{i}, outFile);
end
produced = numel(ordered);
end

function write_instrumented_script(sourcePath, instrumentedPath, outputDir)
sourceText = fileread(sourcePath);
sourceText = strrep(sourceText, sprintf('\r\n'), sprintf('\n'));
sourceText = strrep(sourceText, sprintf('\r'), sprintf('\n'));
lines = regexp(sourceText, '\n', 'split');

bodyLines = cell(0, 1);
bodyLines{end+1,1} = sprintf('nstat_export_init(''%s'');', ...
    strrep(outputDir, '''', ''''''));
for i = 1:numel(lines)
    line = lines{i};
    trimmed = strtrim(line);
    if should_flush_before_line(trimmed)
        bodyLines{end+1,1} = 'nstat_export_before_transition();';
    end
    bodyLines{end+1,1} = line;
end
bodyLines{end+1,1} = 'nstat_export_flush_all();';
bodyLines{end+1,1} = '';
bodyLines{end+1,1} = local_helper_source();

fid = fopen(instrumentedPath, 'w');
if fid < 0
    error('export_helpfile_figures:InstrumentedWriteFailed', ...
        'Could not write instrumented script: %s', instrumentedPath);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', bodyLines{:});
end

function tf = should_flush_before_line(line)
if isempty(line) || startsWith(line, '%')
    tf = false;
    return;
end
tf = ~isempty(regexp(line, '\<figure\>\s*(\(|$)', 'once')) || ...
    ~isempty(regexp(line, '^\s*(close(\s+all)?|clf)\b', 'once'));
end

function text = local_helper_source()
helperLines = {
    'function nstat_export_init(outputDir)'
    'state = struct(''outputDir'', outputDir, ''savedFigures'', {{}}, ''nextOrdinal'', 1);'
    'setappdata(groot, ''NSTAT_EXPORT_STATE'', state);'
    'if exist(outputDir, ''dir'') ~= 7'
    '    mkdir(outputDir);'
    'end'
    'end'
    ''
    'function nstat_export_before_transition()'
    'nstat_export_flush_open_figures();'
    'end'
    ''
    'function nstat_export_flush_all()'
    'nstat_export_flush_open_figures();'
    'end'
    ''
    'function nstat_export_flush_open_figures()'
    'if ~isappdata(groot, ''NSTAT_EXPORT_STATE'')'
    '    return;'
    'end'
    'state = getappdata(groot, ''NSTAT_EXPORT_STATE'');'
    'figs = get(groot, ''Children'');'
    'if isempty(figs)'
    '    state.savedFigures = cell(0, 1);'
    '    setappdata(groot, ''NSTAT_EXPORT_STATE'', state);'
    '    return;'
    'end'
    'figs = figs(:);'
    'figs = flipud(figs);'
    'for iFig = 1:numel(figs)'
    '    fig = figs(iFig);'
    '    if ~isgraphics(fig, ''figure'')'
    '        continue;'
    '    end'
    '    if nstat_export_is_saved(state, fig)'
    '        continue;'
    '    end'
    '    outFile = fullfile(state.outputDir, sprintf(''fig_%03d.png'', state.nextOrdinal));'
    '    try'
    '        exportgraphics(fig, outFile, ''Resolution'', 180);'
    '    catch'
    '        saveas(fig, outFile);'
    '    end'
    '    state.savedFigures{end+1,1} = fig; %#ok<AGROW>'
    '    state.nextOrdinal = state.nextOrdinal + 1;'
    'end'
    'alive = cell(0, 1);'
    'for iSaved = 1:numel(state.savedFigures)'
    '    fig = state.savedFigures{iSaved};'
    '    if isgraphics(fig, ''figure'')'
    '        alive{end+1,1} = fig; %#ok<AGROW>'
    '    end'
    'end'
    'state.savedFigures = alive;'
    'setappdata(groot, ''NSTAT_EXPORT_STATE'', state);'
    'end'
    ''
    'function tf = nstat_export_is_saved(state, fig)'
    'tf = false;'
    'for iSaved = 1:numel(state.savedFigures)'
    '    if isequal(state.savedFigures{iSaved}, fig)'
    '        tf = true;'
    '        return;'
    '    end'
    'end'
    'end'
    };
text = sprintf('%s\n', helperLines{:});
end
