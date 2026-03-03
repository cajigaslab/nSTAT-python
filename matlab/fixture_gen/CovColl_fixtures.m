function CovColl_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for the CovColl class.
%
% MATLAB reference: CovColl.m constructor/core utilities/toStructure/fromStructure

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'CovColl');
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

time = (0:0.1:1)';
cov1 = Covariate(time, sin(2*pi*time), 'sine', 'time', 's', '', {'sine'});
cov2 = Covariate(time, [time time.^2], 'poly', 'time', 's', '', {'t', 't2'});
cc = CovColl({cov1, cov2});

initial_numCov = cc.numCov;
initial_covDimensions = cc.covDimensions;
initial_sampleRate = cc.sampleRate;
initial_minTime = cc.minTime;
initial_maxTime = cc.maxTime;
initial_labels = cc.getAllCovLabels();
initial_cov_mask = cc.covMask;
initial_data_matrix = cc.dataToMatrix('standard');
initial_cov_inds = cc.getCovIndicesFromNames({'sine', 'poly'});
initial_is_cov_present = cc.isCovPresent('sine');

cc_shift = cc.copy();
cc_shift.setCovShift(0.2);
shift_covShift = cc_shift.covShift;
shift_minTime = cc_shift.minTime;
shift_maxTime = cc_shift.maxTime;
cc_shift.resetCovShift();
reset_covShift = cc_shift.covShift;
reset_minTime = cc_shift.minTime;
reset_maxTime = cc_shift.maxTime;

cc_sr = cc.copy();
cc_sr.setSampleRate(5);
sr_sampleRate = cc_sr.sampleRate;
sr_data_matrix = cc_sr.dataToMatrix('standard');

cc_win = cc.copy();
cc_win.restrictToTimeWindow(0.2, 0.8);
win_minTime = cc_win.minTime;
win_maxTime = cc_win.maxTime;
win_data_matrix = cc_win.dataToMatrix('standard');

struct_payload = cc.toStructure();
cc_roundtrip = CovColl.fromStructure(struct_payload);
roundtrip_numCov = cc_roundtrip.numCov;
roundtrip_covDimensions = cc_roundtrip.covDimensions;
roundtrip_sampleRate = cc_roundtrip.sampleRate;
roundtrip_minTime = cc_roundtrip.minTime;
roundtrip_maxTime = cc_roundtrip.maxTime;
roundtrip_labels = cc_roundtrip.getAllCovLabels();
roundtrip_data_matrix = cc_roundtrip.dataToMatrix('standard');

cc_removed = cc.copy();
cc_removed.removeCovariate(2);
removed_numCov = cc_removed.numCov;
removed_labels = cc_removed.getAllCovLabels();
removed_data_matrix = cc_removed.dataToMatrix('standard');

save(outputFile, ...
    'initial_numCov', ...
    'initial_covDimensions', ...
    'initial_sampleRate', ...
    'initial_minTime', ...
    'initial_maxTime', ...
    'initial_labels', ...
    'initial_cov_mask', ...
    'initial_data_matrix', ...
    'initial_cov_inds', ...
    'initial_is_cov_present', ...
    'shift_covShift', ...
    'shift_minTime', ...
    'shift_maxTime', ...
    'reset_covShift', ...
    'reset_minTime', ...
    'reset_maxTime', ...
    'sr_sampleRate', ...
    'sr_data_matrix', ...
    'win_minTime', ...
    'win_maxTime', ...
    'win_data_matrix', ...
    'struct_payload', ...
    'roundtrip_numCov', ...
    'roundtrip_covDimensions', ...
    'roundtrip_sampleRate', ...
    'roundtrip_minTime', ...
    'roundtrip_maxTime', ...
    'roundtrip_labels', ...
    'roundtrip_data_matrix', ...
    'removed_numCov', ...
    'removed_labels', ...
    'removed_data_matrix');

fprintf('Wrote CovColl fixtures to %s\n', outputFile);
end
