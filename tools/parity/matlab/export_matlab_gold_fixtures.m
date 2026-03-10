function export_matlab_gold_fixtures(repoRoot, matlabRepoRoot, fixtureNames)
if nargin < 1 || isempty(repoRoot)
    error('repoRoot is required');
end
if nargin < 2 || isempty(matlabRepoRoot)
    matlabRepoRoot = fullfile(fileparts(repoRoot), 'nSTAT');
end
if nargin < 3 || isempty(fixtureNames)
    fixtureNames = {};
end

repoRoot = char(repoRoot);
matlabRepoRoot = char(matlabRepoRoot);
fixtureNames = cellstr(string(fixtureNames));

addpath(matlabRepoRoot);
addpath(fullfile(matlabRepoRoot, 'helpfiles'));
addpath(genpath(fullfile(matlabRepoRoot, 'libraries')));

fixtureRoot = fullfile(repoRoot, 'tests', 'parity', 'fixtures', 'matlab_gold');
if ~exist(fixtureRoot, 'dir')
    mkdir(fixtureRoot);
end

if should_export(fixtureNames, 'signalobj'); export_signalobj_fixture(fixtureRoot); end
if should_export(fixtureNames, 'confidence_interval'); export_confidence_interval_fixture(fixtureRoot); end
if should_export(fixtureNames, 'covariate'); export_covariate_fixture(fixtureRoot); end
if should_export(fixtureNames, 'nspiketrain'); export_nspiketrain_fixture(fixtureRoot); end
if should_export(fixtureNames, 'nstcoll'); export_nstcoll_fixture(fixtureRoot); end
if should_export(fixtureNames, 'config'); export_config_fixture(fixtureRoot); end
if should_export(fixtureNames, 'covcoll'); export_covcoll_fixture(fixtureRoot); end
if should_export(fixtureNames, 'trial'); export_trial_fixture(fixtureRoot); end
if should_export(fixtureNames, 'events'); export_events_fixture(fixtureRoot); end
if should_export(fixtureNames, 'history'); export_history_fixture(fixtureRoot); end
if should_export(fixtureNames, 'cif'); export_cif_fixture(fixtureRoot); end
if should_export(fixtureNames, 'analysis'); export_analysis_fixture(fixtureRoot); end
if should_export(fixtureNames, 'analysis_binomial'); export_analysis_binomial_fixture(fixtureRoot); end
if should_export(fixtureNames, 'analysis_validation'); export_analysis_validation_fixture(fixtureRoot); end
if should_export(fixtureNames, 'analysis_multineuron'); export_analysis_multineuron_fixture(fixtureRoot); end
if should_export(fixtureNames, 'ksdiscrete'); export_ksdiscrete_fixture(fixtureRoot); end
if should_export(fixtureNames, 'fit_summary'); export_fit_summary_fixture(fixtureRoot); end
if should_export(fixtureNames, 'point_process'); export_point_process_fixture(fixtureRoot); end
if should_export(fixtureNames, 'thinning'); export_thinning_fixture(fixtureRoot); end
if should_export(fixtureNames, 'decoding_predict'); export_decoding_predict_fixture(fixtureRoot); end
if should_export(fixtureNames, 'decoding_smoother'); export_decoding_smoother_fixture(fixtureRoot); end
if should_export(fixtureNames, 'hybrid_filter'); export_hybrid_filter_fixture(fixtureRoot); end
if should_export(fixtureNames, 'nonlinear_decode'); export_nonlinear_decode_fixture(fixtureRoot); end
if should_export(fixtureNames, 'simulated_network'); export_simulated_network_fixture(fixtureRoot); end
end

function tf = should_export(fixtureNames, name)
if isempty(fixtureNames)
    tf = true;
    return;
end
tf = any(strcmp(string(fixtureNames), string(name)));
end

function export_history_fixture(fixtureRoot)
histObj = History([0 0.5 1.0], 0.0, 1.0);
n1 = nspikeTrain([0.0 0.5 1.0], 'n1', 2.0, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
n2 = nspikeTrain([0.25 0.75], 'n2', 2.0, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
coll = nstColl({n1, n2});

singleCov = histObj.computeHistory(n1, 1);
collCov = histObj.computeHistory(coll, 2);
structure = histObj.toStructure;
roundtrip = History.fromStructure(structure);
filterMat = histObj.toFilter(0.5);

fig = figure('Visible', 'off');
histObj.plot();
ax = gca;
lineHandles = flipud(findobj(ax, 'Type', 'line'));
lineLabels = cell(1, numel(lineHandles));
lineX = cell(1, numel(lineHandles));
lineY = cell(1, numel(lineHandles));
for iLine = 1:numel(lineHandles)
    lineLabels{iLine} = get(lineHandles(iLine), 'DisplayName');
    lineX{iLine} = get(lineHandles(iLine), 'XData');
    lineY{iLine} = get(lineHandles(iLine), 'YData');
end
close(fig);

payload = struct();
payload.windowTimes = histObj.windowTimes;
payload.minTime = histObj.minTime;
payload.maxTime = histObj.maxTime;
payload.structure_windowTimes = structure.windowTimes;
payload.roundtrip_windowTimes = roundtrip.windowTimes;
payload.roundtrip_minTime = roundtrip.minTime;
payload.roundtrip_maxTime = roundtrip.maxTime;
payload.single_history_matrix = singleCov.dataToMatrix();
payload.single_history_labels = singleCov.getAllCovLabels;
payload.single_history_name = singleCov.getCov(1).name;
payload.coll_history_matrix = collCov.dataToMatrix();
payload.coll_history_labels = collCov.getAllCovLabels;
payload.coll_cov_names = cell(1, collCov.numCov);
for iCov = 1:collCov.numCov
    payload.coll_cov_names{iCov} = collCov.getCov(iCov).name;
end
payload.filter_num = cell(size(filterMat));
payload.filter_den = cell(size(filterMat));
for idx = 1:numel(filterMat)
    payload.filter_num{idx} = filterMat(idx).Numerator{1};
    payload.filter_den{idx} = filterMat(idx).Denominator{1};
end
payload.filter_delta = filterMat.Ts;
payload.plot_labels = lineLabels;
payload.plot_x = lineX;
payload.plot_y = lineY;

save(fullfile(fixtureRoot, 'history_exactness.mat'), '-struct', 'payload');
end

function export_events_fixture(fixtureRoot)
events = Events([0.2 0.7], {'E1','E2'}, 'g');
fig = figure('Visible', 'off');
ax = axes('Parent', fig);
axis(ax, [0 1 -1 2]);
events.plot(ax);

lineHandles = flipud(findobj(ax, 'Type', 'line'));
textHandles = flipud(findobj(ax, 'Type', 'text'));

payload = struct();
payload.eventTimes = events.eventTimes;
payload.eventLabels = events.eventLabels;
payload.eventColor = events.eventColor;
payload.axis_limits = axis(ax);
payload.plot_line_xdata = get(lineHandles(1), 'XData');
payload.plot_line_ydata = get(lineHandles(1), 'YData');
payload.plot_line_color = get(lineHandles(1), 'Color');
payload.plot_line_width = get(lineHandles(1), 'LineWidth');
payload.plot_label_strings = get(textHandles, 'String');
payload.plot_label_positions = cell2mat(get(textHandles, 'Position')');

close(fig);
save(fullfile(fixtureRoot, 'events_exactness.mat'), '-struct', 'payload');
end

function export_confidence_interval_fixture(fixtureRoot)
t = (0:0.1:0.4)';
bounds = [0.9 1.1; 1.9 2.1; 2.9 3.1; 3.9 4.1; 4.9 5.1];
ci = ConfidenceInterval(t, bounds, 'CI', 'time', 's', 'a.u.', {'lo','hi'}, {'-.k'});
ci.setColor('r');
ci.setValue(0.9);
structure = ci.dataToStructure;
roundtrip = ConfidenceInterval.fromStructure(structure);

fig = figure('Visible','off');
ax = axes('Parent', fig);
ci.plot('r', 0.2, 0);
lineHandles = flipud(findobj(ax, 'Type', 'line'));
lineColors = zeros(numel(lineHandles), 3);
for iLine = 1:numel(lineHandles)
    lineColors(iLine, :) = get(lineHandles(iLine), 'Color');
end
close(fig);

fig = figure('Visible','off');
ax = axes('Parent', fig);
ci.plot([0.1 0.2 0.3], 0.4, 1);
patchHandle = findobj(ax, 'Type', 'patch');
patchFaceColor = get(patchHandle, 'FaceColor');
patchEdgeColor = get(patchHandle, 'EdgeColor');
patchFaceAlpha = get(patchHandle, 'FaceAlpha');
close(fig);

payload = struct();
payload.time = ci.time;
payload.bounds = ci.data;
payload.color = ci.color;
payload.value = ci.value;
payload.name = ci.name;
payload.xlabelval = ci.xlabelval;
payload.xunits = ci.xunits;
payload.yunits = ci.yunits;
payload.dataLabels = ci.dataLabels;
payload.plotProps = ci.plotProps;
payload.structure_time = structure.time;
payload.structure_values = structure.signals.values;
payload.roundtrip_bounds = roundtrip.data;
payload.roundtrip_color = roundtrip.color;
payload.roundtrip_value = roundtrip.value;
payload.roundtrip_name = roundtrip.name;
payload.roundtrip_plotProps = roundtrip.plotProps;
payload.line_plot_colors = lineColors;
payload.patch_face_color = patchFaceColor;
payload.patch_edge_color = patchEdgeColor;
payload.patch_face_alpha = patchFaceAlpha;

save(fullfile(fixtureRoot, 'confidence_interval_exactness.mat'), '-struct', 'payload');
end

function export_signalobj_fixture(fixtureRoot)
t = (0:0.1:0.4)';
data = [1 0; 2 1; 4 0; 8 -1; 16 0];
s = SignalObj(t, data, 'sig', 'time', 's', 'u', {'x1', 'x2'});
s1 = s.getSubSignal(1);
s2 = SignalObj((0.05:0.1:0.45)', [0; 1; 0; -1; 0], 'sig2', 'time', 's', 'u', {'x3'});
specTime = (0:0.01:0.99)';
specData = sin(2*pi*5*specTime);
specSig = SignalObj(specTime, specData, 'spec', 'time', 's', 'u', {'spec'});

filtered = s.filter([0.25 0.5 0.25], 1);
resampled = s.resample(20);
derivative = s.derivative;
integral_sig = s.integral();
xc = xcorr(s.getSubSignal(1), s.getSubSignal(2), 2);
xcv = xcov(s.getSubSignal(1), s.getSubSignal(2), 2);
[s1c, s2c] = s1.makeCompatible(s2, 1);
periodogramCell = specSig.periodogram();
if iscell(periodogramCell)
    periodogramObj = periodogramCell{1};
else
    periodogramObj = periodogramCell;
end
mtmCell = specSig.MTMspectrum();
if iscell(mtmCell)
    mtmObj = mtmCell{1};
else
    mtmObj = mtmCell;
end
[spectrogramObj, ~] = specSig.spectrogram();
if iscell(spectrogramObj)
    spectrogramObj = spectrogramObj{1};
end

payload = struct();
payload.time = s.time;
payload.data = s.data;
payload.spec_time = specSig.time;
payload.spec_data = specSig.data;
payload.filter_b = [0.25 0.5 0.25];
payload.filter_a = 1;
payload.filtered_data = filtered.data;
payload.derivative_data = derivative.data;
payload.integral_data = integral_sig.data;
payload.resample_rate = 20;
payload.resampled_time = resampled.time;
payload.resampled_data = resampled.data;
payload.xcorr_maxlag = 2;
payload.xcorr_time = xc.time;
payload.xcorr_data = xc.data;
payload.xcov_time = xcv.time;
payload.xcov_data = xcv.data;
payload.periodogram_frequency = periodogramObj.Frequencies;
payload.periodogram_power = periodogramObj.Data;
payload.mtm_frequency = mtmObj.Frequencies;
payload.mtm_power = mtmObj.Data(:,1);
payload.spectrogram_time = spectrogramObj.t;
payload.spectrogram_frequency = spectrogramObj.f;
payload.spectrogram_power = spectrogramObj.p;
payload.compat_time = s1c.time;
payload.compat_left_data = s1c.data;
payload.compat_right_data = s2c.data;

save(fullfile(fixtureRoot, 'signalobj_exactness.mat'), '-struct', 'payload');
end

function export_nspiketrain_fixture(fixtureRoot)
spikeTimes = [0.05; 0.10; 0.11; 0.30; 0.47];
binwidth = 0.05;
nst = nspikeTrain(spikeTimes, 'nst', binwidth, 0.0, 0.5, 'time', 's', 'spikes', 'spk', 0);
sig = nst.getSigRep(binwidth, 0.0, 0.5);
parts = nst.partitionNST([0.0 0.2 0.5]);
restoreTrain = nspikeTrain(spikeTimes, 'restore', 0.2, -0.1, 0.8, 'time', 's', 'spikes', 'spk', -1);
restoreTrain.setSigRep(0.1, -0.1, 0.8);
restoreTrain.setMinTime(-0.3);
restoreTrain.setMaxTime(1.1);
restoreTrain.restoreToOriginal();
burstTrain = nspikeTrain([0.0; 0.001; 0.002; 0.007; 0.507; 0.508; 0.509; 0.514], 'bursting', 0.001, 0.0, 0.6, 'time', 's', 'spikes', 'spk', 0);

payload = struct();

fig = figure('Visible','off');
ax = axes('Parent', fig);
h = nst.plotISISpectrumFunction();
payload.isi_spectrum_x = get(h,'XData');
payload.isi_spectrum_y = get(h,'YData');
close(fig);

fig = figure('Visible','off');
ax = axes('Parent', fig);
nst.plotJointISIHistogram();
jointLines = flipud(findobj(ax, 'Type', 'line'));
for iLine = 1:numel(jointLines)
    payload.joint_isi_x{iLine} = get(jointLines(iLine), 'XData');
    payload.joint_isi_y{iLine} = get(jointLines(iLine), 'YData');
    payload.joint_isi_style{iLine} = get(jointLines(iLine), 'LineStyle');
end
close(fig);

fig = figure('Visible','off');
ax = axes('Parent', fig);
counts = nst.plotISIHistogram();
histBars = findobj(ax, 'Type', 'patch');
payload.isi_hist_counts = counts;
if(~isempty(histBars))
    payload.isi_hist_face_color = get(histBars(1), 'FaceColor');
    payload.isi_hist_edge_color = get(histBars(1), 'EdgeColor');
end
close(fig);

fig = figure('Visible','off');
ax = axes('Parent', fig);
nst.plotProbPlot();
probLines = flipud(findobj(ax, 'Type', 'line'));
for iLine = 1:numel(probLines)
    payload.probplot_x{iLine} = get(probLines(iLine), 'XData');
    payload.probplot_y{iLine} = get(probLines(iLine), 'YData');
    payload.probplot_style{iLine} = get(probLines(iLine), 'LineStyle');
end
close(fig);

fig = nst.plotExponentialFit();
payload.expfit_num_axes = numel(findobj(fig, 'Type', 'axes'));
close(fig);
payload.spikeTimes = spikeTimes;
payload.binwidth = binwidth;
payload.minTime = 0.0;
payload.maxTime = 0.5;
payload.sig_time = sig.time;
payload.sig_data = sig.data;
payload.isis = nst.getISIs();
payload.avgFiringRate = nst.avgFiringRate;
payload.B = nst.B;
payload.An = nst.An;
payload.burstIndex = nst.burstIndex;
payload.numBursts = nst.numBursts;
payload.numSpikesPerBurst = nst.numSpikesPerBurst;
payload.part1_spikes = parts.getNST(1).spikeTimes;
payload.part2_spikes = parts.getNST(2).spikeTimes;
payload.restore_min_time = restoreTrain.minTime;
payload.restore_max_time = restoreTrain.maxTime;
payload.burst_avgSpikesPerBurst = burstTrain.avgSpikesPerBurst;
payload.burst_stdSpikesPerBurst = burstTrain.stdSpikesPerBurst;
payload.burst_numBursts = burstTrain.numBursts;
payload.burst_numSpikesPerBurst = burstTrain.numSpikesPerBurst;

save(fullfile(fixtureRoot, 'nspiketrain_exactness.mat'), '-struct', 'payload');
end

function export_covariate_fixture(fixtureRoot)
t = (0:0.1:0.4)';
replicates = [0.0 0.1 0.2 0.3; 0.2 0.3 0.4 0.5; 0.4 0.5 0.6 0.7; 0.6 0.7 0.8 0.9; 0.8 0.9 1.0 1.1];
cov = Covariate(t, replicates, 'Stimulus', 'time', 's', 'a.u.', {'r1','r2','r3','r4'});
meanCov = cov.computeMeanPlusCI(0.05);
ci = ConfidenceInterval(t, [mean(replicates,2)-0.1, mean(replicates,2)+0.1], 'CI', 'time', 's', 'a.u.');
covSingle = Covariate(t, mean(replicates,2), 'StimulusSingle', 'time', 's', 'a.u.', {'stim'});
covSingle.setConfInterval(ci);
structure = covSingle.toStructure;
roundtrip = Covariate.fromStructure(structure);
if(iscell(structure.ci))
    ciStructure = structure.ci{1};
else
    ciStructure = structure.ci;
end

fig = figure('Visible','off');
plot(covSingle);
drawnow;
lineHandles = findobj(gca,'Type','line');
plotColors = zeros(length(lineHandles),3);
for i = 1:length(lineHandles)
    plotColors(i,:) = get(lineHandles(i),'Color');
end
close(fig);

payload = struct();
payload.time = t;
payload.replicates = replicates;
payload.mean_data = meanCov.data;
payload.mean_ci = meanCov.ci{1}.data;
payload.explicit_ci = covSingle.ci{1}.data;
payload.structure_time = structure.time;
payload.structure_data = structure.data;
payload.structure_name = structure.name;
payload.structure_dataLabels = structure.dataLabels;
payload.structure_plotProps = structure.plotProps;
payload.structure_ci_values = ciStructure.signals.values;
payload.structure_ci_name = ciStructure.name;
payload.roundtrip_data = roundtrip.data;
payload.roundtrip_ci = roundtrip.ci{1}.data;
payload.roundtrip_dataLabels = roundtrip.dataLabels;
payload.plot_line_colors = plotColors;

save(fullfile(fixtureRoot, 'covariate_exactness.mat'), '-struct', 'payload');
end

function export_nstcoll_fixture(fixtureRoot)
n1 = nspikeTrain([0.1 0.3], '1', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
n2 = nspikeTrain([0.2], '2', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
coll = nstColl({n1, n2});
dataMat = coll.dataToMatrix([1 2], 0.1, 0.0, 0.5);
collapsed = coll.toSpikeTrain;
coll.setNeighbors;
neighbors1 = coll.getNeighbors(1);
neighbors2 = coll.getNeighbors(2);
ensembleCov = coll.getEnsembleNeuronCovariates(1, [], [0.0 0.1]);
psthCov = coll.psth(0.1, [1 2], 0.0, 0.5);

payload = struct();
payload.numSpikeTrains = coll.numSpikeTrains;
payload.firstName = coll.getNST(1).name;
payload.dataMatrix = dataMat;
payload.firstSpikeTimes = coll.getNST(1).spikeTimes;
payload.secondSpikeTimes = coll.getNST(2).spikeTimes;
payload.collapsedSpikeTimes = collapsed.spikeTimes;
payload.collapsedName = collapsed.name;
payload.collapsedMinTime = collapsed.minTime;
payload.collapsedMaxTime = collapsed.maxTime;
payload.collapsedSampleRate = collapsed.sampleRate;
payload.firstSpikeTime = coll.getFirstSpikeTime;
payload.lastSpikeTime = coll.getLastSpikeTime;
payload.binarySigRep = coll.isSigRepBinary;
payload.nstNameFromInd1 = coll.getNSTnameFromInd(1);
payload.nstFromName1_spikeTimes = coll.getNSTFromName('1').spikeTimes;
[fieldVal, neuronNumbers] = coll.getFieldVal('avgFiringRate');
payload.fieldVal_avgFiringRate = fieldVal;
payload.fieldVal_neuronNumbers = neuronNumbers;
payload.neighbors1 = neighbors1;
payload.neighbors2 = neighbors2;
payload.ensemble_labels = ensembleCov.getAllCovLabels;
payload.ensemble_matrix = ensembleCov.dataToMatrix();
payload.psth_time = psthCov.time;
payload.psth_data = psthCov.data;
ss1 = nspikeTrain([0.1 0.3], '1', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
ss2 = nspikeTrain([0.2], '1', 10, 0.0, 0.5, 'time', 's', 'spikes', 'spk', -1);
ssColl = nstColl({ss1, ss2});
[xK,WK,Qhat,gammahat,logll,fitSummary] = ssColl.ssglm([0.0 0.1 0.2], 2, 2, 'binomial');
payload.ssglm_xK = xK;
payload.ssglm_WK = WK;
payload.ssglm_Qhat = Qhat;
payload.ssglm_gammahat = gammahat;
payload.ssglm_logll = logll;
payload.ssglm_firstSpikeTimes = ss1.spikeTimes;
payload.ssglm_secondSpikeTimes = ss2.spikeTimes;
payload.ssglm_summary_AIC = fitSummary.AIC;
payload.ssglm_summary_BIC = fitSummary.BIC;
payload.ssglm_summary_logLL = fitSummary.logLL;
payload.ssglm_summary_KSStats = fitSummary.KSStats.ks_stat;
payload.ssglm_summary_KSPvalues = fitSummary.KSStats.pValue;
payload.ssglm_summary_withinConfInt = fitSummary.KSStats.withinConfInt;

save(fullfile(fixtureRoot, 'nstcoll_exactness.mat'), '-struct', 'payload');
end

function export_config_fixture(fixtureRoot)
cfg = TrialConfig({{'Position','x'},{'Stimulus'}}, 2.0, [0 0.5 1.0], [], [], 0.5, 'stim_pos');
cfg2 = TrialConfig({{'Stimulus'}}, 2.0, [], [], [], [], 'manual');
structure = cfg.toStructure;
roundtrip = TrialConfig.fromStructure(structure);
coll = ConfigColl({cfg, cfg2});
subset = coll.getSubsetConfigs([1 2]);
rebuilt = ConfigColl.fromStructure(coll.toStructure);
defaultColl = ConfigColl();
emptyColl = ConfigColl([]);
renamed = ConfigColl({cfg, cfg2});
renamed.setConfigNames('', 1);
stringError = struct('identifier', '', 'message', '');
try
    ConfigColl('abc');
catch ME
    stringError.identifier = ME.identifier;
    stringError.message = ME.message;
end

t = (0:0.5:1.0)';
position = Covariate(t, [0 10; 1 11; 2 12], 'Position', 'time', 's', '', {'x','y'});
stimulus = Covariate(t, [5; 6; 7], 'Stimulus', 'time', 's', 'a.u.', {'stim'});
n1 = nspikeTrain([0.0 0.5 1.0], 'n1', 2.0, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
n2 = nspikeTrain([0.25 0.75], 'n2', 2.0, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
cfgApplied = TrialConfig({{'Position','x'},{'Stimulus'}}, 4.0, [0 0.5 1.0], [0 0.5 1.0], [0 1; 1 0], 0.25, 'stim_pos');
trial1 = Trial(nstColl({n1, n2}), CovColl({position, stimulus}));
cfgApplied.setConfig(trial1);
trial2 = Trial(nstColl({n1, n2}), CovColl({position, stimulus}));
ConfigColl({cfgApplied}).setConfig(trial2, 1);

payload = struct();
payload.cfg_name = cfg.name;
payload.cfg_sampleRate = cfg.sampleRate;
payload.cfg_covLag = cfg.covLag;
payload.cfg_covMask = cfg.covMask;
payload.roundtrip_name = roundtrip.name;
payload.roundtrip_covLag = roundtrip.covLag;
payload.roundtrip_ensCovMask = roundtrip.ensCovMask;
payload.roundtrip_covMask = roundtrip.covMask;
payload.config_names = coll.getConfigNames();
payload.subset_names = subset.getConfigNames();
payload.rebuilt_names = rebuilt.getConfigNames();
payload.rebuilt_first_name = rebuilt.getConfig(1).name;
payload.rebuilt_first_covLag = rebuilt.getConfig(1).covLag;
payload.rebuilt_first_ensCovMask = rebuilt.getConfig(1).ensCovMask;
payload.default_numConfigs = defaultColl.numConfigs;
payload.default_names = defaultColl.getConfigNames();
payload.empty_numConfigs = emptyColl.numConfigs;
payload.empty_names = emptyColl.getConfigNames();
payload.renamed_names = renamed.getConfigNames();
payload.string_error_identifier = stringError.identifier;
payload.string_error_message = stringError.message;
payload.applied_sampleRate = trial1.sampleRate;
payload.applied_flat_cov_mask = trial1.flattenCovMask();
payload.applied_history_windowTimes = trial1.history.windowTimes;
payload.applied_ens_history_windowTimes = trial1.ensCovHist.windowTimes;
payload.applied_ens_mask = trial1.ensCovMask;
payload.applied_shifted_position_time = trial1.covarColl.getCov(1).time;
payload.applied_coll_sampleRate = trial2.sampleRate;
payload.applied_coll_flat_cov_mask = trial2.flattenCovMask();
payload.applied_coll_history_windowTimes = trial2.history.windowTimes;
payload.applied_coll_ens_history_windowTimes = trial2.ensCovHist.windowTimes;
payload.applied_coll_ens_mask = trial2.ensCovMask;
payload.applied_coll_shifted_position_time = trial2.covarColl.getCov(1).time;

save(fullfile(fixtureRoot, 'config_exactness.mat'), '-struct', 'payload');
end

function export_covcoll_fixture(fixtureRoot)
t = (0:0.5:1.0)';
position = Covariate(t, [0 10; 1 11; 2 12], 'Position', 'time', 's', '', {'x','y'});
stimulus = Covariate(t, [5; 6; 7], 'Stimulus', 'time', 's', 'a.u.', {'stim'});
coll = CovColl({position, stimulus});
coll.setMask({{'Position','x'},{'Stimulus'}});
maskedLabels = coll.getCovLabelsFromMask;
maskedMatrix = coll.dataToMatrix();
maskedTime = coll.getCov(1).time;
dataStructure = coll.dataToStructure;
structure = coll.toStructure;
postMask1 = coll.covMask{1};
postMask2 = coll.covMask{2};
roundtrip = CovColl.fromStructure(structure);
copyColl = coll.copy;

shifted = CovColl({position, stimulus});
shifted.setCovShift(0.25);
shifted.restrictToTimeWindow(0.25, 1.25);
shiftedStim = shifted.getCov(2);

payload = struct();
payload.masked_labels = maskedLabels;
payload.masked_matrix = maskedMatrix;
payload.masked_time = maskedTime;
payload.data_structure_time = dataStructure.time;
payload.data_structure_values = dataStructure.signals.values;
payload.post_structure_mask_1 = postMask1;
payload.post_structure_mask_2 = postMask2;
payload.structure_numCov = structure.numCov;
payload.structure_minTime = structure.minTime;
payload.structure_maxTime = structure.maxTime;
payload.roundtrip_minTime = roundtrip.minTime;
payload.roundtrip_maxTime = roundtrip.maxTime;
payload.roundtrip_sampleRate = roundtrip.sampleRate;
payload.roundtrip_labels = roundtrip.getCovLabelsFromMask;
payload.roundtrip_matrix = roundtrip.dataToMatrix();
payload.shifted_minTime = shifted.minTime;
payload.shifted_maxTime = shifted.maxTime;
payload.shifted_stim_time = shiftedStim.time;
payload.is_present_position = coll.isCovPresent('Position');
payload.is_present_last_index = coll.isCovPresent(2);
payload.copy_numCov = copyColl.numCov;

save(fullfile(fixtureRoot, 'covcoll_exactness.mat'), '-struct', 'payload');
end

function export_trial_fixture(fixtureRoot)
t = (0:0.5:1.0)';
position = Covariate(t, [0 10; 1 11; 2 12], 'Position', 'time', 's', '', {'x','y'});
stimulus = Covariate(t, [5; 6; 7], 'Stimulus', 'time', 's', 'a.u.', {'stim'});
n1 = nspikeTrain([0.0 0.5 1.0], 'n1', 0.5, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
n2 = nspikeTrain([0.25 0.75], 'n2', 0.5, 0.0, 1.0, 'time', 's', 'spikes', 'spk', -1);
events = Events([0.25 0.75], {'cue','reward'}, 'g');
histObj = History([0.0 0.5 1.0]);

trial = Trial(nstColl({n1, n2}), CovColl({position, stimulus}), events, histObj);
trial.setEnsCovHist([0.0 0.5 1.0]);
trial.setTrialPartition([0.0 0.5 1.0]);
partition = trial.getTrialPartition;
trial.setTrialTimesFor('validation');
structure = trial.toStructure;
roundtrip = Trial.fromStructure(structure);
designMatrix = trial.getDesignMatrix(1);
spikeVector = trial.getSpikeVector;
spikeVector1 = trial.getSpikeVector(1);
ensCovMatrix = trial.getEnsCovMatrix(1);

payload = struct();
payload.partition = partition;
payload.validation_minTime = trial.minTime;
payload.validation_maxTime = trial.maxTime;
payload.hist_labels = trial.getHistLabels;
payload.ens_cov_labels = trial.getEnsCovLabelsFromMask(1);
payload.design_matrix = designMatrix;
payload.ens_cov_matrix = ensCovMatrix;
payload.spike_vector = spikeVector;
payload.spike_vector_1 = spikeVector1;
payload.event_labels = events.eventLabels;
payload.event_times = events.eventTimes;
payload.structure_trainingWindow = structure.trainingWindow;
payload.structure_validationWindow = structure.validationWindow;
payload.structure_minTime = structure.minTime;
payload.structure_maxTime = structure.maxTime;
payload.structure_covMask = structure.covMask;
payload.structure_ensCovMask = structure.ensCovMask;
payload.structure_neuronMask = structure.neuronMask;
payload.roundtrip_partition = roundtrip.getTrialPartition;
payload.roundtrip_minTime = roundtrip.minTime;
payload.roundtrip_maxTime = roundtrip.maxTime;
payload.roundtrip_design_matrix = roundtrip.getDesignMatrix(1);
payload.roundtrip_ens_cov_matrix = roundtrip.getEnsCovMatrix(1);
payload.roundtrip_hist_labels = roundtrip.getHistLabels;
payload.roundtrip_ens_cov_labels = roundtrip.getEnsCovLabelsFromMask(1);

save(fullfile(fixtureRoot, 'trial_exactness.mat'), '-struct', 'payload');
end

function export_cif_fixture(fixtureRoot)
cif = CIF([0.1 0.5], {'stim1', 'stim2'}, {'stim1', 'stim2'}, 'binomial');
stimVal = [0.6; -0.2];
polyCif = build_polynomial_binomial_cif([-2.0 -0.5 0.3 -0.2 -0.1 0.05]);
polyStim = [0.2; -0.4];

payload = struct();
payload.beta = [0.1 0.5];
payload.stimVal = stimVal;
payload.lambda_delta = cif.evalLambdaDelta(stimVal);
payload.gradient = cif.evalGradient(stimVal);
payload.gradient_log = cif.evalGradientLog(stimVal);
payload.jacobian = cif.evalJacobian(stimVal);
payload.jacobian_log = cif.evalJacobianLog(stimVal);
payload.poly_beta = [-2.0 -0.5 0.3 -0.2 -0.1 0.05];
payload.poly_stimVal = polyStim;
payload.poly_lambda_delta = polyCif.evalLambdaDelta(polyStim);
payload.poly_gradient = polyCif.evalGradient(polyStim);
payload.poly_gradient_log = polyCif.evalGradientLog(polyStim);
payload.poly_jacobian = polyCif.evalJacobian(polyStim);
payload.poly_jacobian_log = polyCif.evalJacobianLog(polyStim);

save(fullfile(fixtureRoot, 'cif_exactness.mat'), '-struct', 'payload');
end

function export_analysis_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}));
summary = FitResSummary({fit});
Analysis.KSPlot(fit, 1, 0);
Analysis.plotFitResidual(fit, 0.01, 0);
[glmLambda, glmB, glmDev, glmStats, glmAIC, glmBIC, glmLogLL, glmDistribution] = Analysis.GLMFit(trial, 1, 1, 'GLM');
[helperZ, helperU, helperXAxis, helperKSSorted, helperKSStat] = Analysis.computeKSStats(spikeTrain, fit.lambda, 1);
helperResidual = Analysis.computeFitResidual(spikeTrain, fit.lambda, 0.01);

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times = spikeTrain.spikeTimes;
payload.sample_rate = trial.sampleRate;
payload.coeffs = fit.getCoeffs(1);
payload.lambda_time = fit.lambda.time;
payload.lambda_data = fit.lambda.data(:,1);
payload.AIC = fit.AIC(1);
payload.BIC = fit.BIC(1);
payload.logLL = fit.logLL(1);
payload.distribution = fit.fitType{1};
payload.summaryAIC = summary.AIC(1);
payload.summaryBIC = summary.BIC(1);
payload.summarylogLL = summary.logLL(1);
payload.Z = fit.Z(:,1);
payload.U = fit.U(:,1);
payload.ks_xAxis = fit.KSStats.xAxis(:,1);
payload.ks_sorted = fit.KSStats.KSSorted(:,1);
payload.ks_stat = fit.KSStats.ks_stat(1);
payload.ks_pvalue = fit.KSStats.pValue(1);
payload.ks_within_conf_int = fit.KSStats.withinConfInt(1);
payload.residual_time = fit.Residual.time;
payload.residual_data = fit.Residual.data(:,1);
payload.glmfit_lambda_time = glmLambda.time;
payload.glmfit_lambda_data = glmLambda.data(:,1);
payload.glmfit_coeffs = glmB;
payload.glmfit_dev = glmDev;
payload.glmfit_AIC = glmAIC;
payload.glmfit_BIC = glmBIC;
payload.glmfit_logLL = glmLogLL;
payload.glmfit_distribution = glmDistribution;
payload.analysis_computeKSStats_Z = helperZ;
payload.analysis_computeKSStats_U = helperU;
payload.analysis_computeKSStats_xAxis = helperXAxis;
payload.analysis_computeKSStats_KSSorted = helperKSSorted;
payload.analysis_computeKSStats_ks_stat = helperKSStat;
payload.analysis_computeFitResidual_time = helperResidual.time;
payload.analysis_computeFitResidual_data = helperResidual.data(:,1);

Analysis.KSPlot(fit, 1, 1);
ksAx = gca;
payload.analysis_KSPlot_title = stringify_text(get(get(ksAx, 'Title'), 'String'));
payload.analysis_KSPlot_ylabel = stringify_text(get(get(ksAx, 'YLabel'), 'String'));
payload.analysis_KSPlot_xlabel = stringify_text(get(get(ksAx, 'XLabel'), 'String'));
payload.analysis_KSPlot_xticklabels = cellstr(get(ksAx, 'XTickLabel'));
close(ancestor(ksAx, 'figure'));

Analysis.plotFitResidual(fit, 0.01, 1);
residualAx = gca;
payload.analysis_plotFitResidual_title = stringify_text(get(get(residualAx, 'Title'), 'String'));
payload.analysis_plotFitResidual_ylabel = stringify_text(get(get(residualAx, 'YLabel'), 'String'));
payload.analysis_plotFitResidual_xlabel = stringify_text(get(get(residualAx, 'XLabel'), 'String'));
payload.analysis_plotFitResidual_xticklabels = cellstr(get(residualAx, 'XTickLabel'));
close(ancestor(residualAx, 'figure'));

Analysis.plotInvGausTrans(fit, 1);
invAx = gca;
payload.analysis_plotInvGausTrans_title = stringify_text(get(get(invAx, 'Title'), 'String'));
payload.analysis_plotInvGausTrans_ylabel = stringify_text(get(get(invAx, 'YLabel'), 'String'));
payload.analysis_plotInvGausTrans_xlabel = stringify_text(get(get(invAx, 'XLabel'), 'String'));
payload.analysis_plotInvGausTrans_xticklabels = cellstr(get(invAx, 'XTickLabel'));
close(ancestor(invAx, 'figure'));

Analysis.plotSeqCorr(fit);
seqAx = gca;
payload.analysis_plotSeqCorr_title = stringify_text(get(get(seqAx, 'Title'), 'String'));
payload.analysis_plotSeqCorr_ylabel = stringify_text(get(get(seqAx, 'YLabel'), 'String'));
payload.analysis_plotSeqCorr_xlabel = stringify_text(get(get(seqAx, 'XLabel'), 'String'));
payload.analysis_plotSeqCorr_xticklabels = cellstr(get(seqAx, 'XTickLabel'));
close(ancestor(seqAx, 'figure'));

Analysis.plotCoeffs(fit);
coeffAx = gca;
payload.analysis_plotCoeffs_title = stringify_text(get(get(coeffAx, 'Title'), 'String'));
payload.analysis_plotCoeffs_ylabel = stringify_text(get(get(coeffAx, 'YLabel'), 'String'));
payload.analysis_plotCoeffs_xlabel = stringify_text(get(get(coeffAx, 'XLabel'), 'String'));
payload.analysis_plotCoeffs_xticklabels = cellstr(get(coeffAx, 'XTickLabel'));
coeffLegend = legend(coeffAx);
payload.analysis_plotCoeffs_legend = {};
if ~isempty(coeffLegend) && isgraphics(coeffLegend)
    payload.analysis_plotCoeffs_legend = cellstr(coeffLegend.String);
end
close(ancestor(coeffAx, 'figure'));

save(fullfile(fixtureRoot, 'analysis_exactness.mat'), '-struct', 'payload');
end

function export_analysis_binomial_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain = nspikeTrain([0.1 0.3 0.7], '1', 10.0, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}), 0, 'BNLRCG');

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times = spikeTrain.spikeTimes;
payload.sample_rate = trial.sampleRate;
payload.coeffs = fit.getCoeffs(1);
payload.lambda_time = fit.lambda.time;
payload.lambda_data = fit.lambda.data(:,1);
payload.AIC = fit.AIC(1);
payload.BIC = fit.BIC(1);
payload.logLL = fit.logLL(1);
payload.distribution = fit.fitType{1};
payload.ks_stat = fit.KSStats.ks_stat(1);
payload.ks_pvalue = fit.KSStats.pValue(1);
payload.ks_within_conf_int = fit.KSStats.withinConfInt(1);
payload.residual_time = fit.Residual.time;
payload.residual_data = fit.Residual.data(:,1);

save(fullfile(fixtureRoot, 'analysis_binomial_exactness.mat'), '-struct', 'payload');
end

function export_analysis_validation_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
trial.setTrialPartition([0.0 0.5 1.0]);
trial.setTrialTimesFor('validation');
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}));

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times = spikeTrain.spikeTimes;
payload.partition = trial.getTrialPartition;
payload.validation_minTime = trial.minTime;
payload.validation_maxTime = trial.maxTime;
payload.design_matrix = trial.getDesignMatrix(1);
payload.lambda_time = fit.lambda.time;
payload.lambda_data = fit.lambda.data(:,1);
payload.coeffs = fit.getCoeffs(1);
payload.AIC = fit.AIC(1);
payload.BIC = fit.BIC(1);
payload.logLL = fit.logLL(1);
payload.ks_stat = fit.KSStats.ks_stat(1);
payload.ks_pvalue = fit.KSStats.pValue(1);
payload.ks_within_conf_int = fit.KSStats.withinConfInt(1);
payload.residual_time = fit.Residual.time;
payload.residual_data = fit.Residual.data(:,1);

plotHandle = fit.plotResults;
plotAxes = findall(plotHandle, 'Type', 'axes');
payload.plotResults_num_axes = numel(plotAxes);
payload.plotResults_titles = cell(1, numel(plotAxes));
payload.plotResults_ylabels = cell(1, numel(plotAxes));
payload.plotResults_xlabels = cell(1, numel(plotAxes));
for idx = 1:numel(plotAxes)
    ax = plotAxes(idx);
    payload.plotResults_titles{idx} = stringify_text(get(get(ax, 'Title'), 'String'));
    payload.plotResults_ylabels{idx} = stringify_text(get(get(ax, 'YLabel'), 'String'));
    payload.plotResults_xlabels{idx} = stringify_text(get(get(ax, 'XLabel'), 'String'));
end
close(plotHandle);

save(fullfile(fixtureRoot, 'analysis_validation_exactness.mat'), '-struct', 'payload');
end

function export_analysis_multineuron_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain1 = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
spikeTrain2 = nspikeTrain([0.2 0.6 0.9], '2', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain1, spikeTrain2}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
histCfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [0 0.1 0.2], []);
histCfg.setName('stim_hist');
fits = Analysis.RunAnalysisForAllNeurons(trial, ConfigColl({cfg, histCfg}), 0);
summary = FitResSummary(fits);

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times_1 = spikeTrain1.spikeTimes;
payload.spike_times_2 = spikeTrain2.spikeTimes;
payload.num_fits = numel(fits);
payload.fit1_coeffs = fits{1}.getCoeffs(1);
payload.fit2_coeffs = fits{2}.getCoeffs(1);
payload.fit1_hist_coeffs = fits{1}.getHistCoeffs(2);
payload.fit2_hist_coeffs = fits{2}.getHistCoeffs(2);
payload.fit1_AIC = fits{1}.AIC(1);
payload.fit2_AIC = fits{2}.AIC(1);
payload.fit1_hist_AIC = fits{1}.AIC(2);
payload.fit2_hist_AIC = fits{2}.AIC(2);
payload.fit1_BIC = fits{1}.BIC(1);
payload.fit2_BIC = fits{2}.BIC(1);
payload.fit1_hist_BIC = fits{1}.BIC(2);
payload.fit2_hist_BIC = fits{2}.BIC(2);
payload.fit1_logLL = fits{1}.logLL(1);
payload.fit2_logLL = fits{2}.logLL(1);
payload.fit1_hist_logLL = fits{1}.logLL(2);
payload.fit2_hist_logLL = fits{2}.logLL(2);
payload.summary_AIC = summary.AIC;
payload.summary_BIC = summary.BIC;
payload.summary_logLL = summary.logLL;
payload.summary_KSStats = summary.KSStats;
payload.summary_KSPvalues = summary.KSPvalues;
payload.summary_withinConfInt = summary.withinConfInt;
payload.summary_structure = summary.toStructure;

plotHandle = [];
try
    plotHandle = figure('Visible', 'off', 'Position', [100 100 1600 900]);
    h1 = subplot(2,4,[1 2 5 6]); summary.plotAllCoeffs(h1); grid off;
    title({'GLM Coefficients Across Neurons';'with 95% CIs (* p<0.05)'},'FontWeight','bold','FontSize',11,'FontName','Arial');
    subplot(2,4,[3 4]); boxplot(summary.KSStats, summary.fitNames, 'labelorientation', 'inline');
    ylabel('KS Statistics');
    hx = get(gca, 'XLabel'); hy = get(gca, 'YLabel');
    set([hx hy], 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'bold');
    title('KS Statistics Across Neurons', 'FontWeight', 'bold', 'FontSize', 11, 'FontName', 'Arial');
    subplot(2,4,7); summary.getDiffAIC(1);
    ylabel('\Delta AIC');
    hx = get(gca, 'XLabel'); hy = get(gca, 'YLabel');
    set([hx hy], 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'bold');
    title('Change in AIC Across Neurons', 'FontWeight', 'bold', 'FontSize', 11, 'FontName', 'Arial');
    set(gca, 'XTickLabelRotation', 90);
    subplot(2,4,8); summary.getDiffBIC(1);
    ylabel('\Delta BIC');
    hx = get(gca, 'XLabel'); hy = get(gca, 'YLabel');
    set([hx hy], 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'bold');
    title('Change in BIC Across Neurons', 'FontWeight', 'bold', 'FontSize', 11, 'FontName', 'Arial');
    set(gca, 'XTickLabelRotation', 90);
    allAxes = findall(plotHandle, 'Type', 'axes');
    for idx = 1:length(allAxes)
        ax = allAxes(idx);
        titleStr = stringify_text(get(get(ax, 'Title'), 'String'));
        ylabelStr = stringify_text(get(get(ax, 'YLabel'), 'String'));
        xtickLabels = cellstr(get(ax, 'XTickLabel'));
        legendHandle = legend(ax);
        legendLabels = {};
        if ~isempty(legendHandle) && isgraphics(legendHandle)
            legendLabels = cellstr(legendHandle.String);
        end
        switch titleStr
            case "GLM Coefficients Across Neurons\nwith 95% CIs (* p<0.05)"
                payload.summary_plotSummary_coeff_title = titleStr;
                payload.summary_plotSummary_coeff_ylabel = ylabelStr;
                payload.summary_plotSummary_coeff_xticklabels = xtickLabels;
                payload.summary_plotSummary_coeff_legend = legendLabels;
            case "KS Statistics Across Neurons"
                payload.summary_plotSummary_ks_title = titleStr;
                payload.summary_plotSummary_ks_ylabel = ylabelStr;
                payload.summary_plotSummary_ks_xticklabels = xtickLabels;
            case "Change in AIC Across Neurons"
                payload.summary_plotSummary_aic_title = titleStr;
                payload.summary_plotSummary_aic_ylabel = ylabelStr;
                payload.summary_plotSummary_aic_xticklabels = xtickLabels;
            case "Change in BIC Across Neurons"
                payload.summary_plotSummary_bic_title = titleStr;
                payload.summary_plotSummary_bic_ylabel = ylabelStr;
                payload.summary_plotSummary_bic_xticklabels = xtickLabels;
        end
    end
    payload.summary_plotSummary_num_axes = numel(allAxes);
catch
end
if ~isempty(plotHandle) && isgraphics(plotHandle)
    close(plotHandle);
end

plotAllCoeffsHandle = [];
try
    plotAllCoeffsHandle = figure('Visible','off');
    summary.plotAllCoeffs();
    plotAllCoeffsAx = gca;
    payload.summary_plotAllCoeffs_ylabel = stringify_text(get(get(plotAllCoeffsAx, 'YLabel'), 'String'));
    payload.summary_plotAllCoeffs_xticklabels = cellstr(get(plotAllCoeffsAx, 'XTickLabel'));
    plotAllCoeffsLegend = legend(plotAllCoeffsAx);
    payload.summary_plotAllCoeffs_legend = {};
    if ~isempty(plotAllCoeffsLegend) && isgraphics(plotAllCoeffsLegend)
        payload.summary_plotAllCoeffs_legend = cellstr(plotAllCoeffsLegend.String);
    end
catch
end
if ~isempty(plotAllCoeffsHandle) && isgraphics(plotAllCoeffsHandle)
    close(plotAllCoeffsHandle);
end

coeffOnlyHandle = [];
try
    coeffOnlyHandle = figure('Visible','off');
    coeffOnlyAx = local_axes_handle(summary.plotCoeffsWithoutHistory(2, 0, 1));
    payload.summary_plotCoeffsWithoutHistory_title = stringify_text(get(get(coeffOnlyAx, 'Title'), 'String'));
    payload.summary_plotCoeffsWithoutHistory_ylabel = stringify_text(get(get(coeffOnlyAx, 'YLabel'), 'String'));
    payload.summary_plotCoeffsWithoutHistory_xticklabels = cellstr(get(coeffOnlyAx, 'XTickLabel'));
catch
end
if ~isempty(coeffOnlyHandle) && isgraphics(coeffOnlyHandle)
    close(coeffOnlyHandle);
end

histHandle = [];
try
    histHandle = figure('Visible','off');
    histAx = local_axes_handle(summary.plotHistCoeffs(2, 0, 1));
    payload.summary_plotHistCoeffs_title = stringify_text(get(get(histAx, 'Title'), 'String'));
    payload.summary_plotHistCoeffs_ylabel = stringify_text(get(get(histAx, 'YLabel'), 'String'));
    payload.summary_plotHistCoeffs_xticklabels = cellstr(get(histAx, 'XTickLabel'));
catch
end
if ~isempty(histHandle) && isgraphics(histHandle)
    close(histHandle);
end

summary.plotIC;
icHandle = gcf;
icAxes = findall(icHandle, 'Type', 'axes');
payload.summary_plotIC_num_axes = numel(icAxes);
for idx = 1:length(icAxes)
    ax = icAxes(idx);
    titleStr = stringify_text(get(get(ax, 'Title'), 'String'));
    ylabelStr = stringify_text(get(get(ax, 'YLabel'), 'String'));
    xtickLabels = cellstr(get(ax, 'XTickLabel'));
    switch titleStr
        case "AIC Across Neurons"
            payload.summary_plotIC_aic_title = titleStr;
            payload.summary_plotIC_aic_ylabel = ylabelStr;
            payload.summary_plotIC_aic_xticklabels = xtickLabels;
        case "BIC Across Neurons"
            payload.summary_plotIC_bic_title = titleStr;
            payload.summary_plotIC_bic_ylabel = ylabelStr;
            payload.summary_plotIC_bic_xticklabels = xtickLabels;
        case "log likelihood Across Neurons"
            payload.summary_plotIC_logll_title = titleStr;
            payload.summary_plotIC_logll_ylabel = ylabelStr;
            payload.summary_plotIC_logll_xticklabels = xtickLabels;
    end
end
close(icHandle);

plotAICHandle = figure('Visible','off');
summary.plotAIC();
plotAICAx = gca;
payload.summary_plotAIC_title = stringify_text(get(get(plotAICAx, 'Title'), 'String'));
payload.summary_plotAIC_ylabel = stringify_text(get(get(plotAICAx, 'YLabel'), 'String'));
payload.summary_plotAIC_xticklabels = cellstr(get(plotAICAx, 'XTickLabel'));
close(plotAICHandle);

plotBICHandle = figure('Visible','off');
summary.plotBIC();
plotBICAx = gca;
payload.summary_plotBIC_title = stringify_text(get(get(plotBICAx, 'Title'), 'String'));
payload.summary_plotBIC_ylabel = stringify_text(get(get(plotBICAx, 'YLabel'), 'String'));
payload.summary_plotBIC_xticklabels = cellstr(get(plotBICAx, 'XTickLabel'));
close(plotBICHandle);

plotlogLLHandle = figure('Visible','off');
summary.plotlogLL();
plotlogLLAx = gca;
payload.summary_plotlogLL_title = stringify_text(get(get(plotlogLLAx, 'Title'), 'String'));
payload.summary_plotlogLL_ylabel = stringify_text(get(get(plotlogLLAx, 'YLabel'), 'String'));
payload.summary_plotlogLL_xticklabels = cellstr(get(plotlogLLAx, 'XTickLabel'));
close(plotlogLLHandle);

residualHandle = summary.plotResidualSummary;
residualAxes = findall(residualHandle, 'Type', 'axes');
payload.summary_plotResidual_num_axes = numel(residualAxes);
payload.summary_plotResidual_titles = cell(1, numel(residualAxes));
payload.summary_plotResidual_ylabels = cell(1, numel(residualAxes));
payload.summary_plotResidual_xlabels = cell(1, numel(residualAxes));
payload.summary_plotResidual_line_counts = zeros(1, numel(residualAxes));
payload.summary_plotResidual_legend_labels = {};
for idx = 1:length(residualAxes)
    ax = residualAxes(idx);
    payload.summary_plotResidual_titles{idx} = stringify_text(get(get(ax, 'Title'), 'String'));
    payload.summary_plotResidual_ylabels{idx} = stringify_text(get(get(ax, 'YLabel'), 'String'));
    payload.summary_plotResidual_xlabels{idx} = stringify_text(get(get(ax, 'XLabel'), 'String'));
    payload.summary_plotResidual_line_counts(idx) = numel(findall(ax, 'Type', 'line'));
end
legendHandle = findobj(residualHandle, 'Type', 'legend');
if ~isempty(legendHandle)
    payload.summary_plotResidual_legend_labels = cellstr(legendHandle(1).String);
end
close(residualHandle);

save(fullfile(fixtureRoot, 'analysis_multineuron_exactness.mat'), '-struct', 'payload');
end

function export_ksdiscrete_fixture(fixtureRoot)
t = (0:0.1:1.0)';
stimData = sin(2*pi*t);
stim = Covariate(t, stimData, 'Stimulus', 'time', 's', '', {'stim'});
spikeTrain = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
cfg.setName('stim');
fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}));

pk = fit.lambda.data(:,1) .* (1 / fit.lambda.sampleRate);
pk = min(max(pk, 1e-10), 1);
spikeSignal = spikeTrain.getSigRep.data(:,1);
uniformDraws = [0.25; 0.75];
[Z, ~, xAxis, ~, ~] = ksdiscrete_with_draws(pk, spikeSignal, 'spiketrain', uniformDraws);
U = 1 - exp(-Z);
KSSorted = sort(U, 'ascend');
[differentDists, pValue, ksStat] = kstest2(xAxis, KSSorted);

payload = struct();
payload.time = t;
payload.stim_data = stimData;
payload.spike_times = spikeTrain.spikeTimes;
payload.lambda_time = fit.lambda.time;
payload.lambda_data = fit.lambda.data(:,1);
payload.sample_rate = fit.lambda.sampleRate;
payload.pk = pk;
payload.spike_signal = spikeSignal;
payload.uniform_draws = uniformDraws;
payload.Z = Z;
payload.U = U;
payload.xAxis = xAxis;
payload.KSSorted = KSSorted;
payload.compute_ks_stat = max(abs(KSSorted - xAxis));
payload.ks_stat = ksStat;
payload.ks_pvalue = pValue;
payload.ks_within_conf_int = ~differentDists;

save(fullfile(fixtureRoot, 'ksdiscrete_exactness.mat'), '-struct', 'payload');
end

function export_fit_summary_fixture(fixtureRoot)
t = (0:0.1:1.0)';
lambdaData = [2.0 3.0; 2.5 3.5; 3.0 4.0; 3.5 4.5; 4.0 5.0; 4.5 5.5; 5.0 6.0; 5.5 6.5; 6.0 7.0; 6.5 7.5; 7.0 8.0];
lambda = Covariate(t, lambdaData, '\lambda(t)', 'time', 's', 'Hz', {'stim','stim_hist'});
st1 = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
st2 = nspikeTrain([0.2 0.5 0.8], '2', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
stimCfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
stimCfg.setName('stim');
stimHistCfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [0 0.1 0.2], []);
stimHistCfg.setName('stim_hist');
configColl = ConfigColl({stimCfg, stimHistCfg});
covLabels = {{'stim'}, {'stim','hist1','hist2'}};
numHist = [0 2];
histObjects = {[], []};
ensHistObj = {[], []};
b1 = {[0.5], [0.3 -0.1 -0.05]};
b2 = {[0.4], [0.25 -0.08 -0.02]};
stats1 = {struct('se', [0.05], 'p', [0.01]), struct('se', [0.04 0.03 0.02], 'p', [0.02 0.04 0.06])};
stats2 = {struct('se', [0.06], 'p', [0.03]), struct('se', [0.05 0.04 0.03], 'p', [0.01 0.03 0.07])};
fit1 = FitResult(st1, covLabels, numHist, histObjects, ensHistObj, lambda, b1, [1.0 2.0], stats1, [11.0 7.0], [12.0 8.0], [3.0 5.0], configColl, {}, {}, 'poisson');
fit2 = FitResult(st2, covLabels, numHist, histObjects, ensHistObj, lambda, b2, [1.5 2.5], stats2, [13.0 9.0], [14.0 10.0], [2.0 4.0], configColl, {}, {}, 'poisson');
fixtureZ = [0.2 0.25; 0.4 0.35; 0.6 0.45];
fixtureU = [0.15 0.20; 0.45 0.50; 0.75 0.80];
fixtureXAxis = [0.25 0.25; 0.50 0.50; 0.75 0.75];
fixtureKSSorted = [0.20 0.20; 0.50 0.50; 0.80 0.80];
fixtureX = [-1.04 -0.84; -0.13 0.00; 0.67 0.84];
rhoSig = SignalObj((1:3)', [0.1 0.2; 0.05 0.1; 0.0 0.05], 'rhoSig', 'lag', '', '', {'stim','stim_hist'});
confBoundSig = SignalObj((1:3)', [0.2; 0.1; 0.05], 'confBoundSig', 'lag', '', '', {''});
fit1.setKSStats(fixtureZ, fixtureU, fixtureXAxis, fixtureKSSorted, [0.25 0.50]);
fit2.setKSStats(fixtureZ, fixtureU, fixtureXAxis, fixtureKSSorted, [0.35 0.55]);
fit1.setInvGausStats(fixtureX, rhoSig, confBoundSig);
fit2.setInvGausStats(fixtureX, rhoSig, confBoundSig);
fit1.KSStats.ks_stat = [0.25 0.50];
fit1.KSStats.pValue = [0.90 0.40];
fit1.KSStats.withinConfInt = [1 1];
fit2.KSStats.ks_stat = [0.35 0.55];
fit2.KSStats.pValue = [0.80 0.30];
fit2.KSStats.withinConfInt = [1 0];
Analysis.plotFitResidual(fit1, 0.01, 0);
Analysis.plotFitResidual(fit2, 0.01, 0);
summary = FitResSummary({fit1, fit2});
dAIC = summary.getDiffAIC(1, 0);
dBIC = summary.getDiffBIC(1, 0);
dlogLL = summary.getDifflogLL(1, 0);

payload = struct();
payload.fitNames = summary.fitNames;
payload.neuronNumbers = summary.neuronNumbers;
payload.AIC = summary.AIC;
payload.BIC = summary.BIC;
payload.logLL = summary.logLL;
payload.KSStats = summary.KSStats;
payload.KSPvalues = summary.KSPvalues;
payload.withinConfInt = summary.withinConfInt;
payload.diffAIC = dAIC;
payload.diffBIC = dBIC;
payload.difflogLL = dlogLL;
payload.structure = summary.toStructure;
plotHandle = summary.plotSummary;
allAxes = findall(plotHandle, 'Type', 'axes');
for idx = 1:length(allAxes)
    ax = allAxes(idx);
    titleStr = stringify_text(get(get(ax, 'Title'), 'String'));
    ylabelStr = stringify_text(get(get(ax, 'YLabel'), 'String'));
    xtickLabels = cellstr(get(ax, 'XTickLabel'));
    legendHandle = legend(ax);
    legendLabels = {};
    if ~isempty(legendHandle) && isgraphics(legendHandle)
        legendLabels = cellstr(legendHandle.String);
    end
    switch titleStr
        case "GLM Coefficients Across Neurons\nwith 95% CIs (* p<0.05)"
            payload.plotSummary_coeff_title = titleStr;
            payload.plotSummary_coeff_ylabel = ylabelStr;
            payload.plotSummary_coeff_xticklabels = xtickLabels;
            payload.plotSummary_coeff_legend = legendLabels;
        case "KS Statistics Across Neurons"
            payload.plotSummary_ks_title = titleStr;
            payload.plotSummary_ks_ylabel = ylabelStr;
            payload.plotSummary_ks_xticklabels = xtickLabels;
        case "Change in AIC Across Neurons"
            payload.plotSummary_aic_title = titleStr;
            payload.plotSummary_aic_ylabel = ylabelStr;
            payload.plotSummary_aic_xticklabels = xtickLabels;
        case "Change in BIC Across Neurons"
            payload.plotSummary_bic_title = titleStr;
            payload.plotSummary_bic_ylabel = ylabelStr;
            payload.plotSummary_bic_xticklabels = xtickLabels;
    end
end
payload.plotSummary_num_axes = numel(allAxes);
close(plotHandle);

plotAllCoeffsHandle = figure('Visible','off');
summary.plotAllCoeffs();
plotAllCoeffsAx = gca;
payload.plotAllCoeffs_ylabel = stringify_text(get(get(plotAllCoeffsAx, 'YLabel'), 'String'));
payload.plotAllCoeffs_xticklabels = cellstr(get(plotAllCoeffsAx, 'XTickLabel'));
plotAllCoeffsLegend = legend(plotAllCoeffsAx);
payload.plotAllCoeffs_legend = {};
if ~isempty(plotAllCoeffsLegend) && isgraphics(plotAllCoeffsLegend)
    payload.plotAllCoeffs_legend = cellstr(plotAllCoeffsLegend.String);
end
close(plotAllCoeffsHandle);

coeffOnlyHandle = figure('Visible','off');
coeffOnlyAx = local_axes_handle(summary.plotCoeffsWithoutHistory(2, 0, 1));
payload.plotCoeffsWithoutHistory_title = stringify_text(get(get(coeffOnlyAx, 'Title'), 'String'));
payload.plotCoeffsWithoutHistory_ylabel = stringify_text(get(get(coeffOnlyAx, 'YLabel'), 'String'));
payload.plotCoeffsWithoutHistory_xticklabels = cellstr(get(coeffOnlyAx, 'XTickLabel'));
close(coeffOnlyHandle);

histHandle = figure('Visible','off');
histAx = local_axes_handle(summary.plotHistCoeffs(2, 0, 1));
payload.plotHistCoeffs_title = stringify_text(get(get(histAx, 'Title'), 'String'));
payload.plotHistCoeffs_ylabel = stringify_text(get(get(histAx, 'YLabel'), 'String'));
payload.plotHistCoeffs_xticklabels = cellstr(get(histAx, 'XTickLabel'));
close(histHandle);

fitPlotHandle = fit1.getSubsetFitResult(1).plotResults;
fitPlotAxes = findall(fitPlotHandle, 'Type', 'axes');
payload.fit_plotResults_num_axes = numel(fitPlotAxes);
payload.fit_plotResults_titles = cell(1, numel(fitPlotAxes));
payload.fit_plotResults_ylabels = cell(1, numel(fitPlotAxes));
payload.fit_plotResults_xlabels = cell(1, numel(fitPlotAxes));
for idx = 1:numel(fitPlotAxes)
    ax = fitPlotAxes(idx);
    payload.fit_plotResults_titles{idx} = stringify_text(get(get(ax, 'Title'), 'String'));
    payload.fit_plotResults_ylabels{idx} = stringify_text(get(get(ax, 'YLabel'), 'String'));
    payload.fit_plotResults_xlabels{idx} = stringify_text(get(get(ax, 'XLabel'), 'String'));
end
close(fitPlotHandle);

singleFit = fit1.getSubsetFitResult(1);

ksHandle = figure('Visible','off');
singleFit.KSPlot;
ksAx = gca;
payload.fit_KSPlot_title = stringify_text(get(get(ksAx, 'Title'), 'String'));
payload.fit_KSPlot_ylabel = stringify_text(get(get(ksAx, 'YLabel'), 'String'));
payload.fit_KSPlot_xlabel = stringify_text(get(get(ksAx, 'XLabel'), 'String'));
payload.fit_KSPlot_num_lines = numel(findall(ksAx, 'Type', 'line'));
close(ksHandle);

invHandle = figure('Visible','off');
singleFit.plotInvGausTrans;
invAx = gca;
payload.fit_plotInvGausTrans_title = stringify_text(get(get(invAx, 'Title'), 'String'));
payload.fit_plotInvGausTrans_ylabel = stringify_text(get(get(invAx, 'YLabel'), 'String'));
payload.fit_plotInvGausTrans_xlabel = stringify_text(get(get(invAx, 'XLabel'), 'String'));
payload.fit_plotInvGausTrans_num_lines = numel(findall(invAx, 'Type', 'line'));
close(invHandle);

seqHandle = figure('Visible','off');
singleFit.plotSeqCorr;
seqAx = gca;
payload.fit_plotSeqCorr_title = stringify_text(get(get(seqAx, 'Title'), 'String'));
payload.fit_plotSeqCorr_ylabel = stringify_text(get(get(seqAx, 'YLabel'), 'String'));
payload.fit_plotSeqCorr_xlabel = stringify_text(get(get(seqAx, 'XLabel'), 'String'));
payload.fit_plotSeqCorr_num_lines = numel(findall(seqAx, 'Type', 'line'));
close(seqHandle);

resHandle = figure('Visible','off');
singleFit.plotResidual;
resAx = gca;
payload.fit_plotResidual_title = stringify_text(get(get(resAx, 'Title'), 'String'));
payload.fit_plotResidual_ylabel = stringify_text(get(get(resAx, 'YLabel'), 'String'));
payload.fit_plotResidual_xlabel = stringify_text(get(get(resAx, 'XLabel'), 'String'));
payload.fit_plotResidual_num_lines = numel(findall(resAx, 'Type', 'line'));
close(resHandle);

coeffHandle = figure('Visible','off');
singleFit.plotCoeffs;
coeffAx = gca;
payload.fit_plotCoeffs_title = stringify_text(get(get(coeffAx, 'Title'), 'String'));
payload.fit_plotCoeffs_ylabel = stringify_text(get(get(coeffAx, 'YLabel'), 'String'));
payload.fit_plotCoeffs_xlabel = stringify_text(get(get(coeffAx, 'XLabel'), 'String'));
payload.fit_plotCoeffs_xticklabels = cellstr(get(coeffAx, 'XTickLabel'));
payload.fit_plotCoeffs_num_lines = numel(findall(coeffAx, 'Type', 'line'));
close(coeffHandle);

historyFit = fit1.getSubsetFitResult(2);

coeffNoHistHandle = figure('Visible','off');
historyFit.plotCoeffsWithoutHistory;
coeffNoHistAx = gca;
payload.fit_plotCoeffsWithoutHistory_title = stringify_text(get(get(coeffNoHistAx, 'Title'), 'String'));
payload.fit_plotCoeffsWithoutHistory_ylabel = stringify_text(get(get(coeffNoHistAx, 'YLabel'), 'String'));
payload.fit_plotCoeffsWithoutHistory_xlabel = stringify_text(get(get(coeffNoHistAx, 'XLabel'), 'String'));
payload.fit_plotCoeffsWithoutHistory_xticklabels = cellstr(get(coeffNoHistAx, 'XTickLabel'));
payload.fit_plotCoeffsWithoutHistory_num_lines = numel(findall(coeffNoHistAx, 'Type', 'line'));
close(coeffNoHistHandle);

histCoeffHandle = figure('Visible','off');
historyFit.plotHistCoeffs;
histCoeffAx = gca;
payload.fit_plotHistCoeffs_title = stringify_text(get(get(histCoeffAx, 'Title'), 'String'));
payload.fit_plotHistCoeffs_ylabel = stringify_text(get(get(histCoeffAx, 'YLabel'), 'String'));
payload.fit_plotHistCoeffs_xlabel = stringify_text(get(get(histCoeffAx, 'XLabel'), 'String'));
payload.fit_plotHistCoeffs_xticklabels = cellstr(get(histCoeffAx, 'XTickLabel'));
payload.fit_plotHistCoeffs_num_lines = numel(findall(histCoeffAx, 'Type', 'line'));
close(histCoeffHandle);

payload.fit_structure = fit1.toStructure;
payload.fit_history_structure = fit1.getSubsetFitResult(2).toStructure;

[fitCoeffIndex1, fitCoeffEpochId1, fitCoeffNumEpochs1] = fit1.getCoeffIndex(1);
[fitHistIndex1, fitHistEpochId1, fitHistNumEpochs1] = fit1.getHistIndex(1);
[fitCoeffIndex2, fitCoeffEpochId2, fitCoeffNumEpochs2] = fit1.getCoeffIndex(2);
[fitHistIndex2, fitHistEpochId2, fitHistNumEpochs2] = fit1.getHistIndex(2);
payload.fitCoeffIndex_1 = fitCoeffIndex1;
payload.fitCoeffEpochId_1 = fitCoeffEpochId1;
payload.fitCoeffNumEpochs_1 = fitCoeffNumEpochs1;
payload.fitHistIndex_1 = fitHistIndex1;
payload.fitHistEpochId_1 = fitHistEpochId1;
payload.fitHistNumEpochs_1 = fitHistNumEpochs1;
payload.fitCoeffIndex_2 = fitCoeffIndex2;
payload.fitCoeffEpochId_2 = fitCoeffEpochId2;
payload.fitCoeffNumEpochs_2 = fitCoeffNumEpochs2;
payload.fitHistIndex_2 = fitHistIndex2;
payload.fitHistEpochId_2 = fitHistEpochId2;
payload.fitHistNumEpochs_2 = fitHistNumEpochs2;
[fitParamCoeff1, fitParamSe1, fitParamSig1] = fit1.getParam({'stim'}, 1);
[fitParamCoeff2, fitParamSe2, fitParamSig2] = fit1.getParam({'stim'}, 2);
payload.fitParamCoeff_1 = fitParamCoeff1;
payload.fitParamSe_1 = fitParamSe1;
payload.fitParamSig_1 = fitParamSig1;
payload.fitParamCoeff_2 = fitParamCoeff2;
payload.fitParamSe_2 = fitParamSe2;
payload.fitParamSig_2 = fitParamSig2;

[summaryPlotParams] = summary.plotParams;
payload.plotParams_xLabels = cellstr(summaryPlotParams.xLabels);
payload.plotParams_bAct = summaryPlotParams.bAct;
payload.plotParams_seAct = summaryPlotParams.seAct;
payload.plotParams_sigIndex = summaryPlotParams.sigIndex;
payload.plotParams_numResultsCoeffPresent = summaryPlotParams.numResultsCoeffPresent;
payload.sigCoeffs_fit1 = summary.getSigCoeffs(1);
[coeffMatFit1, coeffLabelsFit1, coeffSeFit1] = summary.getCoeffs(1);
payload.coeffMat_fit1 = coeffMatFit1;
payload.coeffLabels_fit1 = coeffLabelsFit1;
payload.coeffSe_fit1 = coeffSeFit1;
[coeffMatFit2, coeffLabelsFit2, coeffSeFit2] = summary.getCoeffs(2);
payload.coeffMat_fit2 = coeffMatFit2;
payload.coeffLabels_fit2 = coeffLabelsFit2;
payload.coeffSe_fit2 = coeffSeFit2;
[histCoeffMatFit2, histCoeffLabelsFit2] = summary.getHistCoeffs(2);
payload.histCoeffMat_fit2 = histCoeffMatFit2;
payload.histCoeffLabels_fit2 = histCoeffLabelsFit2;

[coeffIndex, coeffEpochId, coeffNumEpochs] = summary.getCoeffIndex;
[histIndex, histEpochId, histNumEpochs] = summary.getHistIndex;
payload.coeffIndex = coeffIndex;
payload.coeffEpochId = coeffEpochId;
payload.coeffNumEpochs = coeffNumEpochs;
payload.histIndex = histIndex;
payload.histEpochId = histEpochId;
payload.histNumEpochs = histNumEpochs;
[coeffIndexFit2, coeffEpochIdFit2, coeffNumEpochsFit2] = summary.getCoeffIndex(2);
[histIndexFit2, histEpochIdFit2, histNumEpochsFit2] = summary.getHistIndex(2);
payload.coeffIndex_fit2 = coeffIndexFit2;
payload.coeffEpochId_fit2 = coeffEpochIdFit2;
payload.coeffNumEpochs_fit2 = coeffNumEpochsFit2;
payload.histIndex_fit2 = histIndexFit2;
payload.histEpochId_fit2 = histEpochIdFit2;
payload.histNumEpochs_fit2 = histNumEpochsFit2;

[coeffSummaryN, coeffSummaryEdges, coeffSummaryPercentSig] = summary.binCoeffs;
payload.coeffSummary_bins = coeffSummaryN;
payload.coeffSummary_edges = coeffSummaryEdges;
payload.coeffSummary_percentSig = coeffSummaryPercentSig;

coeff2dHandle = figure('Visible','off');
coeff2dPlotHandles = summary.plot2dCoeffSummary(gca);
coeff2dAx = gca;
if isempty(summary.plotParams)
    summary.computePlotParams;
end
payload.plot2dCoeffSummary_yticklabels = cellstr(summary.plotParams.xLabels);
payload.plot2dCoeffSummary_num_lines = numel(coeff2dPlotHandles);
textHandles = findall(coeff2dAx, 'Type', 'text');
payload.plot2dCoeffSummary_text = cell(1, numel(textHandles));
for idx = 1:numel(textHandles)
    payload.plot2dCoeffSummary_text{idx} = stringify_text(get(textHandles(idx), 'String'));
end
close(coeff2dHandle);

coeff3dHandle = figure('Visible','off');
coeff3dAx = axes('Parent', coeff3dHandle);
coeff3dPlotHandles = summary.plot3dCoeffSummary(coeff3dAx);
if isempty(summary.plotParams)
    summary.computePlotParams;
end
payload.plot3dCoeffSummary_yticklabels = cellstr(summary.plotParams.xLabels);
payload.plot3dCoeffSummary_num_surfaces = numel(coeff3dPlotHandles);
close(coeff3dHandle);

summary.plotIC;
icHandle = gcf;
icAxes = findall(icHandle, 'Type', 'axes');
payload.plotIC_num_axes = numel(icAxes);
for idx = 1:length(icAxes)
    ax = icAxes(idx);
    titleStr = stringify_text(get(get(ax, 'Title'), 'String'));
    ylabelStr = stringify_text(get(get(ax, 'YLabel'), 'String'));
    xtickLabels = cellstr(get(ax, 'XTickLabel'));
    switch titleStr
        case "AIC Across Neurons"
            payload.plotIC_aic_title = titleStr;
            payload.plotIC_aic_ylabel = ylabelStr;
            payload.plotIC_aic_xticklabels = xtickLabels;
        case "BIC Across Neurons"
            payload.plotIC_bic_title = titleStr;
            payload.plotIC_bic_ylabel = ylabelStr;
            payload.plotIC_bic_xticklabels = xtickLabels;
        case "log likelihood Across Neurons"
            payload.plotIC_logll_title = titleStr;
            payload.plotIC_logll_ylabel = ylabelStr;
            payload.plotIC_logll_xticklabels = xtickLabels;
    end
end
close(icHandle);

plotAICHandle = figure('Visible','off');
summary.plotAIC;
plotAICAx = gca;
payload.plotAIC_title = stringify_text(get(get(plotAICAx, 'Title'), 'String'));
payload.plotAIC_ylabel = stringify_text(get(get(plotAICAx, 'YLabel'), 'String'));
payload.plotAIC_xticklabels = cellstr(get(plotAICAx, 'XTickLabel'));
close(plotAICHandle);

plotBICHandle = figure('Visible','off');
summary.plotBIC;
plotBICAx = gca;
payload.plotBIC_title = stringify_text(get(get(plotBICAx, 'Title'), 'String'));
payload.plotBIC_ylabel = stringify_text(get(get(plotBICAx, 'YLabel'), 'String'));
payload.plotBIC_xticklabels = cellstr(get(plotBICAx, 'XTickLabel'));
close(plotBICHandle);

plotlogLLHandle = figure('Visible','off');
summary.plotlogLL;
plotlogLLAx = gca;
payload.plotlogLL_title = stringify_text(get(get(plotlogLLAx, 'Title'), 'String'));
payload.plotlogLL_ylabel = stringify_text(get(get(plotlogLLAx, 'YLabel'), 'String'));
payload.plotlogLL_xticklabels = cellstr(get(plotlogLLAx, 'XTickLabel'));
close(plotlogLLHandle);

residualHandle = summary.plotResidualSummary;
residualAxes = findall(residualHandle, 'Type', 'axes');
payload.plotResidualSummary_num_axes = numel(residualAxes);
payload.plotResidualSummary_titles = cell(1, numel(residualAxes));
payload.plotResidualSummary_ylabels = cell(1, numel(residualAxes));
payload.plotResidualSummary_xlabels = cell(1, numel(residualAxes));
payload.plotResidualSummary_line_counts = zeros(1, numel(residualAxes));
payload.plotResidualSummary_legend_labels = {};
for idx = 1:length(residualAxes)
    ax = residualAxes(idx);
    payload.plotResidualSummary_titles{idx} = stringify_text(get(get(ax, 'Title'), 'String'));
    payload.plotResidualSummary_ylabels{idx} = stringify_text(get(get(ax, 'YLabel'), 'String'));
    payload.plotResidualSummary_xlabels{idx} = stringify_text(get(get(ax, 'XLabel'), 'String'));
    payload.plotResidualSummary_line_counts(idx) = numel(findall(ax, 'Type', 'line'));
end
legendHandle = findobj(residualHandle, 'Type', 'legend');
if ~isempty(legendHandle)
    payload.plotResidualSummary_legend_labels = cellstr(legendHandle(1).String);
end
close(residualHandle);

payload.roundtrip_supported = false;
payload.roundtrip_error = '';
try
    roundtrip = FitResSummary.fromStructure(payload.structure);
    payload.roundtrip_supported = true;
    payload.roundtrip_AIC = roundtrip.AIC;
    payload.roundtrip_BIC = roundtrip.BIC;
    payload.roundtrip_logLL = roundtrip.logLL;
    payload.roundtrip_neuronNumbers = roundtrip.neuronNumbers;
    payload.roundtrip_fitNames = roundtrip.fitNames;
catch err
    payload.roundtrip_error = err.message;
end

save(fullfile(fixtureRoot, 'fit_summary_exactness.mat'), '-struct', 'payload');
end

function export_point_process_fixture(fixtureRoot)
rng(5);
Ts = .001;
t = (0:Ts:50)';
mu = -3;
H = tf([-1 -2 -4], [1], Ts, 'Variable', 'z^-1');
S = tf([1], 1, Ts, 'Variable', 'z^-1');
E = tf([0], 1, Ts, 'Variable', 'z^-1');
stim = Covariate(t, sin(2*pi*1*t), 'Stimulus', 'time', 's', 'Voltage', {'sin'});
ens = Covariate(t, zeros(length(t), 1), 'Ensemble', 'time', 's', 'Spikes', {'n1'});
[spikeColl, lambda] = CIF.simulateCIF(mu, H, S, E, stim, ens, 5, 'binomial');

spikeCounts = zeros(1, spikeColl.numSpikeTrains);
for i = 1:spikeColl.numSpikeTrains
    spikeCounts(i) = length(spikeColl.getNST(i).spikeTimes);
end

payload = struct();
payload.seed = 5;
payload.lambda_head = lambda.data(1:8, 1);
payload.spike_counts = spikeCounts;

detTime = (0:Ts:0.01)';
detStim = sin(2*pi*1*detTime);
detUniforms = [0.99; 0.01; 0.99; 0.2; 0.8; 0.05; 0.95; 0.15; 0.6; 0.4; 0.3];
[hNum, ~] = tfdata(H, 'v');
[sNum, ~] = tfdata(S, 'v');
[eNum, ~] = tfdata(E, 'v');
[detRate, detDelta, detHist, detEta, detSpikes] = local_discrete_point_process(mu, hNum, sNum, eNum, detStim, zeros(size(detStim)), detUniforms, Ts, 0);
payload.det_time = detTime;
payload.det_uniforms = detUniforms;
payload.det_stimulus = detStim;
payload.det_rate_hz = detRate;
payload.det_lambda_delta = detDelta;
payload.det_history_effect = detHist;
payload.det_eta = detEta;
payload.det_spike_indicator = detSpikes;

save(fullfile(fixtureRoot, 'point_process_exactness.mat'), '-struct', 'payload');
end

function export_thinning_fixture(fixtureRoot)
t = (0:0.1:1.0)';
lambdaData = [2.0; 4.0; 3.5; 5.0; 1.5; 2.5; 4.5; 3.0; 2.0; 1.0; 0.5];
lambdaCov = Covariate(t, lambdaData, '\lambda(t)', 'time', 's', 'Hz', {'\lambda'});
arrivalUniforms = [0.95; 0.5; 0.2; 0.8; 0.3; 0.7; 0.4; 0.6];
thinningUniforms = [0.1; 0.9; 0.2; 0.7; 0.3; 0.6; 0.4; 0.5];
maxTimeRes = 0.05;
[proposalCount, lambdaBound, interarrival, candidateTimes, lambdaRatio, thinningUsed, acceptedTimes, roundedTimes] = ...
    local_thin_from_lambda(lambdaCov, arrivalUniforms, thinningUniforms, maxTimeRes);

payload = struct();
payload.time = t;
payload.lambda_data = lambdaData;
payload.lambda_bound = lambdaBound;
payload.proposal_count = proposalCount;
payload.arrival_uniforms = arrivalUniforms(1:proposalCount);
payload.interarrival_times = interarrival;
payload.candidate_spike_times = candidateTimes;
payload.lambda_ratio = lambdaRatio;
payload.thinning_uniforms = thinningUsed;
payload.accepted_spike_times = acceptedTimes;
payload.rounded_spike_times = roundedTimes;
payload.maxTimeRes = maxTimeRes;

save(fullfile(fixtureRoot, 'thinning_exactness.mat'), '-struct', 'payload');
end

function export_decoding_predict_fixture(fixtureRoot)
x_u = [0.1; -0.2];
W_u = [1.0 0.1; 0.1 2.0];
A = [1.0 0.2; 0.0 0.9];
Q = 0.05 * eye(2);
[x_p, W_p] = DecodingAlgorithms.PPDecode_predict(x_u, W_u, A, Q);

payload = struct();
payload.x_u = x_u;
payload.W_u = W_u;
payload.A = A;
payload.Q = Q;
payload.x_p = x_p;
payload.W_p = W_p;

save(fullfile(fixtureRoot, 'decoding_predict_exactness.mat'), '-struct', 'payload');
end

function export_decoding_smoother_fixture(fixtureRoot)
A = 1.0;
Q = 0.05;
dN = [0 1 0 1];
lags = 2;
mu = -1.0;
beta = 0.75;
fitType = 'binomial';
delta = 0.1;
[x_pLag, W_pLag, x_uLag, W_uLag] = DecodingAlgorithms.PP_fixedIntervalSmoother(A, Q, dN, lags, mu, beta, fitType, delta);

payload = struct();
payload.A = A;
payload.Q = Q;
payload.dN = dN;
payload.lags = lags;
payload.mu = mu;
payload.beta = beta;
payload.fitType = fitType;
payload.delta = delta;
payload.x_pLag = x_pLag;
payload.W_pLag = W_pLag;
payload.x_uLag = x_uLag;
payload.W_uLag = W_uLag;

save(fullfile(fixtureRoot, 'decoding_smoother_exactness.mat'), '-struct', 'payload');
end

function export_hybrid_filter_fixture(fixtureRoot)
A = {1.0, 0.9};
Q = {0.02, 0.05};
p_ij = [0.95 0.05; 0.10 0.90];
Mu0 = [0.6; 0.4];
dN = [0 1 1 0 1];
mu = -1.0;
beta = 0.75;
fitType = 'binomial';
binwidth = 0.1;
[S_est, X, W, MU_u, X_s, W_s, pNGivenS] = DecodingAlgorithms.PPHybridFilterLinear( ...
    A, Q, p_ij, Mu0, dN, mu, beta, fitType, binwidth);

payload = struct();
payload.A1 = A{1};
payload.A2 = A{2};
payload.Q1 = Q{1};
payload.Q2 = Q{2};
payload.p_ij = p_ij;
payload.Mu0 = Mu0;
payload.dN = dN;
payload.mu = mu;
payload.beta = beta;
payload.fitType = fitType;
payload.binwidth = binwidth;
payload.S_est = S_est;
payload.X = X;
payload.W = W;
payload.MU_u = MU_u;
payload.X_s_1 = X_s{1};
payload.X_s_2 = X_s{2};
payload.W_s_1 = W_s{1};
payload.W_s_2 = W_s{2};
payload.pNGivenS = pNGivenS;

save(fullfile(fixtureRoot, 'hybrid_filter_exactness.mat'), '-struct', 'payload');
end

function export_nonlinear_decode_fixture(fixtureRoot)
A = eye(2);
Q = 0.01 * eye(2);
Px0 = 0.05 * eye(2);
delta = 0.1;
dN = [0 1 0 1;
      1 0 1 0];
lambdaCIF = {
    build_polynomial_binomial_cif([-2.0 -0.5 0.3 -0.2 -0.1 0.05]), ...
    build_polynomial_binomial_cif([-1.5 0.4 -0.2 0.15 -0.05 0.02])
};
[x_p, W_p, x_u, W_u] = DecodingAlgorithms.PPDecodeFilter(A, Q, Px0, dN, lambdaCIF, delta);

payload = struct();
payload.A = A;
payload.Q = Q;
payload.Px0 = Px0;
payload.delta = delta;
payload.dN = dN;
payload.beta1 = [-2.0 -0.5 0.3 -0.2 -0.1 0.05];
payload.beta2 = [-1.5 0.4 -0.2 0.15 -0.05 0.02];
payload.x_p = x_p;
payload.W_p = W_p;
payload.x_u = x_u;
payload.W_u = W_u;

save(fullfile(fixtureRoot, 'nonlinear_decode_exactness.mat'), '-struct', 'payload');
end

function export_simulated_network_fixture(fixtureRoot)
rng(4);
Ts = .001;
t = (0:Ts:50)';
mu{1} = -3; mu{2} = -3; %#ok<AGROW>
H{1} = tf([-4 -2 -1], [1], Ts, 'Variable', 'z^-1'); %#ok<AGROW>
H{2} = tf([-4 -2 -1], [1], Ts, 'Variable', 'z^-1'); %#ok<AGROW>
S{1} = tf([1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
S{2} = tf([-1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
E{1} = tf([1], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
E{2} = tf([-4], 1, Ts, 'Variable', 'z^-1'); %#ok<AGROW>
stim = Covariate(t, sin(2*pi*1*t), 'Stimulus', 'time', 's', 'Voltage', {'sin'});
assignin('base', 'S1', S{1}); assignin('base', 'H1', H{1}); assignin('base', 'E1', E{1}); assignin('base', 'mu1', mu{1});
assignin('base', 'S2', S{2}); assignin('base', 'H2', H{2}); assignin('base', 'E2', E{2}); assignin('base', 'mu2', mu{2});
options = simget;
[~,~,yout] = sim('SimulatedNetwork2', [stim.minTime stim.maxTime], options, stim.dataToStructure);
[h1Num, ~] = tfdata(H{1}, 'v');
[h2Num, ~] = tfdata(H{2}, 'v');
[s1Num, ~] = tfdata(S{1}, 'v');
[s2Num, ~] = tfdata(S{2}, 'v');
[e1Num, ~] = tfdata(E{1}, 'v');
[e2Num, ~] = tfdata(E{2}, 'v');
stateMat = yout(:,1:2);
probMat = zeros(size(stateMat));
for n = 1:size(stateMat, 1)
    hist1 = 0; hist2 = 0;
    for lag = 1:length(h1Num)
        if n-lag >= 1
            hist1 = hist1 + h1Num(lag) * stateMat(n-lag,1);
            hist2 = hist2 + h2Num(lag) * stateMat(n-lag,2);
        end
    end
    ens1 = 0; ens2 = 0;
    if n > 1
        ens1 = e1Num(1) * stateMat(n-1,2);
        ens2 = e2Num(1) * stateMat(n-1,1);
    end
    eta1 = mu{1} + hist1 + s1Num(1) * stim.data(n) + ens1;
    eta2 = mu{2} + hist2 + s2Num(1) * stim.data(n) + ens2;
    probMat(n,1) = exp(eta1) / (1 + exp(eta1));
    probMat(n,2) = exp(eta2) / (1 + exp(eta2));
end

payload = struct();
payload.actual_network = [0 1; -4 0];
payload.prob_head = probMat(1:5,:);
payload.state_head = yout(1:5,1:2);
payload.spike_counts = [sum(yout(:,1) > .5), sum(yout(:,2) > .5)];

detTime = (0:Ts:0.009)';
detStim = sin(2*pi*1*detTime);
detUniforms = [
    0.99 0.99;
    0.01 0.99;
    0.99 0.02;
    0.20 0.80;
    0.60 0.10;
    0.05 0.95;
    0.90 0.40;
    0.15 0.30;
    0.70 0.20;
    0.25 0.85
];
[stateDet, probDet, etaDet, histDet, ensDet] = local_two_neuron_network(mu{1}, mu{2}, h1Num, h2Num, s1Num, s2Num, e1Num, e2Num, detStim, detUniforms);
payload.det_time = detTime;
payload.det_uniforms = detUniforms;
payload.det_stimulus = detStim;
payload.det_probability = probDet;
payload.det_state = stateDet;
payload.det_eta = etaDet;
payload.det_history_effect = histDet;
payload.det_ensemble_effect = ensDet;

save(fullfile(fixtureRoot, 'simulated_network_exactness.mat'), '-struct', 'payload');
end

function [rateHz, lambdaDelta, histEffect, eta, spikes] = local_discrete_point_process(mu, hNum, sNum, eNum, stimInput, ensInput, uniforms, Ts, simTypeSelect)
stimInput = stimInput(:);
ensInput = ensInput(:);
uniforms = uniforms(:);
n = length(stimInput);
rateHz = zeros(n,1);
lambdaDelta = zeros(n,1);
histEffect = zeros(n,1);
eta = zeros(n,1);
spikes = zeros(n,1);
for idx = 1:n
    stimEffect = 0;
    for lag = 1:length(sNum)
        if idx-lag+1 >= 1
            stimEffect = stimEffect + sNum(lag) * stimInput(idx-lag+1);
        end
    end
    ensEffect = 0;
    for lag = 1:length(eNum)
        if idx-lag+1 >= 1
            ensEffect = ensEffect + eNum(lag) * ensInput(idx-lag+1);
        end
    end
    histVal = 0;
    for lag = 1:length(hNum)
        if idx-lag >= 1
            histVal = histVal + hNum(lag) * spikes(idx-lag);
        end
    end
    histEffect(idx) = histVal;
    eta(idx) = mu + stimEffect + ensEffect + histVal;
    if simTypeSelect == 0
        lambdaDelta(idx) = exp(eta(idx)) / (1 + exp(eta(idx)));
        rateHz(idx) = lambdaDelta(idx) / Ts;
        spikes(idx) = uniforms(idx) < lambdaDelta(idx);
    else
        rateHz(idx) = exp(eta(idx));
        lambdaDelta(idx) = 1 - exp(-rateHz(idx) * Ts);
        spikes(idx) = uniforms(idx) < lambdaDelta(idx);
    end
end
end

function [proposalCount, lambdaBound, interarrival, candidateTimes, lambdaRatio, thinningUsed, acceptedTimes, roundedTimes] = local_thin_from_lambda(lambdaCov, arrivalUniforms, thinningUniforms, maxTimeRes)
lambdaBound = max(lambdaCov);
Tmax = lambdaCov.maxTime;
proposalCount = ceil(lambdaBound * (1.5 * Tmax));
arrivalUniforms = arrivalUniforms(:);
thinningUniforms = thinningUniforms(:);
u = arrivalUniforms(1:proposalCount);
interarrival = -log(u) ./ lambdaBound;
candidateTimes = cumsum(interarrival);
candidateTimes = candidateTimes(candidateTimes <= Tmax);
if isempty(candidateTimes)
    lambdaRatio = [];
    thinningUsed = [];
    acceptedTimes = [];
else
    lambdaRatio = lambdaCov.getValueAt(candidateTimes) ./ lambdaBound;
    thinningUsed = thinningUniforms(1:length(lambdaRatio));
    acceptedTimes = candidateTimes(lambdaRatio >= thinningUsed);
end
if isempty(maxTimeRes)
    roundedTimes = acceptedTimes;
else
    roundedTimes = unique(ceil(acceptedTimes ./ maxTimeRes) * maxTimeRes);
end
end

function [stateMat, probMat, etaMat, histMat, ensMat] = local_two_neuron_network(mu1, mu2, h1Num, h2Num, s1Num, s2Num, e1Num, e2Num, stimInput, uniforms)
stimInput = stimInput(:);
uniforms = double(uniforms);
n = length(stimInput);
stateMat = zeros(n,2);
probMat = zeros(n,2);
etaMat = zeros(n,2);
histMat = zeros(n,2);
ensMat = zeros(n,2);
for idx = 1:n
    hist1 = 0;
    hist2 = 0;
    for lag = 1:length(h1Num)
        if idx-lag >= 1
            hist1 = hist1 + h1Num(lag) * stateMat(idx-lag,1);
        end
    end
    for lag = 1:length(h2Num)
        if idx-lag >= 1
            hist2 = hist2 + h2Num(lag) * stateMat(idx-lag,2);
        end
    end
    stim1 = s1Num(1) * stimInput(idx);
    stim2 = s2Num(1) * stimInput(idx);
    ens1 = 0;
    ens2 = 0;
    if idx > 1
        ens1 = e1Num(1) * stateMat(idx-1,2);
        ens2 = e2Num(1) * stateMat(idx-1,1);
    end
    histMat(idx,:) = [hist1 hist2];
    ensMat(idx,:) = [ens1 ens2];
    etaMat(idx,:) = [mu1 + hist1 + stim1 + ens1, mu2 + hist2 + stim2 + ens2];
    probMat(idx,:) = exp(etaMat(idx,:)) ./ (1 + exp(etaMat(idx,:)));
    stateMat(idx,1) = uniforms(idx,1) < probMat(idx,1);
    stateMat(idx,2) = uniforms(idx,2) < probMat(idx,2);
end
end

function [rst,varargout] = ksdiscrete_with_draws(pk,st,spikeflag,draws)
[m1,m2]=size(pk);
if (m1 ~=1 && m2 ~=1); error('pk must be a vector'); end
if (m2>m1); pk=pk'; end
[m1,~]=size(pk);
if any(pk<0) || any(pk>1); error('all values for pk must be within [0,1]'); end

if strcmp(spikeflag,'spiketrain')
    [n1,n2]=size(st);
    if (n1 ~=1 && n2 ~=1); error('spike train must be a vector'); end
    if (n2>n1); st=st'; end
    if m1 ~= n1; error('pk and spike train must be same length'); end
    spikeindicies=find(st==1);
    Nspikes=length(spikeindicies);
elseif strcmp(spikeflag,'spikeind')
    [n1,n2]=size(st);
    if (n1 ~=1 && n2 ~=1); error('spike indicies must be a vector'); end
    if (n2>n1); st=st'; end
    spikeindicies=unique(st);
    Nspikes=length(spikeindicies);
else
    error('invalid spikeflag');
end

if isempty(spikeindicies)
    rst = pk;
    return;
end
if spikeindicies(1)<1; error('There is at least one spike with index less than 0'); end
if spikeindicies(Nspikes)>length(pk); error('There is at least one spike with a index greater than the length of pk'); end

qk=-log(1-pk);
rst=zeros(Nspikes-1,1);
rstold=zeros(Nspikes-1,1);
for r=1:Nspikes-1
    ind1=spikeindicies(r);
    ind2=spikeindicies(r+1);
    total=sum(qk(ind1+1:ind2-1));
    delta=-(1/qk(ind2))*log(1-draws(r)*(1-exp(-qk(ind2))));
    if(delta~=0)
        total=total+qk(ind2)*delta;
    end
    rst(r)=total;
    rstold(r)=sum(qk(ind1+1:ind2));
end

rstsort=sort(rst);
varargout{1}=rstsort;
inrst=1/(Nspikes-1);
xrst=(0.5*inrst:inrst:1-0.5*inrst)';
varargout{2}=xrst;
cb=1.36*sqrt(inrst);
varargout{3}=cb;
varargout{4}=sort(rstold);
end

function cifObj = build_polynomial_binomial_cif(beta)
beta = beta(:)';
x = sym('x', 'real');
y = sym('y', 'real');
cifObj = CIF(beta(1:3), {'1', 'x', 'y'}, {'x', 'y'}, 'binomial');
cifObj.b = beta;
cifObj.varIn = [sym(1); x; y; x^2; y^2; x * y];
cifObj.stimVars = [x; y];
cifObj.fitType = 'binomial';
cifObj.history = [];
cifObj.histCoeffs = [];
cifObj.histVars = {};
cifObj.histCoeffVars = {};
cifObj.spikeTrain = [];
cifObj.historyMat = [];
cifObj.lambdaDelta = simplify(exp(beta * cifObj.varIn) ./ (1 + exp(beta * cifObj.varIn)));
cifObj.lambdaDeltaFunction = matlabFunction(cifObj.lambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.gradientLambdaDelta = simplify(jacobian(cifObj.lambdaDelta, cifObj.stimVars));
cifObj.gradientLogLambdaDelta = simplify(jacobian(log(cifObj.lambdaDelta), cifObj.stimVars));
cifObj.gradientFunction = matlabFunction(cifObj.gradientLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.gradientLogFunction = matlabFunction(cifObj.gradientLogLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.jacobianLambdaDelta = simplify(jacobian(cifObj.gradientLambdaDelta, cifObj.stimVars));
cifObj.jacobianFunction = matlabFunction(cifObj.jacobianLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.jacobianLogLambdaDelta = simplify(jacobian(cifObj.gradientLogLambdaDelta, cifObj.stimVars));
cifObj.jacobianLogFunction = matlabFunction(cifObj.jacobianLogLambdaDelta, 'vars', symvar(cifObj.varIn));
cifObj.lambdaDeltaGamma = [];
cifObj.LogLambdaDeltaGamma = [];
cifObj.gradientLambdaDeltaGamma = [];
cifObj.gradientLogLambdaDeltaGamma = [];
cifObj.jacobianLambdaDeltaGamma = [];
cifObj.jacobianLogLambdaDeltaGamma = [];
cifObj.lambdaDeltaGammaFunction = [];
cifObj.LogLambdaDeltaGammaFunction = [];
cifObj.gradientFunctionGamma = [];
cifObj.gradientLogFunctionGamma = [];
cifObj.jacobianFunctionGamma = [];
cifObj.jacobianLogFunctionGamma = [];
cifObj.indepVars = symvar(cifObj.lambdaDelta);

vars = symvar(cifObj.varIn);
if length(vars) == 1
    cifObj.argstr = 'val';
else
    argstr = 'val(1)';
    for i = 2:length(vars)
        argstr = strcat(argstr, [',val(' num2str(i) ')']);
    end
    cifObj.argstr = argstr;
end
cifObj.argstrLDGamma = '';
end

function out = stringify_text(value)
if isstring(value)
    out = char(strjoin(cellstr(value), newline));
elseif ischar(value)
    out = value;
elseif iscell(value)
    parts = cellfun(@stringify_text, value, 'UniformOutput', false);
    out = strjoin(parts, newline);
else
    out = '';
end
end

function ax = local_axes_handle(handleObj)
if iscell(handleObj)
    handleObj = [handleObj{:}];
end
if isa(handleObj, 'matlab.graphics.axis.Axes')
    ax = handleObj;
    if numel(ax) > 1
        ax = ax(1);
    end
    return;
end
ax = ancestor(handleObj, 'axes');
if isempty(ax)
    ax = gca;
elseif numel(ax) > 1
    ax = ax(1);
end
end
