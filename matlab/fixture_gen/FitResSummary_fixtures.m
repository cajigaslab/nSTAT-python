function FitResSummary_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for FitResSummary class parity checks.
%
% MATLAB reference: FitResSummary.m getDiffAIC/getDiffBIC/getDifflogLL/
% getCoeffIndex/getHistIndex/getCoeffs.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'FitResSummary');
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

rng(0, 'twister');

time = (0:1:4)';
X = [
    0.0  0.0;
    1.0  0.0;
    0.0  1.0;
    1.0  1.0;
    2.0 -1.0
];

beta1 = [0.4; -0.2];
beta2 = [0.1; 0.3];

lambda1 = exp(X * beta1) ./ (1 + exp(X * beta1));
lambda2 = exp(X * beta2) ./ (1 + exp(X * beta2));

spikeObj = nspikeTrain([1 3], '1', 1, 0, 4, 'time', 's', '', '', 0);
lambda = Covariate(time, [lambda1 lambda2], '\Lambda(t)', 'time', 's', 'Hz', {'\lambda_1', '\lambda_2'});

covLabels = {{'stim1', 'stim2'}, {'stim1', 'stim2'}};
numHist = [0 0];
histObjects = {[], []};
ensHistObj = {[], []};
b = {beta1, beta2};
dev = [1.5 1.2];
stats = {struct('se', [0.05; 0.08]), struct('se', [0.04; 0.07])};
AIC = [3.2 2.8];
BIC = [3.5 3.1];
logLL = [-1.6 -1.4];

cfg1 = TrialConfig({'stim1', 'stim2'}, 1, [], [], [], [], 'cfg1');
cfg2 = TrialConfig({'stim1', 'stim2'}, 1, [], [], [], [], 'cfg2');
configColl = ConfigColl({cfg1, cfg2});

fitType = {'binomial', 'binomial'};
fitObj = FitResult(spikeObj, covLabels, numHist, histObjects, ensHistObj, ...
    lambda, b, dev, stats, AIC, BIC, logLL, configColl, {X, X}, {time, time}, fitType);
fitObj.computePlotParams();
fitObj.setKSStats([0.1 0.2; 0.2 0.3], [0.1 0.2; 0.3 0.4], [0.1 0.1; 0.9 0.9], [0.1 0.2; 0.8 0.9], [0.2 0.3]);

summaryObj = FitResSummary({fitObj});

diff_aic = summaryObj.getDiffAIC(1, 0);
diff_bic = summaryObj.getDiffBIC(1, 0);
diff_logll = summaryObj.getDifflogLL(1, 0);

[coeff_index, coeff_epoch_id, coeff_num_epochs] = summaryObj.getCoeffIndex(1, 0);
[hist_index, hist_epoch_id, hist_num_epochs] = summaryObj.getHistIndex(1, 0);
[coeff_mat, coeff_labels, coeff_se] = summaryObj.getCoeffs(1);

save(outputFile, ...
    'AIC', ...
    'BIC', ...
    'logLL', ...
    'diff_aic', ...
    'diff_bic', ...
    'diff_logll', ...
    'coeff_index', ...
    'coeff_epoch_id', ...
    'coeff_num_epochs', ...
    'hist_index', ...
    'hist_epoch_id', ...
    'hist_num_epochs', ...
    'coeff_mat', ...
    'coeff_labels', ...
    'coeff_se');

fprintf('Wrote FitResSummary fixtures to %s\n', outputFile);
end
