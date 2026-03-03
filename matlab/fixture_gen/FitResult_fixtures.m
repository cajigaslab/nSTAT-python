function FitResult_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for FitResult class parity checks.
%
% MATLAB reference: FitResult.m constructor/evalLambda/getCoeffIndex/
% getCoeffs/getParam/isValDataPresent/computePlotParams.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'FitResult');
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

beta = [0.4; -0.2];
lin = X * beta;
lambda_data = exp(lin) ./ (1 + exp(lin));

spikeObj = nspikeTrain([1 3], '1', 1, 0, 4, 'time', 's', '', '', 0);
lambda = Covariate(time, lambda_data, '\Lambda(t)', 'time', 's', 'Hz', {'\lambda_1'});

covLabels = {{'stim1', 'stim2'}};
numHist = 0;
histObjects = {[]};
ensHistObj = {[]};
b = {beta};
dev = 1.5;
stats = {struct('se', [0.05; 0.08])};
AIC = 3.2;
BIC = 3.5;
logLL = -1.6;

cfg = TrialConfig({'stim1', 'stim2'}, 1, [], [], [], [], 'cfg1');
configColl = ConfigColl({cfg});
XvalData = {X};
XvalTime = {time};
fitType = {'binomial'};

fitObj = FitResult(spikeObj, covLabels, numHist, histObjects, ensHistObj, ...
    lambda, b, dev, stats, AIC, BIC, logLL, configColl, XvalData, XvalTime, fitType);

fitObj.computePlotParams();

lambda_eval = fitObj.evalLambda(1, X);
coeff_index = fitObj.getCoeffIndex(1, 0);
[coeff_mat, coeff_labels, coeff_se] = fitObj.getCoeffs(1);
[param_vals, param_se, param_sig] = fitObj.getParam({'stim1', 'stim2'}, 1);
is_val_present = fitObj.isValDataPresent();

plot_bAct = fitObj.getPlotParams().bAct;
plot_seAct = fitObj.getPlotParams().seAct;
plot_sigIndex = fitObj.getPlotParams().sigIndex;
plot_xLabels = fitObj.getPlotParams().xLabels;

aic_value = fitObj.AIC(1);
bic_value = fitObj.BIC(1);
logll_value = fitObj.logLL(1);

save(outputFile, ...
    'time', ...
    'X', ...
    'beta', ...
    'lambda_data', ...
    'lambda_eval', ...
    'coeff_index', ...
    'coeff_mat', ...
    'coeff_labels', ...
    'coeff_se', ...
    'param_vals', ...
    'param_se', ...
    'param_sig', ...
    'is_val_present', ...
    'plot_bAct', ...
    'plot_seAct', ...
    'plot_sigIndex', ...
    'plot_xLabels', ...
    'aic_value', ...
    'bic_value', ...
    'logll_value');

fprintf('Wrote FitResult fixtures to %s\n', outputFile);
end
