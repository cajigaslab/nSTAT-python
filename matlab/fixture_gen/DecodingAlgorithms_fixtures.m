function DecodingAlgorithms_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for DecodingAlgorithms parity checks.
%
% MATLAB reference: DecodingAlgorithms.computeSpikeRateCIs.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'DecodingAlgorithms');
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

xK = [0.25 -0.05 0.40];
[numBasis, K] = size(xK);

Wku = zeros(numBasis, numBasis, K, K);
for r = 1:numBasis
    Wku(r, r, :, :) = 1e-12 * eye(K);
end

dN = [
    0 1 0 1 0 0;
    1 0 1 0 0 1;
    0 0 1 1 1 0
];

t0 = 0.0;
delta = 0.2;
tf = (size(dN, 2) - 1) * delta;
fitType = 'binomial';
gamma = [];
windowTimes = [];
Mc = 40;
alphaVal = 0.05;

[spikeRateSig, ProbMat, sigMat] = DecodingAlgorithms.computeSpikeRateCIs( ...
    xK, Wku, dN, t0, tf, fitType, delta, gamma, windowTimes, Mc, alphaVal);

spike_rate_data = spikeRateSig.dataToMatrix;
spike_rate_time = spikeRateSig.time;

save(outputFile, ...
    'xK', ...
    'Wku', ...
    'dN', ...
    't0', ...
    'tf', ...
    'fitType', ...
    'delta', ...
    'gamma', ...
    'windowTimes', ...
    'Mc', ...
    'alphaVal', ...
    'spike_rate_data', ...
    'spike_rate_time', ...
    'ProbMat', ...
    'sigMat');

fprintf('Wrote DecodingAlgorithms fixtures to %s\n', outputFile);
end
