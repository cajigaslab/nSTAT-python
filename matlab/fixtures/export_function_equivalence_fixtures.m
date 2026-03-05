function export_function_equivalence_fixtures(outRoot)
% Export deterministic MATLAB gold fixtures for function-level parity checks.
% Usage:
%   export_function_equivalence_fixtures('tests/parity/fixtures/matlab_gold/functions')

if nargin < 1 || isempty(outRoot)
    outRoot = fullfile('tests','parity','fixtures','matlab_gold','functions');
end
if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

% Function case: psth_counts
caseDir = fullfile(outRoot, 'psth_counts');
if exist(caseDir, 'dir') ~= 7
    mkdir(caseDir);
end
s1 = nspikeTrain([0.1;0.4]);
s2 = nspikeTrain(0.2);
edges = [0.0 0.25 0.5];
trains = nstColl({s1,s2});
[rate, counts] = trains.computePSTH(edges);
save(fullfile(caseDir, 'case.mat'), 'rate', 'counts');

% Function case: linear_decode_basic (simple least-squares reference)
caseDir = fullfile(outRoot, 'linear_decode_basic');
if exist(caseDir, 'dir') ~= 7
    mkdir(caseDir);
end
X = [0 1; 1 2; 2 3; 3 4];
y = [0;1;2;3];
Xaug = [ones(size(X,1),1), X];
coefficients = Xaug \ y;
decoded = Xaug * coefficients;
save(fullfile(caseDir, 'case.mat'), 'coefficients', 'decoded');

% Function case: kalman_filter_scalar
caseDir = fullfile(outRoot, 'kalman_filter_scalar');
if exist(caseDir, 'dir') ~= 7
    mkdir(caseDir);
end
obs = [1.0; 0.5; 0.2];
a = 1.0; h = 1.0; q = 0.01; r = 0.04;
x = 0.0; p = 1.0;
state = zeros(size(obs));
cov = zeros(size(obs));
for k = 1:numel(obs)
    xPred = a * x;
    pPred = a * p * a + q;
    innovation = obs(k) - h * xPred;
    s = h * pPred * h + r;
    kg = pPred * h / s;
    x = xPred + kg * innovation;
    p = (1 - kg * h) * pPred;
    state(k) = x;
    cov(k) = p;
end
save(fullfile(caseDir, 'case.mat'), 'state', 'cov');

fprintf('Function fixtures exported to %s\n', outRoot);
end
