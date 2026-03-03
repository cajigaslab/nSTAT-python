function CIF_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for CIF class parity checks.
%
% MATLAB reference: CIF.m evalLambdaDelta/evalGradient/evalGradientLog/
% evalJacobian/evalJacobianLog/CIFCopy/isSymBeta.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'CIF');
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

beta = [0.4 -0.25];
xnames = {'x1'; 'x2'};
stimNames = {'x1'; 'x2'};
stim_vals = [
    -1.00  0.20;
    -0.25 -0.50;
     0.00  0.00;
     0.50  0.70;
     1.20 -1.00
];

cif_p = CIF(beta, xnames, stimNames, 'poisson');
cif_b = CIF(beta, xnames, stimNames, 'binomial');

N = size(stim_vals, 1);
poisson_lambda_delta = zeros(N, 1);
poisson_gradient = zeros(N, 2);
poisson_gradient_log = zeros(N, 2);
poisson_jacobian = zeros(2, 2, N);
poisson_jacobian_log = zeros(2, 2, N);

binomial_lambda_delta = zeros(N, 1);
binomial_gradient = zeros(N, 2);
binomial_gradient_log = zeros(N, 2);
binomial_jacobian = zeros(2, 2, N);
binomial_jacobian_log = zeros(2, 2, N);

for k = 1:N
    s = stim_vals(k, :)';

    poisson_lambda_delta(k, 1) = cif_p.evalLambdaDelta(s);
    poisson_gradient(k, :) = cif_p.evalGradient(s);
    poisson_gradient_log(k, :) = cif_p.evalGradientLog(s);
    poisson_jacobian(:, :, k) = cif_p.evalJacobian(s);
    poisson_jacobian_log(:, :, k) = cif_p.evalJacobianLog(s);

    binomial_lambda_delta(k, 1) = cif_b.evalLambdaDelta(s);
    binomial_gradient(k, :) = cif_b.evalGradient(s);
    binomial_gradient_log(k, :) = cif_b.evalGradientLog(s);
    binomial_jacobian(:, :, k) = cif_b.evalJacobian(s);
    binomial_jacobian_log(:, :, k) = cif_b.evalJacobianLog(s);
end

cif_copy = cif_p.CIFCopy();
copy_b = cif_copy.b;
copy_fitType = cif_copy.fitType;
is_sym_beta = cif_p.isSymBeta();

save(outputFile, ...
    'beta', ...
    'stim_vals', ...
    'poisson_lambda_delta', ...
    'poisson_gradient', ...
    'poisson_gradient_log', ...
    'poisson_jacobian', ...
    'poisson_jacobian_log', ...
    'binomial_lambda_delta', ...
    'binomial_gradient', ...
    'binomial_gradient_log', ...
    'binomial_jacobian', ...
    'binomial_jacobian_log', ...
    'copy_b', ...
    'copy_fitType', ...
    'is_sym_beta');

fprintf('Wrote CIF fixtures to %s\n', outputFile);
end
