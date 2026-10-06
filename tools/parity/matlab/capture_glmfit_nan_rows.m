function capture_glmfit_nan_rows()
%CAPTURE_GLMFIT_NAN_ROWS  Gold-fixture capture for Analysis.GLMFit's poisson
% ('GLM') path with a NaN row in the design matrix.
%
% MATLAB's glmfit (toolbox/stats/stats/glmfit.m) calls statremovenan(y,x,...)
% before fitting: rows where X or y is NaN are dropped from the regression
% (b, dev, stats.se), but Analysis.GLMFit (Analysis.m:483) then evaluates
% data = exp(X*b) on the ORIGINAL, full X -- so a NaN row in X still gives a
% NaN row in data/lambda. Analysis.GLMFit's eps floors use MATLAB's `max`,
% which (unlike numpy's np.maximum) ignores NaN and returns the other
% operand: max(NaN, eps) == eps. So lambdaDelta and oneMinusLambdaDelta are
% both floored to eps at the NaN row (data*delta and 1-data*delta are both
% NaN there), and that row's logLL contribution collapses to exactly
% log(eps)*(y + (1-y)) = log(eps), independent of y's value at that row.
%
% This fixture pins all of: b, dev, stats.se, stats.covb, AIC, BIC, logLL,
% and the full per-row lambda/data vector (including the NaN row) for one
% small poisson design with a NaN in one non-intercept column of one row.
%
% Reproduces: tests/parity/fixtures/matlab_gold/glmfit_nan_rows.mat
%
% The MATLAB nSTAT checkout is read from the NSTAT_MATLAB_PATH environment
% variable (required) only to put nSTAT on the path for consistency with the
% other captures; this fixture exercises glmfit directly (no nSTAT classes
% needed to reproduce Analysis.GLMFit's post-processing, which is ported
% verbatim below from Analysis.m:565-634).

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo)
    error('capture:noMatlabPath', 'Set NSTAT_MATLAB_PATH to the MATLAB nSTAT checkout.');
end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'glmfit_nan_rows.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

rng(42);
sampleRate = 1000;
delta = 1 / sampleRate;

n = 30;
p = 3; % intercept ('one') + 2 covariates, constant 'off' (X carries its own intercept column)
Xtrue = [ones(n, 1), randn(n, 1), randn(n, 1)];
btrue = [-1.0; 0.5; -0.3];
eta = Xtrue * btrue;
y = poissrnd(exp(eta));

% Inject a NaN into one non-intercept column of one row (row 7, column 2).
nanRow = 7;
X = Xtrue;
X(nanRow, 2) = NaN;

[b, dev, stats] = glmfit(X, y, 'poisson', 'link', 'log', 'constant', 'off');
b = real(b);

data = exp(X * b) .* sampleRate; % Analysis.GLMFit's `data`, full X (incl. the NaN row)
AIC = 2 * length(b) + real(dev);
BIC = length(b) * log(length(y)) + real(dev);
lambdaDelta = max(data * delta, eps);
oneMinusLambdaDelta = max(1 - data * delta, eps);
logLL = sum(y .* log(lambdaDelta) + (1 - y) .* log(oneMinusLambdaDelta));

out = struct();
out.X = X;
out.y = y;
out.nanRow = nanRow;
out.sampleRate = sampleRate;
out.b = b;
out.dev = real(dev);
out.se = stats.se;
out.covb = stats.covb;
out.dfe = stats.dfe;
out.data = data;
out.AIC = AIC;
out.BIC = BIC;
out.logLL = logLL;
out.lambdaDelta = lambdaDelta;
out.oneMinusLambdaDelta = oneMinusLambdaDelta;
out.matlab_version = version;

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
