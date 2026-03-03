function Analysis_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for Analysis parity checks.
%
% This fixture focuses on deterministic GLM fit/diagnostic quantities used by
% nSTAT-python Analysis compatibility methods.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'Analysis');
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

X = [
    -1.00  0.20;
    -0.50 -0.10;
     0.00  0.00;
     0.30  0.80;
     0.70 -0.60;
     1.10  0.40;
     1.60 -1.20;
     2.00  0.90
];

y_poisson = [0; 1; 0; 2; 1; 3; 2; 4];
y_binomial = [0; 0; 1; 0; 1; 1; 0; 1];
dt = 0.1;

% Poisson fit with intercept (MATLAB default for glmfit).
[b_poisson, ~, ~] = glmfit(X, y_poisson, 'poisson', 'link', 'log');
eta_poisson = [ones(size(X,1),1), X] * b_poisson;
mu_poisson = exp(eta_poisson);
loglik_poisson = sum(y_poisson .* log(mu_poisson) - mu_poisson - gammaln(y_poisson + 1));
residual_poisson = (y_poisson - mu_poisson) ./ sqrt(max(mu_poisson, 1e-12));
cum_poisson = cumsum(max(mu_poisson, 1e-12));
invgaus_poisson = cum_poisson(y_poisson > 0);
[ks_d_poisson, ks_n_poisson] = compute_ks(invgaus_poisson);

% Binomial fit with intercept (MATLAB default for glmfit).
[b_binomial, ~, ~] = glmfit(X, y_binomial, 'binomial', 'link', 'logit');
eta_binomial = [ones(size(X,1),1), X] * b_binomial;
p_binomial = exp(eta_binomial) ./ (1 + exp(eta_binomial));
p_binomial = min(max(p_binomial, 1e-9), 1 - 1e-9);
loglik_binomial = sum(y_binomial .* log(p_binomial) + (1 - y_binomial) .* log(1 - p_binomial));
residual_binomial = (y_binomial - p_binomial) ./ sqrt(max(p_binomial .* (1 - p_binomial), 1e-12));
cum_binomial = cumsum(p_binomial);
invgaus_binomial = cum_binomial(y_binomial > 0);
[ks_d_binomial, ks_n_binomial] = compute_ks(invgaus_binomial);

% Benjamini-Hochberg expected mask.
p_values = [0.001; 0.01; 0.03; 0.04; 0.20; 0.60];
alpha = 0.05;
[p_sorted, order] = sort(p_values);
threshold = alpha * ((1:numel(p_values))' / numel(p_values));
passing = find(p_sorted <= threshold);
fdr_mask = false(size(p_values));
if ~isempty(passing)
    cutoff = p_sorted(max(passing));
    fdr_mask = p_values <= cutoff;
end
fdr_order = order;
fdr_threshold = threshold;

save(outputFile, ...
    'X', ...
    'y_poisson', ...
    'y_binomial', ...
    'dt', ...
    'b_poisson', ...
    'mu_poisson', ...
    'loglik_poisson', ...
    'residual_poisson', ...
    'invgaus_poisson', ...
    'ks_d_poisson', ...
    'ks_n_poisson', ...
    'b_binomial', ...
    'p_binomial', ...
    'loglik_binomial', ...
    'residual_binomial', ...
    'invgaus_binomial', ...
    'ks_d_binomial', ...
    'ks_n_binomial', ...
    'p_values', ...
    'alpha', ...
    'fdr_mask', ...
    'fdr_order', ...
    'fdr_threshold');

fprintf('Wrote Analysis fixtures to %s\n', outputFile);
end


function [d_stat, n_events] = compute_ks(values)
if isempty(values)
    d_stat = 0;
    n_events = 0;
    return;
end
z = sort(values ./ max(max(values), 1e-12));
n = numel(z);
ecdf = (1:n)' / n;
d_plus = max(ecdf - z);
d_minus = max(z - ((0:(n-1))' / n));
d_stat = max(d_plus, d_minus);
n_events = n;
end
