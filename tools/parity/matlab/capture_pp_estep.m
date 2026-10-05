function capture_pp_estep()
%CAPTURE_PP_ESTEP  Gold-fixture capture for PointProcessEM.PP_EStep.
%
% E-step of the point-process state-space EM: forward point-process
% adaptive filter (PPDecode_updateLinear / PPDecode_predict), RTS smoother,
% lag-one cross-covariances, sufficient statistics and the expected
% complete-data log-likelihood.
%
% Reproduces: tests/parity/fixtures/matlab_gold/pp_estep.mat
%
% Four cases share one rng(42) latent path (dx = 2 states, C = 3 cells,
% N = 150 bins of delta = 1 ms):
%   c1  poisson,  no history  (HkAll = zeros(N,1,C), gamma = 0)
%   c2  poisson,  history     (nW = 2 windows ~= C cells, nonzero gamma)
%   c3  binomial, history     (nW = 4 windows ~= C cells, nonzero gamma)
%   c4  binomial, no history  (HkAll = zeros(N,1,C), gamma = 0)
% History tensors are built exactly as nstat.decoding.PointProcessEM.PP_EM
% builds them: HkAll(:,:,c) = History(windowTimes,0,maxTime)
%                               .computeHistory(nst{c}).dataToMatrix
% (N x nW x C).  nW ~= C on purpose: a square nW == C history would hide
% window/cell orientation errors.  mu is moderate so that no exp() argument
% comes near the clipping guards some ports apply.
%
% Saved per case (prefix cK_): inputs A, Q, dN, mu, beta, fitType, gamma,
% HkAll, x0, Px0, windowTimes, delta and outputs x_K, W_K, logll plus every
% field of the ExpectationSums struct (cK_ES_<field>).  PP_EStep itself is
% deterministic; rng(42) only seeds the synthetic inputs.
%
% MATLAB nSTAT repo is resolved from NSTAT_MATLAB_PATH, defaulting to
% /Users/iahncajigas/projects/nstat.

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo); matlabRepo = '/Users/iahncajigas/projects/nstat'; end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'pp_estep.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

rng(42);
dx = 2; C = 3; N = 150; delta = 0.001;
maxTime = (N - 1) * delta;

A   = [0.98 0.02; -0.03 0.96];
Q   = diag([0.02 0.015]);
x0  = [0.2; -0.1];
Px0 = diag([0.05 0.08]);
beta = [0.9 -0.7 0.5; 0.4 0.8 -0.6];      % dx x C
muP  = [-2.2; -1.9; -2.5];                % poisson log-rate per bin
muB  = [-1.6; -1.3; -1.9];                % binomial logit per bin

% Latent AR(1) path, MATLAB PP_EStep convention x_1 = A*x0 + w_1.
x = zeros(dx, N);
cholQ = chol(Q, 'lower');
xPrev = x0;
for k = 1:N
    xPrev = A * xPrev + cholQ * randn(dx, 1);
    x(:, k) = xPrev;
end
etaP = muP + beta' * x;
etaB = muB + beta' * x;
dNP = double(rand(C, N) < exp(etaP));
dNB = double(rand(C, N) < exp(etaB) ./ (1 + exp(etaB)));

cases = struct( ...
    'name',        {'c1', 'c2', 'c3', 'c4'}, ...
    'fitType',     {'poisson', 'poisson', 'binomial', 'binomial'}, ...
    'windowTimes', {[], [0 0.001 0.003], [0 0.001 0.002 0.004 0.008], []}, ...
    'gamma',       {0, [-0.8 -0.5 -1.0; -0.3 -0.2 -0.4], ...
                    [-1.2 -0.9 -1.0; -0.6 -0.4 -0.5; -0.3 -0.2 -0.25; -0.1 -0.05 -0.08], 0});

out = struct();
for i = 1:numel(cases)
    cs = cases(i);
    if strcmp(cs.fitType, 'poisson')
        dN = dNP; mu = muP;
    else
        dN = dNB; mu = muB;
    end
    if isempty(cs.windowTimes)
        HkAll = zeros(N, 1, C);
    else
        histObj = History(cs.windowTimes, 0, maxTime);
        nW = numel(cs.windowTimes) - 1;
        HkAll = zeros(N, nW, C);
        for c = 1:C
            nst = nspikeTrain((find(dN(c, :) == 1) - 1) * delta);
            nst.setMinTime(0);
            nst.setMaxTime(maxTime);
            HkAll(:, :, c) = histObj.computeHistory(nst).dataToMatrix;
        end
        assert(isequal(size(HkAll), [N nW C]), 'HkAll must be N x nW x C');
        assert(isequal(size(cs.gamma), [nW C]), 'gamma must be nW x C');
    end
    fprintf('  [%s] %s, size(HkAll) = %s, spikes per cell = %s\n', cs.name, ...
        cs.fitType, mat2str(size(HkAll)), mat2str(sum(dN, 2)'));

    [x_K, W_K, logll, ES] = nstat.decoding.PointProcessEM.PP_EStep( ...
        A, Q, dN, mu, beta, cs.fitType, cs.gamma, HkAll, x0, Px0);

    p = [cs.name '_'];
    out.([p 'A']) = A;
    out.([p 'Q']) = Q;
    out.([p 'dN']) = dN;
    out.([p 'mu']) = mu;
    out.([p 'beta']) = beta;
    out.([p 'fitType']) = cs.fitType;
    out.([p 'gamma']) = cs.gamma;
    out.([p 'HkAll']) = HkAll;
    out.([p 'x0']) = x0;
    out.([p 'Px0']) = Px0;
    out.([p 'windowTimes']) = cs.windowTimes;
    out.([p 'delta']) = delta;
    out.([p 'x_K']) = x_K;
    out.([p 'W_K']) = W_K;
    out.([p 'logll']) = logll;
    fn = fieldnames(ES);
    for j = 1:numel(fn)
        out.([p 'ES_' fn{j}]) = ES.(fn{j});
    end
end
out.case_names = {cases.name};
out.matlab_version = version;

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
