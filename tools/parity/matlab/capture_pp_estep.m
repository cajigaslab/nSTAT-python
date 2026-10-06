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
% Six cases share one rng(42) latent path (dx = 2 states, C = 3 cells,
% N = 150 bins of delta = 1 ms):
%   c1  poisson,  no history  (HkAll = zeros(N,1,C), gamma = 0)
%   c2  poisson,  history     (nW = 2 windows ~= C cells, nonzero gamma)
%   c3  binomial, history     (nW = 4 windows ~= C cells, nonzero gamma)
%   c4  binomial, no history  (HkAll = zeros(N,1,C), gamma = 0)
%   c5  poisson,  history     (nW = 3 windows == C cells, non-symmetric gamma)
%   c6  binomial, history     (nW = 3 windows == C cells, non-symmetric gamma)
% History tensors are built exactly as nstat.decoding.PointProcessEM.PP_EM
% builds them: HkAll(:,:,c) = History(windowTimes,0,maxTime)
%                               .computeHistory(nst{c}).dataToMatrix
% (N x nW x C).  c5 / c6 (square history) were appended when this fixture
% was recaptured from the repaired MATLAB (fix/pp-em @ a457b54, pending
% upstream merge): its PP_EStep orients each history slice by its columns
% (C3), so the square log-likelihood no longer transposes the slice.  The
% case gammas are literals and every rng draw happens before the case loop,
% so c1-c4 are bit-identical to the earlier capture.  mu is moderate so that
% no exp() argument comes near the clipping guards some ports apply.
%
% Saved per case (prefix cK_): inputs A, Q, dN, mu, beta, fitType, gamma,
% HkAll, x0, Px0, windowTimes, delta and outputs x_K, W_K, logll plus every
% field of the ExpectationSums struct (cK_ES_<field>).  PP_EStep itself is
% deterministic; rng(42) only seeds the synthetic inputs.
%
% The MATLAB nSTAT checkout is read from the NSTAT_MATLAB_PATH environment
% variable (required).

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo)
    error('capture:noMatlabPath', 'Set NSTAT_MATLAB_PATH to the MATLAB nSTAT checkout.');
end
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

gammaSq = [-1.2 -0.1 -0.8; -0.3 -1.0 -0.05; -0.9 -0.6 -0.4];     % nW == C, non-symmetric
cases = struct( ...
    'name',        {'c1', 'c2', 'c3', 'c4', 'c5', 'c6'}, ...
    'fitType',     {'poisson', 'poisson', 'binomial', 'binomial', 'poisson', 'binomial'}, ...
    'windowTimes', {[], [0 0.001 0.003], [0 0.001 0.002 0.004 0.008], [], ...
                    [0 0.001 0.002 0.004], [0 0.001 0.002 0.004]}, ...
    'gamma',       {0, [-0.8 -0.5 -1.0; -0.3 -0.2 -0.4], ...
                    [-1.2 -0.9 -1.0; -0.6 -0.4 -0.5; -0.3 -0.2 -0.25; -0.1 -0.05 -0.08], 0, ...
                    gammaSq, gammaSq});

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
% The MATLAB checkout's git commit (the repaired fix/pp-em branch, pending
% upstream merge), as export_pplfp_gold_fixtures.m records it.
srcDir = fileparts(which('nstat.decoding.PointProcessEM'));
[status, sha] = system(['git -C "' srcDir '" rev-parse --short HEAD']);
if status ~= 0
    sha = 'unknown commit';
end
out.matlab_source_note = ['Captured from the repaired MATLAB nSTAT fix/pp-em @ ' strtrim(sha) ...
                          ' (pending upstream merge)'];

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
