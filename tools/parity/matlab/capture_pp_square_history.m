function capture_pp_square_history()
%CAPTURE_PP_SQUARE_HISTORY  Gold fixture: PPAF filters with a square history.
%
% MATLAB reorients the history coefficients gamma only with
%   if(size(gamma,2)~=C) gamma=gamma'; end
% (PPAF.PPDecodeFilterLinear, PPAF.PP_fixedIntervalSmoother), and
% PPAF.PPDecode_updateLinear / PointProcessEM.PP_EStep use gamma as given,
% so a square gamma (nW history windows == C cells) is never transposed.
% These cases pin that rule, plus the N == C (time bins == cells) and C == 1
% layouts of the (N x nW x C) history tensor.
%
% Reproduces: tests/parity/fixtures/matlab_gold/pp_square_history.mat
%
% dx = 2 states, delta = 1 ms, rng(42) synthetic inputs (latent AR(1) path,
% Bernoulli spikes), non-symmetric gamma = -0.2 - 0.8*rand(nW, C):
%   pdfl_pois_sq     PPDecodeFilterLinear      poisson,  C = 3, nW = 3, N = 150
%   pdfl_binom_sq    PPDecodeFilterLinear      binomial, C = 4, nW = 4, N = 150
%   pfis_pois_sq     PP_fixedIntervalSmoother  poisson,  C = 3, nW = 3, N = 150, lags = 1
%   pdfl_pois_ctrl   PPDecodeFilterLinear      poisson,  C = 3, nW = 2, N = 150 (control, nW ~= C)
%   estep_pois_sq    PP_EStep                  poisson,  C = 3, nW = 3, N = 150
%   estep_pois_NeqC  PP_EStep                  poisson,  C = 6, nW = 2, N = 6   (N == C)
%   estep_binom_C1   PP_EStep                  binomial, C = 1, nW = 2, N = 150 (HkAll is N x nW)
%
% Saved per case (prefix <name>_): inputs A, Q, dN, mu, beta, fitType, gamma,
% windowTimes, delta, x0, Pi0 (PP_EStep: Px0), lags (smoother), HkAll and
% sizes = [N nW C dx]; outputs x_p, W_p, x_u, W_u (PPDecodeFilterLinear),
% x_pLag, W_pLag, x_uLag, W_uLag (PP_fixedIntervalSmoother) or x_K, W_K
% (PP_EStep; its logll is deliberately not captured here).
%
% HkAll is the history tensor each function consumes: for the two filters it
% is rebuilt exactly as PPAF.m builds it internally from windowTimes
% (History.computeHistory on nspikeTrain.resample(1/delta)); for PP_EStep it
% is built as PointProcessEM.PP_EM builds it and passed in.  MATLAB keeps
% HkAll as N x nW x C, which it stores as N x nW when C == 1.
%
% MATLAB nSTAT repo is resolved from NSTAT_MATLAB_PATH, defaulting to
% /Users/iahncajigas/projects/nstat.

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo); matlabRepo = '/Users/iahncajigas/projects/nstat'; end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'pp_square_history.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

rng(42);
dx = 2; delta = 0.001;
A   = [0.98 0.02; -0.03 0.96];
Q   = diag([0.02 0.015]);
x0  = [0.2; -0.1];
Pi0 = diag([0.05 0.08]);

cases = struct( ...
    'name',        {'pdfl_pois_sq', 'pdfl_binom_sq', 'pfis_pois_sq', 'pdfl_pois_ctrl', ...
                    'estep_pois_sq', 'estep_pois_NeqC', 'estep_binom_C1'}, ...
    'func',        {'PPDecodeFilterLinear', 'PPDecodeFilterLinear', 'PP_fixedIntervalSmoother', ...
                    'PPDecodeFilterLinear', 'PP_EStep', 'PP_EStep', 'PP_EStep'}, ...
    'fitType',     {'poisson', 'binomial', 'poisson', 'poisson', 'poisson', 'poisson', 'binomial'}, ...
    'C',           {3, 4, 3, 3, 3, 6, 1}, ...
    'N',           {150, 150, 150, 150, 150, 6, 150}, ...
    'windowTimes', {[0 0.001 0.002 0.004], [0 0.001 0.002 0.004 0.008], [0 0.001 0.002 0.004], ...
                    [0 0.001 0.003], [0 0.001 0.002 0.004], [0 0.001 0.002], [0 0.001 0.003]}, ...
    'mu',          {[-2.2; -1.9; -2.5], [-1.6; -1.3; -1.9; -1.5], [-2.2; -1.9; -2.5], ...
                    [-2.2; -1.9; -2.5], [-2.2; -1.9; -2.5], -0.4 * ones(6, 1), -1.3});
lags = 1;

out = struct();
for i = 1:numel(cases)
    cs = cases(i);
    C = cs.C; N = cs.N; wt = cs.windowTimes; nW = numel(wt) - 1;
    maxTime = (N - 1) * delta;

    beta = 0.8 * randn(dx, C);
    gamma = -0.2 - 0.8 * rand(nW, C);
    if nW == C
        assert(norm(gamma - gamma', 'fro') > 0.1, 'square gamma must be non-symmetric');
    end

    % Latent AR(1) path (x_1 = A*x0 + w_1) and Bernoulli spikes.
    x = zeros(dx, N);
    cholQ = chol(Q, 'lower');
    xPrev = x0;
    for k = 1:N
        xPrev = A * xPrev + cholQ * randn(dx, 1);
        x(:, k) = xPrev;
    end
    eta = cs.mu + beta' * x;
    if strcmp(cs.fitType, 'poisson')
        p = exp(eta);
    else
        p = exp(eta) ./ (1 + exp(eta));
    end
    dN = double(rand(C, N) < p);

    % History tensor exactly as the called function consumes it.
    histObj = History(wt, 0, maxTime);
    HkAll = zeros(N, nW, C);
    for c = 1:C
        nst = nspikeTrain((find(dN(c, :) == 1) - 1) * delta);
        nst.setMinTime(0);
        nst.setMaxTime(maxTime);
        if ~strcmp(cs.func, 'PP_EStep')
            nst = nst.resample(1 / delta);   % PPAF.m:470-473
        end
        HkAll(:, :, c) = histObj.computeHistory(nst).dataToMatrix;
    end

    p_ = [cs.name '_'];
    switch cs.func
        case 'PPDecodeFilterLinear'
            [x_p, W_p, x_u, W_u] = nstat.decoding.PPAF.PPDecodeFilterLinear( ...
                A, Q, dN, cs.mu, beta, cs.fitType, delta, gamma, wt, x0, Pi0);
            out.([p_ 'x_p']) = x_p; out.([p_ 'W_p']) = W_p;
            out.([p_ 'x_u']) = x_u; out.([p_ 'W_u']) = W_u;
            out.([p_ 'Pi0']) = Pi0;
        case 'PP_fixedIntervalSmoother'
            [x_pLag, W_pLag, x_uLag, W_uLag] = nstat.decoding.PPAF.PP_fixedIntervalSmoother( ...
                A, Q, dN, lags, cs.mu, beta, cs.fitType, delta, gamma, wt, x0, Pi0);
            out.([p_ 'x_pLag']) = x_pLag; out.([p_ 'W_pLag']) = W_pLag;
            out.([p_ 'x_uLag']) = x_uLag; out.([p_ 'W_uLag']) = W_uLag;
            out.([p_ 'Pi0']) = Pi0;
            out.([p_ 'lags']) = lags;
        case 'PP_EStep'
            [x_K, W_K] = nstat.decoding.PointProcessEM.PP_EStep( ...
                A, Q, dN, cs.mu, beta, cs.fitType, gamma, HkAll, x0, Pi0);
            out.([p_ 'x_K']) = x_K; out.([p_ 'W_K']) = W_K;
            out.([p_ 'Px0']) = Pi0;
    end

    out.([p_ 'func']) = cs.func;
    out.([p_ 'A']) = A;
    out.([p_ 'Q']) = Q;
    out.([p_ 'dN']) = dN;
    out.([p_ 'mu']) = cs.mu;
    out.([p_ 'beta']) = beta;
    out.([p_ 'fitType']) = cs.fitType;
    out.([p_ 'gamma']) = gamma;
    out.([p_ 'windowTimes']) = wt;
    out.([p_ 'delta']) = delta;
    out.([p_ 'x0']) = x0;
    out.([p_ 'HkAll']) = HkAll;
    out.([p_ 'sizes']) = [N nW C dx];
    fprintf('  [%s] %s %s, N=%d nW=%d C=%d, size(HkAll)=%s, spikes per cell=%s\n', ...
        cs.name, cs.func, cs.fitType, N, nW, C, mat2str(size(HkAll)), mat2str(sum(dN, 2)'));
end
out.case_names = {cases.name};
out.matlab_version = version;

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
