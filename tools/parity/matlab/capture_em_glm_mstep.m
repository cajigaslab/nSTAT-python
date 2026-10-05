function capture_em_glm_mstep()
%CAPTURE_EM_GLM_MSTEP  Gold-fixture capture for the GLM M-step of the EM drivers.
%
% One M-step call with MstepMethod = 'GLM' of
%   nstat.decoding.PointProcessEM.PP_MStep  and
%   nstat.decoding.PPLFP.PPLFP_MStep
% from the repaired MATLAB (fix/pp-em, final frozen head aa88a2b; pending
% upstream merge; the M-steps are identical at 8dbd0e4, and a capture from
% there was bit-identical apart from this note).  The GLM M-step regresses each cell's spikes on the smoothed means
% x_K through Covariate -> Trial -> TrialConfig ->
% Analysis.RunAnalysisForAllNeurons ('GLM' for poisson, 'BNLRCG' for
% binomial) -> FitResSummary and reads mu ('constant'), beta ('v<i>') and the
% history coefficients (window labels) BY LABEL; a coefficient FitResSummary
% reports as NaN (se >= 100) keeps its previous value (F3, R4a, F1); the time
% base is delta (C6, R4c).  A, Q (C, R, alpha), x0 and Px0 are the closed-form
% updates.  There is no Monte Carlo in this path, so the outputs are
% deterministic.
%
% Reproduces: tests/parity/fixtures/matlab_gold/em_glm_mstep.mat
%
% rng(42) once; every case then draws its own synthetic data in order
% (dx = 2 states, N = 1500 bins, an AR(1) latent path with diagonal Q):
%   pp_pois    PP_MStep,    poisson,  C = 3, windows [0 2 5 10] ms
%   pp_binom   PP_MStep,    binomial, C = 3, windows [0 2 5 10] ms
%   lfp_pois   PPLFP_MStep, poisson,  C = 3, windows [0 2 5 10] ms (dy = 2)
%   lfp_binom  PPLFP_MStep, binomial, C = 3, windows [0 2 5 10] ms
%   pp_unest   PP_MStep,    poisson,  C = 4, hard 1-bin refractory period,
%              windows [0 1 5 20] ms: the (0,1] ms window is separated for
%              every cell (se >= 100), so its gamma row keeps its previous
%              value (R4a)
%   lfp_unest  PPLFP_MStep, the same refractory problem
%   pp_c1      PP_MStep,    poisson,  C = 1 (single cell; FitResSummary
%              getCoeffs returns a 1 x nLabels row, F3)
%   lfp_c1     PPLFP_MStep, binomial, C = 1
%   pp_2ms     PP_MStep,    poisson,  C = 3, delta = 2 ms, windows
%              [0 4 10 20] ms (the delta time base, R4c)
% The history tensor is built exactly as PP_EM / PPLFP_EM build it
% (History(windowTimes,0,maxTime).computeHistory of nspikeTrain(t,'',delta)),
% the M-step inputs x_K / W_K / ExpectationSums come from one PP_EStep /
% PPLFP_EStep at the generating parameters, and the previous gamma is
% -0.2 everywhere (nonzero, so the history is fitted).  The constraints are
% the PP_EMCreateConstraints() / PPLFP_EMCreateConstraints() defaults.
%
% Saved per case (prefix <name>_): inputs dN, x_K, x0, Px0, every
% ExpectationSums field (ES_<field>), fitType, mu, beta, gamma, windowTimes,
% HkAll, delta (and y for PPLFP), and the M-step outputs Ahat, Qhat,
% muhat_new, betahat_new, gammahat_new, x0hat, Px0hat (and Chat, Rhat,
% alphahat for PPLFP).  W_K is not saved: the GLM branch never reads it (only
% the Newton-Raphson branch draws from it), and it would triple the file.
%
% The MATLAB nSTAT checkout is read from the NSTAT_MATLAB_PATH environment
% variable (required).

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo)
    error('capture:noMatlabPath', 'Set NSTAT_MATLAB_PATH to the MATLAB nSTAT checkout.');
end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'em_glm_mstep.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

rng(42);
wStd = [0 0.002 0.005 0.010];
wRef = [0 0.001 0.005 0.020];
cases = struct( ...
    'name',       {'pp_pois', 'pp_binom', 'lfp_pois', 'lfp_binom', 'pp_unest', 'lfp_unest', ...
                   'pp_c1', 'lfp_c1', 'pp_2ms'}, ...
    'family',     {'PP', 'PP', 'PPLFP', 'PPLFP', 'PP', 'PPLFP', 'PP', 'PPLFP', 'PP'}, ...
    'fitType',    {'poisson', 'binomial', 'poisson', 'binomial', 'poisson', 'poisson', ...
                   'poisson', 'binomial', 'poisson'}, ...
    'C',          {3, 3, 3, 3, 4, 4, 1, 1, 3}, ...
    'refractory', {false, false, false, false, true, true, false, false, false}, ...
    'delta',      {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002}, ...
    'windowTimes', {wStd, wStd, wStd, wStd, wRef, wRef, wStd, wStd, [0 0.004 0.010 0.020]});

dx = 2; N = 1500;
A = [0.98 0.02; -0.03 0.96];
Q = diag([0.01 0.02]);
x0 = zeros(dx, 1);
Px0 = 1e-3 * eye(dx);
betaAll = [0.9 -0.7 0.5 0.6; 0.4 0.8 -0.6 -0.3];
Cm = [1 0.5; -0.3 1];
R = diag([0.05 0.08]);
alpha = [0.1; -0.1];

out = struct();
for i = 1:numel(cases)
    cs = cases(i);
    C = cs.C; delta = cs.delta; wt = cs.windowTimes; nW = numel(wt) - 1;
    maxTime = (N - 1) * delta;
    x = zeros(dx, N);
    xPrev = x0;
    cholQ = chol(Q, 'lower');
    for k = 1:N
        xPrev = A * xPrev + cholQ * randn(dx, 1);
        x(:, k) = xPrev;
    end
    rate = linspace(40, 60, C)' * delta;            % spikes per bin
    beta = betaAll(:, 1:C);
    if strcmp(cs.fitType, 'poisson')
        mu = log(rate);
        p = min(exp(mu + beta' * x), 1);
    else
        mu = log(rate ./ (1 - rate));
        e = exp(mu + beta' * x);
        p = e ./ (1 + e);
    end
    u = rand(C, N);
    dN = zeros(C, N);
    for c = 1:C
        for k = 1:N
            if u(c, k) < p(c, k) && ~(cs.refractory && k > 1 && dN(c, k-1) == 1)
                dN(c, k) = 1;
            end
        end
    end
    histObj = History(wt, 0, maxTime);
    HkAll = zeros(N, nW, C);
    for c = 1:C
        nst = nspikeTrain((find(dN(c, :) == 1) - 1) * delta, '', delta);
        nst.setMinTime(0);
        nst.setMaxTime(maxTime);
        HkAll(:, :, c) = histObj.computeHistory(nst).dataToMatrix;
    end
    gamma = -0.2 * ones(nW, C);
    p = [cs.name '_'];
    fprintf('  [%s] %s %s, C = %d, delta = %g, spikes per cell = %s\n', cs.name, cs.family, ...
        cs.fitType, C, delta, mat2str(sum(dN, 2)'));
    if strcmp(cs.family, 'PP')
        [x_K, W_K, ~, ES] = nstat.decoding.PointProcessEM.PP_EStep( ...
            A, Q, dN, mu, beta, cs.fitType, gamma, HkAll, x0, Px0);
        cons = nstat.decoding.PointProcessEM.PP_EMCreateConstraints();
        [Ahat, Qhat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat] = ...
            nstat.decoding.PointProcessEM.PP_MStep(dN, x_K, W_K, x0, Px0, ES, cs.fitType, ...
                mu, beta, gamma, wt, HkAll, cons, 'GLM', delta);
    else
        y = Cm * x + alpha + chol(R, 'lower') * randn(2, N);
        [x_K, W_K, ~, ES] = nstat.decoding.PPLFP.PPLFP_EStep( ...
            A, Q, Cm, R, y, alpha, dN, mu, beta, cs.fitType, delta, gamma, HkAll, x0, Px0);
        cons = nstat.decoding.PPLFP.PPLFP_EMCreateConstraints();
        [Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat] = ...
            nstat.decoding.PPLFP.PPLFP_MStep(dN, y, x_K, W_K, x0, Px0, ES, cs.fitType, ...
                mu, beta, gamma, wt, HkAll, cons, 'GLM', delta);
        out.([p 'y']) = y;
        out.([p 'Chat']) = Chat;
        out.([p 'Rhat']) = Rhat;
        out.([p 'alphahat']) = alphahat;
    end
    fprintf('      gammahat_new = %s\n', mat2str(gammahat_new, 4));
    out.([p 'family']) = cs.family;
    out.([p 'fitType']) = cs.fitType;
    out.([p 'dN']) = dN;
    out.([p 'x_K']) = x_K;
    out.([p 'x0']) = x0;
    out.([p 'Px0']) = Px0;
    out.([p 'mu']) = mu;
    out.([p 'beta']) = beta;
    out.([p 'gamma']) = gamma;
    out.([p 'windowTimes']) = wt;
    out.([p 'HkAll']) = HkAll;
    out.([p 'delta']) = delta;
    fn = fieldnames(ES);
    for j = 1:numel(fn)
        out.([p 'ES_' fn{j}]) = ES.(fn{j});
    end
    out.([p 'Ahat']) = Ahat;
    out.([p 'Qhat']) = Qhat;
    out.([p 'muhat_new']) = muhat_new;
    out.([p 'betahat_new']) = betahat_new;
    out.([p 'gammahat_new']) = gammahat_new;
    out.([p 'x0hat']) = x0hat;
    out.([p 'Px0hat']) = Px0hat;
end
out.case_names = {cases.name};
out.matlab_version = version;
out.matlab_source_note = ['Captured from the repaired MATLAB nSTAT fix/pp-em @ aa88a2b ' ...
                          '(pending upstream merge)'];

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
