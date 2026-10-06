function capture_em_drivers()
%CAPTURE_EM_DRIVERS  End-to-end gold for the EM drivers PP_EM and PPLFP_EM.
%
% Runs the real drivers of the repaired MATLAB (fix/pp-em, final frozen head
% aa88a2b; pending upstream merge as nSTAT PR #135) end to end:
%   nstat.decoding.PointProcessEM.PP_EM   (13 outputs, so the SE pass runs;
%                                          10 for pp_sep, see below)
%   nstat.decoding.PPLFP.PPLFP_EM         (15 outputs, the bare default call)
%
% The Monte Carlo draws of the Newton-Raphson M-step and of the SE pass use
% MATLAB randn (Ziggurat), which Python cannot reproduce
% (nstat.extras.matlab_rng matches rand only), so the gold is designed to
% separate what is deterministic from what is not:
%   * Deterministic (compared tightly): the returned estimates are the inputs
%     of the selected iterate's E-step, so one PP_EStep / PPLFP_EStep at the
%     returned estimates in the ORIGINAL coordinates (es_*) reproduces
%     xKFinal / WKFinal, its logll is IC.llcomp and its observation term is
%     IC.llobs (F10).  The closed-form M-step updates (A, Q, [C, R, alpha,]
%     x0, Px0) given those expectation sums (ms<j>_*) for three constraint
%     sets each.  The number of parameters (from IC.AIC and IC.llobs).
%   * Monte Carlo dependent (compared with tolerances justified by the
%     measured spread over Python seeds): every estimate, SE, Pvals, nIter
%     and the per-iteration logll trace (ll_trace, parsed from the
%     "logll:" lines the driver prints, captured with diary; num2str
%     precision).
%
% rng(42 + i) at the start of case i; each case simulates its own data
% (dx = 2, AR(1) latent state with a non-diagonal A and diagonal Q, C = 4
% cells, delta = 1 ms, history simulated causally):
%   pp_pois    PP_EM, poisson,  N = 800, rates 60..100 Hz, explicit windows
%              [0 2 5 10] ms (nW = 3 ~= C = 4), gammaTrue = [-1 -0.5 -0.2]'
%              per cell, gamma0 = 0.5*gammaTrue (3 x 4)
%   pp_binom   PP_EM, binomial, the same design
%   pp_defwin  PP_EM, poisson, windowTimes = [] and a nonzero 3 x 1 shared
%              gamma0 = [-0.25 -0.15 -0.05]': the driver's default-window
%              rule (windowTimes = 0:delta:size(gamma,1)*delta = [0 1 2 3] ms)
%              and its shared-column expansion (gamma -> 3 x 4) are exercised
%              by PP_EM itself (data simulated on those windows)
%   lfp_def    PPLFP_EM(y, dN, A0, Q0, C0, R0, alpha0, mu0, beta0): the bare
%              default call (poisson, no history, NewtonRaphson, x0 / Px0
%              not estimated, mcIter = 1000), N = 400, rates 30..60 Hz, dy = 2
%   pp_sep     PP_EM, poisson, N = 600, rates 30..60 Hz, windows [0 1 2 5] ms
%              and a refractory gammaTrue = [-3 -1 -0.3]': some cell has no
%              spike with a spike in its (0,1] ms window, so that history
%              coefficient is not identifiable (separated) and the
%              Newton-Raphson M-step walks it towards -Inf.  This case
%              records MATLAB's behaviour there and requests 10 outputs (no
%              SE): with SEs requested MATLAB does not return -- the observed
%              information is singular, eye/IObs is Inf/NaN and nearestSPD's
%              chol/eig loop never ends (svd and eig of NaN return NaN in
%              R2025b).
% The other PP cases are checked at capture time to have no separated window
% (every cell has a spike with a nonzero count in every window).
% PP cases pass PP_EMCreateConstraints(1,0,1,0,0,0,0,100,0): the defaults
% with mcIter = 100 (the PP SE pass honours mcIter since G2).  Initial values:
% A0 = A, Q0 = Q, mu0 = mu + 0.3, beta0 = 0.5*beta (and C0 = C, R0 = R,
% alpha0 = alpha for PPLFP).
%
% Saved per case (prefix <name>_): the inputs (dN, A0, Q0, mu0, beta0, gamma0,
% windowTimes as passed, delta, fitType, cons = the constraint fields in
% PP_EMCreateConstraints / PPLFP_EMCreateConstraints argument order, and y,
% C0, R0, alpha0 for PPLFP), the generating truth (x, mu, beta, gammaTrue),
% every driver output (IC, SE and Pvals as structs), ll_trace, the windows
% the E-step uses (es_windowTimes) with its HkAll (es_HkAll), the E-step at
% the returned estimates (es_x_K, es_W_K, es_logll, es_ES_<field>) and the
% closed-form M-step outputs ms<j>_<output> for the constraint sets
% ms<j>_cons, j = 1..3.
%
% Reproduces: tests/parity/fixtures/matlab_gold/em_drivers.mat
%
% The MATLAB nSTAT checkout is read from the NSTAT_MATLAB_PATH environment
% variable (required).

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo)
    error('capture:noMatlabPath', 'Set NSTAT_MATLAB_PATH to the MATLAB nSTAT checkout.');
end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'em_drivers.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

PPEM = 'nstat.decoding.PointProcessEM';
wStd = [0 0.002 0.005 0.010];
wSep = [0 0.001 0.002 0.005];
cases = struct( ...
    'name',     {'pp_pois', 'pp_binom', 'pp_defwin', 'lfp_def', 'pp_sep'}, ...
    'family',   {'PP', 'PP', 'PP', 'PPLFP', 'PP'}, ...
    'fitType',  {'poisson', 'binomial', 'poisson', 'poisson', 'poisson'}, ...
    'N',        {800, 800, 800, 400, 600}, ...
    'rates',    {[60 100], [60 100], [60 100], [30 60], [30 60]}, ...
    'wt',       {wStd, wStd, [], [], wSep}, ...
    'simWt',    {wStd, wStd, [0 0.001 0.002 0.003], [], wSep}, ...
    'gammaSim', {[-1; -0.5; -0.2], [-1; -0.5; -0.2], [-0.5; -0.3; -0.1], [], [-3; -1; -0.3]}, ...
    'nOut',     {13, 13, 13, 15, 10});

dx = 2; C = 4; delta = 0.001;
A = [0.98 0.02; -0.03 0.96];
Q = diag([0.01 0.02]);
betaTrue = [0.9 -0.7 0.5 0.6; 0.4 0.8 -0.6 -0.3];
Cm = [1 0.5; -0.3 1];
R = diag([0.05 0.08]);
alpha = [0.1; -0.1];

out = struct();
diaryFile = [tempname '.txt'];
for i = 1:numel(cases)
    cs = cases(i);
    rng(42 + i);
    N = cs.N; p = [cs.name '_'];
    % --- simulate ------------------------------------------------------
    x = zeros(dx, N);
    xPrev = zeros(dx, 1);
    cholQ = chol(Q, 'lower');
    for k = 1:N
        xPrev = A * xPrev + cholQ * randn(dx, 1);
        x(:, k) = xPrev;
    end
    rate = linspace(cs.rates(1), cs.rates(2), C)' * delta;
    if strcmp(cs.fitType, 'poisson')
        mu = log(rate);
    else
        mu = log(rate ./ (1 - rate));
    end
    simWt = cs.simWt;
    nWs = max(numel(simWt) - 1, 0);
    gammaTrue = repmat(cs.gammaSim, 1, C);
    u = rand(C, N);
    dN = zeros(C, N);
    for k = 1:N
        eta = mu + betaTrue' * x(:, k);
        for w = 1:nWs
            lags = find((1:N) * delta > simWt(w) + 1e-12 & (1:N) * delta <= simWt(w + 1) + 1e-12);
            lags = lags(lags < k);
            if ~isempty(lags)
                eta = eta + gammaTrue(w, :)' .* sum(dN(:, k - lags), 2);
            end
        end
        if strcmp(cs.fitType, 'poisson')
            pk = min(exp(eta), 1);
        else
            pk = exp(eta) ./ (1 + exp(eta));
        end
        dN(:, k) = double(u(:, k) < pk);
    end
    mu0 = mu + 0.3;
    beta0 = 0.5 * betaTrue;
    fprintf('  [%s] %s %s, spikes per cell = %s\n', cs.name, cs.family, cs.fitType, mat2str(sum(dN, 2)'));

    % --- the history the driver builds (es_windowTimes / es_HkAll) ------
    if strcmp(cs.family, 'PP')
        if isempty(cs.wt)
            gamma0 = 0.5 * cs.gammaSim;                 % shared nW x 1 column
        else
            gamma0 = 0.5 * gammaTrue;                   % nW x C
        end
    else
        gamma0 = [];
    end
    if isempty(cs.wt) && ~isempty(gamma0) && any(gamma0(:) ~= 0)
        esWt = 0:delta:size(gamma0, 1) * delta;          % the driver's default-window rule
    else
        esWt = cs.wt;
    end
    if isempty(esWt)
        HkAll = zeros(N, 1, C);
    else
        histObj = History(esWt, 0, (N - 1) * delta);
        HkAll = zeros(N, numel(esWt) - 1, C);
        for c = 1:C
            nst = nspikeTrain((find(dN(c, :) == 1) - 1) * delta, '', delta);
            nst.setMinTime(0);
            nst.setMaxTime((N - 1) * delta);
            HkAll(:, :, c) = histObj.computeHistory(nst).dataToMatrix;
        end
        % spikes with a nonzero count in each window, per cell (0 = separated)
        nSpkHist = squeeze(sum(HkAll > 0 & permute(repmat(dN, [1 1 size(HkAll, 2)]), [2 3 1]), 1));
        fprintf('      spikes with history, window x cell = %s\n', mat2str(nSpkHist));
        if ~strcmp(cs.name, 'pp_sep') && any(nSpkHist(:) == 0)
            error('capture:separated', 'case %s has a separated history window', cs.name);
        end
        if strcmp(cs.name, 'pp_sep') && all(nSpkHist(:) > 0)
            error('capture:notSeparated', 'case pp_sep has no separated history window');
        end
        out.([p 'nSpkHist']) = nSpkHist;
    end

    % --- run the driver -------------------------------------------------
    if strcmp(cs.family, 'PP')
        consVec = [1 0 1 0 0 0 0 100 0];
        cons = nstat.decoding.PointProcessEM.PP_EMCreateConstraints(consVec(1), consVec(2), ...
            consVec(3), consVec(4), consVec(5), consVec(6), consVec(7), consVec(8), consVec(9));
        o = cell(1, cs.nOut);
        diary(diaryFile); t0 = tic;
        [o{1:cs.nOut}] = nstat.decoding.PointProcessEM.PP_EM(dN, A, Q, mu0, beta0, cs.fitType, delta, ...
            gamma0, cs.wt, [], [], cons);
        diary off; close all;
        names = {'xKFinal', 'WKFinal', 'Ahat', 'Qhat', 'muhat', 'betahat', 'gammahat', 'x0hat', ...
                 'Px0hat', 'IC', 'SE', 'Pvals', 'nIter'};
        names = names(1:cs.nOut);
    else
        y = Cm * x + alpha + chol(R, 'lower') * randn(2, N);
        consVec = [1 0 1 0 1 0 0 0 0 1000 0];          % PPLFP_EMCreateConstraints() defaults
        o = cell(1, 15);
        diary(diaryFile); t0 = tic;
        [o{1:15}] = nstat.decoding.PPLFP.PPLFP_EM(y, dN, A, Q, Cm, R, alpha, mu0, beta0);
        diary off; close all;
        names = {'xKFinal', 'WKFinal', 'Ahat', 'Qhat', 'Chat', 'Rhat', 'alphahat', 'muhat', ...
                 'betahat', 'gammahat', 'x0hat', 'Px0hat', 'IC', 'SE', 'Pvals'};
        out.([p 'y']) = y;
        out.([p 'C0']) = Cm;
        out.([p 'R0']) = R;
        out.([p 'alpha0']) = alpha;
    end
    fprintf('      %s ran in %.1f s\n', cs.family, toc(t0));
    txt = fileread(diaryFile);
    delete(diaryFile);
    tok = regexp(txt, 'logll: (\S+)', 'tokens');
    llTrace = cellfun(@(t) str2double(t{1}), tok);
    for j = 1:numel(names)
        out.([p names{j}]) = o{j};
    end
    est = cell2struct(o, names, 2);
    fprintf('      iterations printed = %d, gammahat = %s\n', numel(llTrace), mat2str(est.gammahat, 4));

    % --- the E-step at the returned estimates (original coordinates) ----
    if strcmp(cs.family, 'PP')
        [eX, eW, eLL, eES] = nstat.decoding.PointProcessEM.PP_EStep(est.Ahat, est.Qhat, dN, ...
            est.muhat, est.betahat, cs.fitType, est.gammahat, HkAll, est.x0hat, est.Px0hat);
        msCons = {[1 0 1 0 0 0 0 50 0], [1 1 0 0 1 1 0 50 0], [1 0 1 1 1 1 1 50 0]};
    else
        [eX, eW, eLL, eES] = nstat.decoding.PPLFP.PPLFP_EStep(est.Ahat, est.Qhat, est.Chat, ...
            est.Rhat, y, est.alphahat, dN, est.muhat, est.betahat, cs.fitType, delta, ...
            est.gammahat, HkAll, est.x0hat, est.Px0hat);
        msCons = {[1 0 1 0 1 0 0 0 0 50 0], [1 1 0 0 0 0 1 1 0 50 0], [1 0 1 1 1 1 1 1 1 50 0]};
    end
    if isfield(est, 'IC')
        relLL = abs(eLL - est.IC.llcomp) / abs(est.IC.llcomp);
    else
        relLL = 0;                                       % pp_sep: no IC requested
    end
    relX = max(abs(eX(:) - est.xKFinal(:))) / max(abs(est.xKFinal(:)));
    fprintf('      E-step at the estimates: |logll - IC.llcomp|/|IC.llcomp| = %.3g, max|x_K - xKFinal|/max|xKFinal| = %.3g\n', ...
        relLL, relX);
    if relLL > 1e-9 || relX > 1e-9
        error('capture:identity', 'E-step at the returned estimates does not reproduce the driver (%s)', cs.name);
    end
    out.([p 'es_windowTimes']) = esWt;
    out.([p 'es_HkAll']) = HkAll;
    out.([p 'es_x_K']) = eX;
    out.([p 'es_W_K']) = eW;
    out.([p 'es_logll']) = eLL;
    fn = fieldnames(eES);
    for j = 1:numel(fn)
        out.([p 'es_ES_' fn{j}]) = eES.(fn{j});
    end

    % --- the closed-form M-step updates given those sums -----------------
    % (The Monte Carlo mu / beta / gamma outputs of these calls are not
    % saved; rng(7) before each call only makes the capture repeatable.)
    if strcmp(cs.name, 'pp_sep')
        msCons = {};
    end
    for j = 1:numel(msCons)
        cv = num2cell(msCons{j});
        q = sprintf('%sms%d_', p, j);
        rng(7);
        if strcmp(cs.family, 'PP')
            mcons = nstat.decoding.PointProcessEM.PP_EMCreateConstraints(cv{:});
            evalc(['[mA, mQ, ~, ~, ~, mx0, mPx0] = ' PPEM '.PP_MStep(dN, eX, eW, est.x0hat, ' ...
                   'est.Px0hat, eES, cs.fitType, est.muhat, est.betahat, est.gammahat, esWt, ' ...
                   'HkAll, mcons, ''NewtonRaphson'', delta);']);
        else
            mcons = nstat.decoding.PPLFP.PPLFP_EMCreateConstraints(cv{:});
            evalc(['[mA, mQ, mC, mR, mAlpha, ~, ~, ~, mx0, mPx0] = nstat.decoding.PPLFP.PPLFP_MStep(' ...
                   'dN, y, eX, eW, est.x0hat, est.Px0hat, eES, cs.fitType, est.muhat, est.betahat, ' ...
                   'est.gammahat, esWt, HkAll, mcons, ''NewtonRaphson'', delta);']);
            out.([q 'Chat']) = mC;
            out.([q 'Rhat']) = mR;
            out.([q 'alphahat']) = mAlpha;
        end
        out.([q 'cons']) = msCons{j};
        out.([q 'Ahat']) = mA;
        out.([q 'Qhat']) = mQ;
        out.([q 'x0hat']) = mx0;
        out.([q 'Px0hat']) = mPx0;
    end

    out.([p 'family']) = cs.family;
    out.([p 'fitType']) = cs.fitType;
    out.([p 'dN']) = dN;
    out.([p 'A0']) = A;
    out.([p 'Q0']) = Q;
    out.([p 'mu0']) = mu0;
    out.([p 'beta0']) = beta0;
    out.([p 'gamma0']) = gamma0;
    out.([p 'windowTimes']) = cs.wt;
    out.([p 'delta']) = delta;
    out.([p 'cons']) = consVec;
    out.([p 'x']) = x;
    out.([p 'mu']) = mu;
    out.([p 'beta']) = betaTrue;
    out.([p 'gammaTrue']) = gammaTrue;
    out.([p 'll_trace']) = llTrace;
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
