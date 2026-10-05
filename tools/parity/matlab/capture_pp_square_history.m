function capture_pp_square_history()
%CAPTURE_PP_SQUARE_HISTORY  Gold fixture: PPAF filters with a square history
% and the History.computeHistory window rule.
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
% dx = 2 states, delta = 1 ms unless noted, rng(42) synthetic inputs (latent AR(1) path,
% Bernoulli spikes), non-symmetric gamma = -0.2 - 0.8*rand(nW, C):
%   pdfl_pois_sq     PPDecodeFilterLinear      poisson,  C = 3, nW = 3, N = 150
%   pdfl_binom_sq    PPDecodeFilterLinear      binomial, C = 4, nW = 4, N = 150
%   pfis_pois_sq     PP_fixedIntervalSmoother  poisson,  C = 3, nW = 3, N = 150, lags = 1
%   pdfl_pois_ctrl   PPDecodeFilterLinear      poisson,  C = 3, nW = 2, N = 150 (control, nW ~= C)
%   estep_pois_sq    PP_EStep                  poisson,  C = 3, nW = 3, N = 150
%   estep_pois_NeqC  PP_EStep                  poisson,  C = 6, nW = 2, N = 6   (N == C)
%   estep_binom_C1   PP_EStep                  binomial, C = 1, nW = 2, N = 150 (HkAll is N x nW)
%   pdfl_pois_offgrid  PPDecodeFilterLinear    poisson,  C = 3, nW = 3, N = 150,
%                    windowTimes [0 1.5 4 6.5] ms (edges off the 1 ms grid)
%   pdfl_pois_colon  PPDecodeFilterLinear      poisson,  C = 3, nW = 10, N = 150,
%                    windowTimes = 0:delta:(9+1)*delta.  Pins MATLAB's rounding
%                    of these edges only: a real PP_EM / PPLFP_EM default call
%                    with numel(gamma) = 9 would pass a 9-row gamma for these
%                    10 windows; this case uses a 10-row gamma.
%   pdfl_binom_delta2  PPDecodeFilterLinear    binomial, C = 3, nW = 3, N = 150,
%                    delta = 2 ms, windowTimes [0 2 4 10] ms
% The last three pin how History.computeHistory turns windowTimes into lags:
% window [t(i), t(i+1)] sums the spikes ceil(t(i)*sampleRate)+1 ..
% ceil(t(i+1)*sampleRate) samples back, sampleRate = 1/delta.  They were
% appended after the first seven, so the rng(42) draws (and data) of the
% first seven are unchanged.  Every case keeps C ~= dx: MATLAB
% PPDecodeFilterLinear transposes a square (dx == C) beta, which Python does not.
%
% Saved per case (prefix <name>_): inputs A, Q, dN, mu, beta, fitType, gamma,
% windowTimes, delta, x0, Pi0 (PP_EStep: Px0), lags (smoother), HkAll and
% sizes = [N nW C dx]; outputs x_p, W_p, x_u, W_u (PPDecodeFilterLinear),
% x_pLag, W_pLag, x_uLag, W_uLag (PP_fixedIntervalSmoother) or x_K, W_K
% (PP_EStep; its logll is not captured here -- the square-history logll is in
% pp_estep.mat c5 / c6).
%
% Appended after the ten cases (fields outside case_names, so every earlier
% field is unchanged):
%   emdef_*   PP_EM's / PPLFP_EM's default history: with windowTimes = [] and
%             a non-zero gamma, the repaired MATLAB (B9) builds one window per
%             history coefficient, windowTimes = 0:delta:size(gamma,1)*delta,
%             and HkAll(:,:,k) = History(windowTimes,0,maxTime).computeHistory(
%             nspikeTrain((find(dN(k,:)==1)-1)*delta)).dataToMatrix (PPLFP_EM;
%             PP_EM's nspikeTrain(...,'',delta) is the same object at 1 ms).
%             Those lines are reproduced here (the EMs continue into a
%             Monte-Carlo EM that a gold cannot pin).  gamma is 8 x 2 and
%             delta = 1 ms, so 8 windows.  (MATLAB master's rule
%             0:delta:(length(gamma)+1)*delta gave 9 windows for 8 rows.)
%   colon_*   MATLAB a:d:b outputs (colon_v{i} = colon_a(i):colon_d(i):colon_b(i))
%             for the default-edge family 0:d:(m+1)*d (7 deltas x m = 0..40)
%             and 200 random signed triples, pinning nstat.core._matlab_colon_exact.
% Appended after colon_* (so every earlier field and rng draw is unchanged):
%   pp2ms_*   PP_EM's history at delta = 2 ms (repaired C6): emdef_dN with
%             windowTimes [0 4 10 20] ms, each train built as
%             nspikeTrain(t, '', delta) (PointProcessEM.m, repaired).
%   b1sq_*    PPDecodeFilterLinear with ns == C == 2 and a non-symmetric
%             (ns x C) beta, no history (repaired B1: a square beta is no
%             longer transposed).
%   pphf_*    PPHybridFilterLinear with history (repaired B2: it errored on
%             any windowTimes): two identical models, windowTimes
%             [0 2 5 10] ms and a shared 3 x 1 gamma over C = 3 cells, plus
%             PPDecodeFilterLinear on the same inputs (pphf_pdfl_*).
%
% HkAll is the history tensor each function consumes: for the two filters it
% is rebuilt exactly as PPAF.m builds it internally from windowTimes
% (History.computeHistory on nspikeTrain.resample(1/delta)); for PP_EStep it
% is built as PointProcessEM.PP_EM builds it and passed in.  MATLAB keeps
% HkAll as N x nW x C, which it stores as N x nW when C == 1.
%
% The MATLAB nSTAT checkout is read from the NSTAT_MATLAB_PATH environment
% variable (required).

matlabRepo = getenv('NSTAT_MATLAB_PATH');
if isempty(matlabRepo)
    error('capture:noMatlabPath', 'Set NSTAT_MATLAB_PATH to the MATLAB nSTAT checkout.');
end
here = fileparts(mfilename('fullpath'));
outMat = fullfile(here, '..', '..', '..', 'tests', 'parity', 'fixtures', ...
                  'matlab_gold', 'pp_square_history.mat');

addpath(matlabRepo);
addpath(genpath(fullfile(matlabRepo, 'libraries')));

rng(42);
dx = 2;
A   = [0.98 0.02; -0.03 0.96];
Q   = diag([0.02 0.015]);
x0  = [0.2; -0.1];
Pi0 = diag([0.05 0.08]);

cases = struct( ...
    'name',        {'pdfl_pois_sq', 'pdfl_binom_sq', 'pfis_pois_sq', 'pdfl_pois_ctrl', ...
                    'estep_pois_sq', 'estep_pois_NeqC', 'estep_binom_C1', ...
                    'pdfl_pois_offgrid', 'pdfl_pois_colon', 'pdfl_binom_delta2'}, ...
    'func',        {'PPDecodeFilterLinear', 'PPDecodeFilterLinear', 'PP_fixedIntervalSmoother', ...
                    'PPDecodeFilterLinear', 'PP_EStep', 'PP_EStep', 'PP_EStep', ...
                    'PPDecodeFilterLinear', 'PPDecodeFilterLinear', 'PPDecodeFilterLinear'}, ...
    'fitType',     {'poisson', 'binomial', 'poisson', 'poisson', 'poisson', 'poisson', 'binomial', ...
                    'poisson', 'poisson', 'binomial'}, ...
    'C',           {3, 4, 3, 3, 3, 6, 1, 3, 3, 3}, ...
    'N',           {150, 150, 150, 150, 150, 6, 150, 150, 150, 150}, ...
    'delta',       {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002}, ...
    'windowTimes', {[0 0.001 0.002 0.004], [0 0.001 0.002 0.004 0.008], [0 0.001 0.002 0.004], ...
                    [0 0.001 0.003], [0 0.001 0.002 0.004], [0 0.001 0.002], [0 0.001 0.003], ...
                    [0 0.0015 0.004 0.0065], 0:0.001:(9+1)*0.001, [0 0.002 0.004 0.010]}, ...
    'mu',          {[-2.2; -1.9; -2.5], [-1.6; -1.3; -1.9; -1.5], [-2.2; -1.9; -2.5], ...
                    [-2.2; -1.9; -2.5], [-2.2; -1.9; -2.5], -0.4 * ones(6, 1), -1.3, ...
                    [-2.0; -1.7; -2.3], [-2.0; -1.7; -2.3], [-1.5; -1.2; -1.8]});
lags = 1;

out = struct();
for i = 1:numel(cases)
    cs = cases(i);
    C = cs.C; N = cs.N; wt = cs.windowTimes; nW = numel(wt) - 1; delta = cs.delta;
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

% --- PP_EM / PPLFP_EM default history edges (see header) -----------------
emC = 2; emN = 200; delta = 0.001;
gamma = -0.2 - 0.8 * rand(8, emC);
dN = double(rand(emC, emN) < 0.15);
windowTimes = 0:delta:size(gamma,1)*delta;                     % repaired B9 default rule
minTime = 0; maxTime = (size(dN,2)-1)*delta;
histObj = History(windowTimes,minTime,maxTime);
HkAll = zeros(emN, numel(windowTimes)-1, emC);
for k = 1:emC
    nst = nspikeTrain((find(dN(k,:)==1)-1)*delta);
    nst.setMinTime(minTime);
    nst.setMaxTime(maxTime);
    HkAll(:,:,k) = histObj.computeHistory(nst).dataToMatrix;
end
out.emdef_dN = dN;
out.emdef_gamma = gamma;
out.emdef_delta = delta;
out.emdef_windowTimes = windowTimes;
out.emdef_HkAll = HkAll;
fprintf('  [emdef] EM default edges, size(gamma,1)=%d: %d windows, size(HkAll)=%s\n', ...
    size(gamma,1), numel(windowTimes)-1, mat2str(size(HkAll)));

% --- MATLAB colon outputs (see header) ------------------------------------
colonA = []; colonD = []; colonB = []; colonV = {};
for d = [0.001 0.002 0.0005 0.003 0.004 0.0001 0.01]
    for m = 0:40
        colonA(end+1) = 0; colonD(end+1) = d; colonB(end+1) = (m+1)*d; %#ok<AGROW>
        colonV{end+1} = 0:d:(m+1)*d; %#ok<AGROW>
    end
end
for i = 1:200
    a = (rand-0.5) * 10^(randi([-3 2]));
    if rand < 0.3, a = 0; end
    d = (0.05 + rand) * 10^(randi([-4 1]));
    if rand < 0.2, d = -d; end
    n = randi([0 120]);
    r = rand;
    if r < 0.4
        b = a + n*d;
    elseif r < 0.6
        b = a + (n + 0.5)*d;
    elseif r < 0.8
        b = a + n*d + randi([-4 4])*eps(a + n*d);
    else
        b = a + (n + 1e-11*(rand-0.5))*d;
    end
    colonA(end+1) = a; colonD(end+1) = d; colonB(end+1) = b; %#ok<AGROW>
    colonV{end+1} = a:d:b; %#ok<AGROW>
end
out.colon_a = colonA;
out.colon_d = colonD;
out.colon_b = colonB;
out.colon_v = colonV;
fprintf('  [colon] %d MATLAB colon outputs\n', numel(colonV));

% --- PP_EM history at delta = 2 ms (see header) ---------------------------
dN2 = out.emdef_dN; delta2 = 0.002; wt2 = [0 0.004 0.010 0.020];
maxTime2 = (size(dN2,2)-1)*delta2;
histObj = History(wt2, 0, maxTime2);
HkAll2 = zeros(size(dN2,2), numel(wt2)-1, size(dN2,1));
for k = 1:size(dN2,1)
    nst = nspikeTrain((find(dN2(k,:)==1)-1)*delta2, '', delta2);   % PointProcessEM.m PP_EM (repaired)
    nst.setMinTime(0);
    nst.setMaxTime(maxTime2);
    HkAll2(:,:,k) = histObj.computeHistory(nst).dataToMatrix;
end
out.pp2ms_dN = dN2;
out.pp2ms_delta = delta2;
out.pp2ms_windowTimes = wt2;
out.pp2ms_HkAll = HkAll2;
fprintf('  [pp2ms] PP_EM history at delta = 2 ms, size(HkAll)=%s\n', mat2str(size(HkAll2)));

% --- PPDecodeFilterLinear with ns == C (see header) -----------------------
nsq = 2; Nsq = 300;
betaSq = [0.9 -0.4; 0.3 0.7];                                   % ns x C, non-symmetric
muSq = log([30; 40] * 0.001);
x = zeros(nsq, Nsq); xPrev = x0; cholQ = chol(Q, 'lower');
for k = 1:Nsq
    xPrev = A * xPrev + cholQ * randn(nsq, 1);
    x(:, k) = xPrev;
end
dNsq = double(rand(nsq, Nsq) < min(exp(muSq + betaSq' * x), 1));
[x_p, W_p, x_u, W_u] = nstat.decoding.PPAF.PPDecodeFilterLinear( ...
    A, Q, dNsq, muSq, betaSq, 'poisson', 0.001, [], [], x0, Pi0);
out.b1sq_A = A; out.b1sq_Q = Q; out.b1sq_dN = dNsq; out.b1sq_mu = muSq; out.b1sq_beta = betaSq;
out.b1sq_x0 = x0; out.b1sq_Pi0 = Pi0; out.b1sq_delta = 0.001;
out.b1sq_x_p = x_p; out.b1sq_W_p = W_p; out.b1sq_x_u = x_u; out.b1sq_W_u = W_u;
fprintf('  [b1sq] PPDecodeFilterLinear ns = C = 2, spikes per cell = %s\n', mat2str(sum(dNsq, 2)'));

% --- PPHybridFilterLinear with history (see header) -----------------------
Cph = 3; Nph = 300; deltaPh = 0.001;
betaPh = [0.9 -0.4 0.2; 0.3 0.7 -0.5];
muPh = log([30; 40; 25] * deltaPh);
wtPh = [0 0.002 0.005 0.010]; gPh = [-0.8; -0.4; -0.2];       % shared numWindows x 1 gamma
x = zeros(2, Nph); xPrev = x0;
for k = 1:Nph
    xPrev = A * xPrev + cholQ * randn(2, 1);
    x(:, k) = xPrev;
end
dNph = double(rand(Cph, Nph) < min(exp(muPh + betaPh' * x), 1));
pij = [0.9 0.1; 0.1 0.9]; Mu0 = [0.5; 0.5];
[S_est, X, W, MU_u, X_s, W_s, pNGivenS] = nstat.decoding.PPHF.PPHybridFilterLinear( ...
    {A, A}, {Q, Q}, pij, Mu0, dNph, muPh, betaPh, 'poisson', deltaPh, gPh, wtPh, {x0, x0}, {Pi0, Pi0});
[~, ~, xuPh, WuPh] = nstat.decoding.PPAF.PPDecodeFilterLinear( ...
    A, Q, dNph, muPh, betaPh, 'poisson', deltaPh, gPh, wtPh, x0, Pi0);
out.pphf_A = A; out.pphf_Q = Q; out.pphf_p_ij = pij; out.pphf_Mu0 = Mu0; out.pphf_dN = dNph;
out.pphf_mu = muPh; out.pphf_beta = betaPh; out.pphf_binwidth = deltaPh; out.pphf_gamma = gPh;
out.pphf_windowTimes = wtPh; out.pphf_x0 = x0; out.pphf_Pi0 = Pi0;
out.pphf_S_est = S_est; out.pphf_X = X; out.pphf_W = W; out.pphf_MU_u = MU_u;
out.pphf_X_s1 = X_s{1}; out.pphf_X_s2 = X_s{2}; out.pphf_W_s1 = W_s{1}; out.pphf_W_s2 = W_s{2};
out.pphf_pNGivenS = pNGivenS;
out.pphf_pdfl_x_u = xuPh; out.pphf_pdfl_W_u = WuPh;
fprintf('  [pphf] PPHybridFilterLinear with history, max|X - x_u(PDFL)| = %g\n', max(abs(X(:) - xuPh(:))));

out.case_names = {cases.name};
out.matlab_version = version;
out.matlab_source_note = ['emdef_* / pp2ms_* / b1sq_* / pphf_* captured from the repaired MATLAB ' ...
                          'nSTAT fix/pp-em @ a457b54 (pending upstream merge); the ten cases and ' ...
                          'colon_* are bit-identical to the MATLAB master capture'];

save(outMat, '-struct', 'out', '-v7');
fprintf('Saved gold fixture: %s\n', outMat);
end
