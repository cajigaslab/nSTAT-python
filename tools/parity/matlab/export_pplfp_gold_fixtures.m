function export_pplfp_gold_fixtures(repoRoot, matlabRepoRoot)
% export_pplfp_gold_fixtures
% =========================================================================
% Recipes for the 4 PPLFP-family MATLAB gold fixtures in
% tests/parity/fixtures/matlab_gold/:
%
%   pplfp_SE.mat       — PPLFP_ComputeParamStandardErrors output (SE struct)
%   pplfp_EStep.mat    — PPLFP_EStep output (x_K, W_K, logll, sufficient stats)
%   pplfp_MStep.mat    — PPLFP_MStep output (betahat_new + closed-form updates)
%   pplfp_EM.mat       — PPLFP_EM driver output (xKFinal, full final params)
%
% v11 iter 49 reconstruction
% --------------------------
% The 4 .mat files were originally captured ad-hoc by v9 iter ~38-40 via
% direct `/opt/homebrew/bin/matlab -batch` snippets that were never committed.
% The committed fixtures pass the v10 drift detector (4/4 PASS), but the exact
% capture inputs (seed, init matrices, mcIter, dN realisation) were unknown.
%
% Per v11 iter 49 plan: rather than guess inputs from scratch, every input
% the MATLAB function needs is itself already serialised into each committed
% .mat (A, Q, C, R, y, dN, alpha, mu, beta, gamma, x0, Px0, HkAll, fitType,
% delta — plus, for EM, the Ahat0/Qhat0/Chat0/Rhat0/alphahat0 inits and
% mcIter). These recipes therefore re-load each fixture's inputs and re-run
% the matching MATLAB function with `rng(42)` set before any MC path.
%
% Re-baselining policy
% --------------------
% If a recapture run produces values that drift outside the spec tolerances
% but Python's drift detector still PASSes on the new .mat, the new outputs
% are committed as the fresh baseline and `pplfp-<name>-recipe-rebaseline`
% Case-D entries should be added to `parity/matlab_defects.yml`. If Python
% then FAILS drift on the new .mat, do not commit — investigate first.
%
% USAGE
% -----
%   export_pplfp_gold_fixtures('/Users/iahncajigas/projects/nstat-python', ...
%                              '/Users/iahncajigas/projects/nstat');

if nargin < 1 || isempty(repoRoot)
    error('repoRoot is required');
end
if nargin < 2 || isempty(matlabRepoRoot)
    matlabRepoRoot = fullfile(fileparts(repoRoot), 'nSTAT');
end

repoRoot = char(repoRoot);
matlabRepoRoot = char(matlabRepoRoot);

addpath(matlabRepoRoot);
addpath(genpath(fullfile(matlabRepoRoot, 'libraries')));

fixtureRoot = fullfile(repoRoot, 'tests', 'parity', 'fixtures', 'matlab_gold');
if ~exist(fixtureRoot, 'dir')
    mkdir(fixtureRoot);
end

% Each capture is wrapped to isolate failures: an upstream bug in one
% function (e.g. PPLFP_EM's internal `K = size(dN,1)` mis-sizing of HkAll
% — see Case D ledger entry pplfp-em-recipe-rebaseline) must not block the
% other three captures.
captures = { ...
    @export_pplfp_EStep_fixture, 'EStep'; ...
    @export_pplfp_MStep_fixture, 'MStep'; ...
    @export_pplfp_EM_fixture,    'EM'; ...
    @export_pplfp_SE_fixture,    'SE'};
for ci = 1:size(captures, 1)
    fn = captures{ci, 1};
    label = captures{ci, 2};
    try
        fn(fixtureRoot);
    catch ME
        fprintf('  [%s] FAILED: %s — keeping committed .mat untouched.\n', ...
            label, ME.message);
    end
end

end


function export_pplfp_EStep_fixture(fixtureRoot)
% pplfp_EStep.mat
% ---------------
% Inputs (loaded from committed fixture): A, Q, C, R, y, alpha, dN, mu, beta,
% gamma, HkAll, x0, Px0, fitType, delta.
% Outputs (re-run): x_K, W_K, logll, ExpectationSums fields (Sx0, Sx0x0,
% Sxkm1xk, Sxkm1xkm1, Sxkxk, Sxkxkm1, Sxkyk, Sykyk, sumXkTerms, sumYkTerms,
% sumPPll).
%
% PPLFP_EStep is fully deterministic (no `normrnd`/`rand` calls in the body),
% so byte-equivalent reproduction is expected when called with identical
% inputs.

fixturePath = fullfile(fixtureRoot, 'pplfp_EStep.mat');
fprintf('  [EStep] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

% Preserve original input shapes/dtypes; pull into named locals so the final
% `save` call is explicit.
A = double(data.A);
Q = double(data.Q);
C = double(data.C);
R = double(data.R);
y = double(data.y);
alpha = double(data.alpha(:));
dN = double(data.dN);
mu = double(data.mu(:));
beta = double(data.beta);
fitType = char(string(data.fitType));
delta = double(data.delta);
gamma = double(data.gamma(:));
HkAll = double(data.HkAll);
x0 = double(data.x0(:));
Px0 = double(data.Px0);

% Mirror Python recipe: if gamma is all-zero, the original capture had no
% history term — pass empty (function default substitutes scalar 0).
% PPLFP_EStep line 2127 (sumPPll vectorised loop) reads `gamma` raw — if
% empty it skips the repmat default and `gammaC'*Hk` fails on dim
% mismatch. Use scalar 0 instead of [] for the zero-history case.
if isempty(gamma) || max(abs(gamma)) == 0
    gammaArg = 0;
    HkAllArg = zeros(size(HkAll, 1), 1, size(HkAll, 3));
else
    gammaArg = gamma;
    HkAllArg = HkAll;
end

rng(42); %#ok<RNG> deterministic guard even though EStep has no RNG
[x_K, W_K, logll, ExpectationSums] = ...
    nstat.decoding.PPLFP.PPLFP_EStep(A, Q, C, R, y, alpha, dN, mu, beta, ...
        fitType, delta, gammaArg, HkAllArg, x0, Px0);

Sx0 = ExpectationSums.Sx0;
Sx0x0 = ExpectationSums.Sx0x0;
Sxkm1xk = ExpectationSums.Sxkm1xk;
Sxkm1xkm1 = ExpectationSums.Sxkm1xkm1;
Sxkxk = ExpectationSums.Sxkxk;
Sxkxkm1 = ExpectationSums.Sxkxkm1;
Sxkyk = ExpectationSums.Sxkyk;
Sykyk = ExpectationSums.Sykyk;
sumXkTerms = ExpectationSums.sumXkTerms;
sumYkTerms = ExpectationSums.sumYkTerms;
sumPPll = ExpectationSums.sumPPll;

save(fixturePath, ...
    'A', 'Q', 'C', 'R', 'y', 'alpha', 'dN', 'mu', 'beta', 'gamma', ...
    'HkAll', 'x0', 'Px0', 'fitType', 'delta', ...
    'x_K', 'W_K', 'logll', ...
    'Sx0', 'Sx0x0', 'Sxkm1xk', 'Sxkm1xkm1', 'Sxkxk', 'Sxkxkm1', ...
    'Sxkyk', 'Sykyk', 'sumXkTerms', 'sumYkTerms', 'sumPPll', '-v7');
fprintf('  [EStep] saved %s\n', fixturePath);
end


function export_pplfp_MStep_fixture(fixtureRoot)
% pplfp_MStep.mat
% ---------------
% Inputs (from committed fixture): A, Q, C, R, y, alpha, dN, mu, beta, gamma,
% HkAll, x0, Px0, fitType, delta, plus filter outputs x_K, W_K and the
% sufficient-stats inputs gammahat_in (legacy field).
%
% MStep needs ExpectationSums; the committed fixture does NOT carry it, so
% (matching the Python recipe) we recompute it via PPLFP_EStep first, then
% pass to PPLFP_MStep with MstepMethod='NewtonRaphson'.
%
% Outputs (re-run): Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new,
% gammahat_new, x0hat, Px0hat.
%
% PPLFP_MStep's NewtonRaphson branch calls `normrnd` for McExp=50 MC samples
% per inner iteration; output is RNG-dependent. `rng(42)` is set for
% deterministic re-runs but the absolute values may drift from the original
% capture's stream.

fixturePath = fullfile(fixtureRoot, 'pplfp_MStep.mat');
fprintf('  [MStep] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
Q = double(data.Q);
C = double(data.C);
R = double(data.R);
y = double(data.y);
alpha = double(data.alpha(:));
dN = double(data.dN);
mu = double(data.mu(:));
beta = double(data.beta);
fitType = char(string(data.fitType));
delta = double(data.delta);
gamma = double(data.gamma(:));
HkAll = double(data.HkAll);
x0 = double(data.x0(:));
Px0 = double(data.Px0);
gammahat_in = double(data.gammahat_in);

% PPLFP_EStep line 2127 (sumPPll vectorised loop) reads `gamma` raw — if
% empty it skips the repmat default and `gammaC'*Hk` fails on dim
% mismatch. Use scalar 0 instead of [] for the zero-history case.
if isempty(gamma) || max(abs(gamma)) == 0
    gammaArg = 0;
    HkAllArg = zeros(size(HkAll, 1), 1, size(HkAll, 3));
else
    gammaArg = gamma;
    HkAllArg = HkAll;
end

rng(42); %#ok<RNG>
[~, ~, ~, ExpectationSums] = ...
    nstat.decoding.PPLFP.PPLFP_EStep(A, Q, C, R, y, alpha, dN, mu, beta, ...
        fitType, delta, gammaArg, HkAllArg, x0, Px0);

x_K = double(data.x_K);
W_K = double(data.W_K);

% Default constraints (mcIter=1000 by default; lower to keep capture fast
% and to match the v9-era capture envelope).
constraints = nstat.decoding.PPLFP.PPLFP_EMCreateConstraints( ...
    1, 0, 1, 0, 1, 0, 1, 1, 0, 50, 0);

rng(42); %#ok<RNG>
[Ahat, Qhat, Chat, Rhat, alphahat, muhat_new, betahat_new, gammahat_new, ...
    x0hat, Px0hat] = nstat.decoding.PPLFP.PPLFP_MStep( ...
        dN, y, x_K, W_K, x0, Px0, ExpectationSums, fitType, ...
        mu, beta, gammaArg, [], HkAllArg, constraints, 'NewtonRaphson');

save(fixturePath, ...
    'A', 'Q', 'C', 'R', 'y', 'alpha', 'dN', 'mu', 'beta', 'gamma', ...
    'HkAll', 'x0', 'Px0', 'fitType', 'delta', ...
    'x_K', 'W_K', 'gammahat_in', ...
    'Ahat', 'Qhat', 'Chat', 'Rhat', 'alphahat', 'muhat_new', ...
    'betahat_new', 'gammahat_new', 'x0hat', 'Px0hat', '-v7');
fprintf('  [MStep] saved %s\n', fixturePath);
end


function export_pplfp_EM_fixture(fixtureRoot)
% pplfp_EM.mat
% ------------
% Inputs (from committed fixture): y, dN, Ahat0, Qhat0, Chat0, Rhat0,
% alphahat0, mu, beta, fitType, delta, x0, Px0, mcIter.
% Outputs (re-run): xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat,
% muhat, betahat, gammahat, x0hat, Px0hat, IC, SE, Pvals.
%
% EM is RNG-dependent (NewtonRaphson MC + optional Ikeda mvnrnd) AND
% iteration-sensitive (log-likelihood floating-point noise can change
% convergence-iter). `rng(42)` is set for determinism.
%
% v12 iter 57 status — BLOCKED by upstream regression
% ---------------------------------------------------
% Although cajigaslab/nSTAT#90 (PPLFP_EM `K=size(dN,1)` mis-sizing) is
% marked fixed upstream (commit 49a84d6), the fix actually broke a
% different code path: `PPLFP_Decode_update` lines 328/357 were changed
% from `HkAll(:,:,time_index)` to `squeeze(HkAll(time_index,:,:))`, but
% `PPLFP_EStep` line 273 still passes `Histtermperm = permute(HkAll,[2 3 1])`
% (which puts time on the 3rd dim, per the inline comment "expects History
% with time on 3rd index"). The new slice indexes the FIRST dim of the
% permuted tensor (hist_cols), producing wrong-shape Histterm and a matmul
% failure at line 368. Result: ALL FOUR PPLFP recipes (EStep/MStep/EM/SE)
% now fail end-to-end on the upstream MATLAB checkout. The committed
% fixtures remain valid and Python drift continues to PASS against them.
% Re-filing upstream is queued for v13.

fixturePath = fullfile(fixtureRoot, 'pplfp_EM.mat');
fprintf('  [EM] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

y = double(data.y);
dN = double(data.dN);
Ahat0 = double(data.Ahat0);
Qhat0 = double(data.Qhat0);
Chat0 = double(data.Chat0);
Rhat0 = double(data.Rhat0);
alphahat0 = double(data.alphahat0(:));
mu = double(data.mu(:));
beta = double(data.beta);
fitType = char(string(data.fitType));
delta = double(data.delta);
x0 = double(data.x0(:));
Px0 = double(data.Px0);
mcIter = double(data.mcIter);

constraints = nstat.decoding.PPLFP.PPLFP_EMCreateConstraints( ...
    1, 0, 1, 0, 1, 0, 0, 0, 0, mcIter, 0);  % Estimatex0=0, EstimatePx0=0

% PPLFP_EM internally builds HkAll. Its `else` branch (no windowTimes)
% has the upstream bug `K=size(dN,1)` which mis-sizes HkAll's 3rd dim to
% numCells instead of time-steps. Use a single-window windowTimes so the
% History/computeHistory path runs and HkAll's K-dim is correct. This
% gives `gammahat` shape (1, numCells) — matching the committed fixture's
% gammahat slot.
K = size(dN, 2);
maxTime = (K - 1) * delta;
windowTimes = [0, maxTime];

rng(42); %#ok<RNG>
% Pass gamma = 0 (scalar) instead of [] — the EStep at PPLFP.m:2147 computes
% `gammaC' * Hk` after repmat, which collapses to a 0x0 * (1,2) matmul
% mismatch when gamma is empty. Scalar 0 broadcasts cleanly to (1,numCells)
% via the numel(gamma)==1 branch at PPLFP.m:2139. Discovered v13 iter 63
% post-#98-fix root-cause investigation (Python recipe was always passing
% gamma=0; the MATLAB capture script's gamma=[] was the parity mismatch).
[xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, muhat, betahat, ...
    gammahat, x0hat, Px0hat, IC, SE, Pvals] = ...
    nstat.decoding.PPLFP.PPLFP_EM(y, dN, Ahat0, Qhat0, Chat0, Rhat0, ...
        alphahat0, mu, beta, fitType, delta, 0, windowTimes, x0, Px0, ...
        constraints, 'NewtonRaphson');

save(fixturePath, ...
    'y', 'dN', 'Ahat0', 'Qhat0', 'Chat0', 'Rhat0', 'alphahat0', ...
    'mu', 'beta', 'fitType', 'delta', 'x0', 'Px0', 'mcIter', ...
    'xKFinal', 'WKFinal', 'Ahat', 'Qhat', 'Chat', 'Rhat', 'alphahat', ...
    'muhat', 'betahat', 'gammahat', 'x0hat', 'Px0hat', ...
    'IC', 'SE', 'Pvals', '-v7');
fprintf('  [EM] saved %s\n', fixturePath);
end


function export_pplfp_SE_fixture(fixtureRoot)
% pplfp_SE.mat
% ------------
% Inputs (from committed fixture): A, Q, C, R, y, alpha, dN, mu, beta, gamma,
% HkAll, x0, Px0, fitType, delta, plus EM-converged params (Ahat, Qhat,
% Chat, Rhat, alphahat, betahat_new, gammahat_new, muhat_new, x0hat, Px0hat)
% and smoothed state (xKFinal, WKFinal).
% Outputs (re-run): SE struct, Pvals struct, nTerms.
%
% PPLFP_ComputeParamStandardErrors uses Monte Carlo (mcIter samples) via
% normrnd for the observed-information block — output is RNG-dependent. Only
% SE.alpha is regressed (it's deterministic from N and Rhat); other fields
% carry the MC envelope per matlab_defects.yml entry pplfp-se-mc-drift.

fixturePath = fullfile(fixtureRoot, 'pplfp_SE.mat');
fprintf('  [SE] loading inputs from %s\n', fixturePath);
data = load(fixturePath);

A = double(data.A);
Q = double(data.Q);
C = double(data.C);
R = double(data.R);
y = double(data.y);
alpha = double(data.alpha(:));
dN = double(data.dN);
mu = double(data.mu(:));
beta = double(data.beta);
fitType = char(string(data.fitType));
delta = double(data.delta);
gamma = double(data.gamma(:));
HkAll = double(data.HkAll);
x0 = double(data.x0(:));
Px0 = double(data.Px0);
Ahat = double(data.Ahat);
Qhat = double(data.Qhat);
Chat = double(data.Chat);
Rhat = double(data.Rhat);
alphahat = double(data.alphahat(:));
betahat_new = double(data.betahat_new);
gammahat_new = double(data.gammahat_new(:));
muhat_new = double(data.muhat_new(:));
x0hat = double(data.x0hat(:));
Px0hat = double(data.Px0hat);
xKFinal = double(data.xKFinal);
WKFinal = double(data.WKFinal);

% Reconstitute ExpectationSums via PPLFP_EStep using the *EM-converged*
% params (Ahat, Qhat, Chat, Rhat, alphahat, betahat_new, muhat_new,
% gammahat_new) — this matches MATLAB's PPLFP_EM final-step call signature.
if isempty(gammahat_new) || max(abs(gammahat_new)) == 0
    gammaSEArg = 0;
    HkAllSEArg = zeros(size(HkAll, 1), 1, size(HkAll, 3));
else
    gammaSEArg = gammahat_new;
    HkAllSEArg = HkAll;
end

rng(42); %#ok<RNG>
[~, ~, ~, ExpectationSumsFinal] = ...
    nstat.decoding.PPLFP.PPLFP_EStep(Ahat, Qhat, Chat, Rhat, y, alphahat, ...
        dN, muhat_new, betahat_new, fitType, delta, gammaSEArg, HkAllSEArg, ...
        x0hat, Px0hat);

constraints = nstat.decoding.PPLFP.PPLFP_EMCreateConstraints( ...
    1, 0, 1, 0, 1, 0, 1, 1, 0, 500, 0);

rng(42); %#ok<RNG>
[SE, Pvals, nTerms] = ...
    nstat.decoding.PPLFP.PPLFP_ComputeParamStandardErrors( ...
        y, dN, xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, ...
        x0hat, Px0hat, ExpectationSumsFinal, fitType, muhat_new, ...
        betahat_new, gammaSEArg, [], HkAllSEArg, constraints);

save(fixturePath, ...
    'A', 'Q', 'C', 'R', 'y', 'alpha', 'dN', 'mu', 'beta', 'gamma', ...
    'HkAll', 'x0', 'Px0', 'fitType', 'delta', ...
    'Ahat', 'Qhat', 'Chat', 'Rhat', 'alphahat', 'betahat_new', ...
    'gammahat_new', 'muhat_new', 'x0hat', 'Px0hat', ...
    'xKFinal', 'WKFinal', 'SE', 'Pvals', 'nTerms', '-v7');
fprintf('  [SE] saved %s\n', fixturePath);
end
