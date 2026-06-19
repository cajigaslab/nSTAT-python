function export_pplfp_gold_fixtures(repoRoot, matlabRepoRoot)
% export_pplfp_gold_fixtures
% =========================================================================
% Recipes for the 4 PPLFP-family MATLAB gold fixtures in
% tests/parity/fixtures/matlab_gold/:
%
%   pplfp_SE.mat       — PPLFP_ComputeParamStandardErrors output (SE struct)
%   pplfp_EStep.mat    — PPLFP_EStep output (x_K, W_K, logll, sufficient stats)
%   pplfp_MStep.mat    — PPLFP_MStep output (betahat_new)
%   pplfp_EM.mat       — PPLFP_EM driver output (xKFinal)
%
% These four fixtures were originally captured ad-hoc by v9 iter ~38-40 via
% direct `/opt/homebrew/bin/matlab -batch` snippets that were never committed
% to the repo. This script reconstructs the capture recipe to the best of the
% Python-side knowledge in `tools/parity/numerical_drift.py` (recipes
% `_recipe_pplfp_estep`, `_recipe_pplfp_mstep`, `_recipe_pplfp_em`,
% `_recipe_pplfp_se_alpha`) and the per-fixture field set expected by the spec
% (`parity/numerical_drift_spec.yml`).
%
% USAGE
% -----
%   export_pplfp_gold_fixtures('/Users/iahncajigas/projects/nstat-python', ...
%                              '/Users/iahncajigas/projects/nstat');
%
% WARNING / TODO
% --------------
% The four committed .mat fixtures pass the drift detector (36/36 PASS as of
% v10 iter 47), but the exact original capture inputs (random seed,
% initialisation matrices, mcIter, etc.) for the SE / MStep / EM fixtures
% were never written down. The recipes below use:
%   * rng(42) — the v9-era convention (memory).
%   * Small canonical dims (K = 10 or 30 steps, 2 cells, 1-state SSM).
%   * Convergence-friendly initialisations.
%
% If a recapture run produces values that drift outside the spec tolerances,
% the original capture inputs differed and this script needs revising. DO NOT
% overwrite the committed .mat files unless you have separately verified the
% downstream test suite (`tests/test_matlab_gold_fixtures.py`) still passes.
%
% Mark each fixture as TODO in the per-fixture function header until the
% recipe has been re-verified by an end-to-end MATLAB run.

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

export_pplfp_EStep_fixture(fixtureRoot);
export_pplfp_MStep_fixture(fixtureRoot);
export_pplfp_EM_fixture(fixtureRoot);
export_pplfp_SE_fixture(fixtureRoot);

end


function export_pplfp_EStep_fixture(fixtureRoot) %#ok<DEFNU>
% pplfp_EStep.mat
% ---------------
% Inputs: A, Q, C, R, y, alpha, dN, mu, beta, gamma, HkAll, x0, Px0,
%         fitType, delta
% Outputs (saved): x_K, W_K, plus all inputs.
%
% TODO (v10 iter 47): canonical inputs for this fixture are not committed.
% The Python recipe `_recipe_pplfp_estep` expects fields:
%   A, Q, C, R, y, alpha, dN, mu, beta, fitType, delta, gamma, HkAll,
%   x0, Px0, x_K (baseline).
% A v9-style canonical capture would use:
%   rng(42); K = 10; numCells = 2; nx = 2;
%   A = 0.95*eye(nx); Q = 0.01*eye(nx);
%   C = randn(numCells, nx); R = 0.1*eye(numCells);
%   alpha = zeros(nx, 1); x0 = zeros(nx, 1); Px0 = 0.1*eye(nx);
%   y = randn(numCells, K);
%   dN = double(rand(numCells, K) < 0.1);
%   mu = -2.5*ones(numCells, 1); beta = 0.1*randn(nx, numCells);
%   gamma = zeros(numCells, 1); HkAll = zeros(K, 1, numCells);
%   fitType = 'poisson'; delta = 1.0;
% Then:
%   [x_K, W_K, logll, ExpectationSums] = PPLFP_EStep(A, Q, C, R, y, alpha, ...
%       dN, mu, beta, fitType, delta, gamma, HkAll, x0, Px0);
%
% Until verified end-to-end, this stub is a NO-OP — it does not overwrite the
% committed fixture. Remove the early return to enable.
fprintf(['  [TODO] export_pplfp_EStep_fixture: capture recipe unverified; ' ...
         'skipping (will not overwrite committed fixture)\n']);
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_pplfp_MStep_fixture(fixtureRoot) %#ok<DEFNU>
% pplfp_MStep.mat
% ---------------
% Saved fields: all EStep inputs PLUS sufficient stats from EStep run, PLUS
% updated parameters from MStep — Ahat, Qhat, Chat, Rhat, alphahat, muhat_new,
% betahat_new (baseline), gammahat_new, x0hat, Px0hat.
%
% TODO (v10 iter 47): see header. Recipe sketch:
%   [same EStep setup as export_pplfp_EStep_fixture]
%   [run PPLFP_EStep to get x_K, W_K, ExpectationSums]
%   [pack inputs + x_K, W_K, ExpectationSums]
%   result = PPLFP_MStep(dN, y, x_K, W_K, x0, Px0, ExpectationSums, ...
%                        fitType, mu, beta, gamma, [], HkAll, ...
%                        'NewtonRaphson');
%   betahat_new = result.betahat_new;
%   save(.., 'betahat_new', .., '-v7');
%
% Drift detector reads `betahat_new` field; tolerance is Case-C relaxed
% (rtol=1e+1, atol=1e+1) because Newton-Raphson uses MC sampling that
% does not match MATLAB's normrnd stream.
fprintf(['  [TODO] export_pplfp_MStep_fixture: capture recipe unverified; ' ...
         'skipping (will not overwrite committed fixture)\n']);
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_pplfp_EM_fixture(fixtureRoot) %#ok<DEFNU>
% pplfp_EM.mat
% ------------
% Driver fixture for full PPLFP_EM. Saved fields include initial guesses
% (Ahat0, Qhat0, Chat0, Rhat0, alphahat0) and final smoothed state
% xKFinal (the comparison baseline).
%
% TODO (v10 iter 47): see header. Recipe sketch:
%   rng(42); K = 30; numCells = 2; nx = 2;
%   y = randn(numCells, K);
%   dN = double(rand(numCells, K) < 0.1);
%   Ahat0 = 0.9*eye(nx); Qhat0 = 0.01*eye(nx);
%   Chat0 = randn(numCells, nx); Rhat0 = 0.1*eye(numCells);
%   alphahat0 = zeros(nx, 1);
%   mu = -2.5*ones(numCells, 1); beta = 0.1*randn(nx, numCells);
%   x0 = zeros(nx, 1); Px0 = 0.1*eye(nx);
%   Constraints = PPLFP_EMCreateConstraints('EstimateA', 1, 'AhatDiag', 0, ...
%       'QhatDiag', 1, 'RhatDiag', 1, 'Estimatex0', 0, 'EstimatePx0', 0, ...
%       'mcIter', 50);
%   [xKFinal, ...] = PPLFP_EM(y, dN, Ahat0, Qhat0, Chat0, Rhat0, alphahat0, ...
%       mu, beta, 'poisson', 1.0, x0, Px0, Constraints, 'NewtonRaphson');
%   mcIter = 50;
%   save(.., 'xKFinal', 'Ahat0', .., 'mcIter', '-v7');
fprintf(['  [TODO] export_pplfp_EM_fixture: capture recipe unverified; ' ...
         'skipping (will not overwrite committed fixture)\n']);
return  %#ok<UNRCH>
end %#ok<DEFNU>


function export_pplfp_SE_fixture(fixtureRoot) %#ok<DEFNU>
% pplfp_SE.mat
% ------------
% Captures the SE struct returned by PPLFP_ComputeParamStandardErrors after
% a full EM converged run. Only the SE.alpha field is currently compared
% (rest are RNG-sensitive — see matlab_defects.yml entry pplfp-se-mc-drift).
%
% TODO (v10 iter 47): see header. Recipe sketch:
%   [same EM setup as export_pplfp_EM_fixture]
%   [run PPLFP_EM to get xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat,
%    alphahat, muhat_new, betahat_new, gammahat_new, x0hat, Px0hat,
%    ExpectationSums]
%   [SE, Pvals, nTerms] = PPLFP_ComputeParamStandardErrors(y, dN, ...
%       xKFinal, WKFinal, Ahat, Qhat, Chat, Rhat, alphahat, x0hat, ...
%       Px0hat, ExpectationSums, 'poisson', muhat_new, betahat_new, ...
%       gammahat_new, [], HkAll);
%   save(.., 'SE', .., '-v7');
fprintf(['  [TODO] export_pplfp_SE_fixture: capture recipe unverified; ' ...
         'skipping (will not overwrite committed fixture)\n']);
return  %#ok<UNRCH>
end %#ok<DEFNU>
