function capture_randn_ref(repoRoot)
%CAPTURE_RANDN_REF  Gold-fixture capture for tests/test_matlab_rng.py.
%
% Produces tests/parity/fixtures/matlab_gold/randn_ref.mat (captured with
% MATLAB R2025b, 2026-10-05), used by
% tests/test_matlab_rng.py::TestMatlabRandnReference::test_known_divergence_from_matlab_randn.
%
% Saves the first 20 draws of MATLAB's ``randn`` (Ziggurat) under the
% repo-wide ``rng(42)`` convention, as the column vector ``r``.  The Python
% test documents that nstat.extras.matlab_rng.MatlabRNG.randn (Box-Muller)
% does NOT reproduce this stream bit-for-bit.
%
% Usage (from MATLAB):
%     capture_randn_ref('/path/to/nstat-python')

if nargin < 1 || isempty(repoRoot)
    repoRoot = pwd;
end
fixtureRoot = fullfile(repoRoot, 'tests', 'parity', 'fixtures', 'matlab_gold');
if ~exist(fixtureRoot, 'dir')
    mkdir(fixtureRoot);
end

rng(42); %#ok<RNG>
r = randn(20, 1); %#ok<NASGU>
save(fullfile(fixtureRoot, 'randn_ref.mat'), 'r');
fprintf('  [randn_ref] wrote %s\n', fullfile(fixtureRoot, 'randn_ref.mat'));
end
