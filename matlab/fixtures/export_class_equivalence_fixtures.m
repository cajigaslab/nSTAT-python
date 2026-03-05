function export_class_equivalence_fixtures(outRoot)
% Export deterministic MATLAB gold fixtures for class-level parity checks.
% Usage:
%   export_class_equivalence_fixtures('tests/parity/fixtures/matlab_gold/classes')

if nargin < 1 || isempty(outRoot)
    outRoot = fullfile('tests','parity','fixtures','matlab_gold','classes');
end
if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

cases = {'signalobj_basic', 'covariate_basic', 'nstcoll_basic'};
for i = 1:numel(cases)
    caseId = cases{i};
    caseDir = fullfile(outRoot, caseId);
    if exist(caseDir, 'dir') ~= 7
        mkdir(caseDir);
    end

    switch caseId
        case 'signalobj_basic'
            time = [0;1;2];
            data = [1;2;3];
            obj = SignalObj(time, data, 'sig');
            expectedDimension = obj.dimension;
            expectedDataShape = size(obj.data);
            save(fullfile(caseDir, 'case.mat'), 'expectedDimension', 'expectedDataShape');

        case 'covariate_basic'
            time = [0;1;2];
            data = [0.1;0.2;0.3];
            obj = Covariate(time, data, 'cov');
            expectedDimension = obj.dimension;
            expectedDataShape = size(obj.data);
            save(fullfile(caseDir, 'case.mat'), 'expectedDimension', 'expectedDataShape');

        case 'nstcoll_basic'
            s1 = nspikeTrain([0.1;0.4]);
            s2 = nspikeTrain(0.2);
            obj = nstColl({s1, s2});
            expectedNumSpikeTrains = obj.numSpikeTrains;
            save(fullfile(caseDir, 'case.mat'), 'expectedNumSpikeTrains');
    end
end

fprintf('Class fixtures exported to %s\n', outRoot);
end
