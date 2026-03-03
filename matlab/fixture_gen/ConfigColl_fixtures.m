function ConfigColl_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for the ConfigColl class.
%
% MATLAB reference: ConfigColl.m constructor/add/get/set/toStructure/fromStructure

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'ConfigColl');
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end
    outputFile = fullfile(outputDir, 'basic.mat');
else
    outputDir = fileparts(outputFile);
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end
end

tc1 = TrialConfig({'Force', 'f_x'}, 2000, [.1 .2], -1, 2);
tc2 = TrialConfig({'Position', 'x'}, 2000, [.1 .2], -1, 2);
tcc = ConfigColl({tc1, tc2});

initial_numConfigs = tcc.numConfigs;
initial_configNames_prop = tcc.configNames;
initial_getConfigNames = tcc.getConfigNames();
initial_config2_name = tcc.getConfig(2).name;

tcc.setConfigNames({'cfgA', 'cfgB'});
names_after_set = tcc.getConfigNames();

tc3 = TrialConfig({'Velocity', 'v_x'}, 1000, [.05 .1], -1, 2, [], 'cfgC');
tcc.addConfig(tc3);
names_after_add = tcc.getConfigNames();
numConfigs_after_add = tcc.numConfigs;

subset = tcc.getSubsetConfigs([1 3]);
subset_names = subset.getConfigNames();

struct_payload = tcc.toStructure();
roundtrip = ConfigColl.fromStructure(struct_payload);
roundtrip_numConfigs = roundtrip.numConfigs;
roundtrip_configNames_prop = roundtrip.configNames;
roundtrip_getConfigNames = roundtrip.getConfigNames();
roundtrip_struct = roundtrip.toStructure();

save(outputFile, ...
    'initial_numConfigs', ...
    'initial_configNames_prop', ...
    'initial_getConfigNames', ...
    'initial_config2_name', ...
    'names_after_set', ...
    'names_after_add', ...
    'numConfigs_after_add', ...
    'subset_names', ...
    'struct_payload', ...
    'roundtrip_numConfigs', ...
    'roundtrip_configNames_prop', ...
    'roundtrip_getConfigNames', ...
    'roundtrip_struct');

fprintf('Wrote ConfigColl fixtures to %s\n', outputFile);
end
