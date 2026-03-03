function Covariate_fixtures(outputFile)
% Generate deterministic MATLAB fixtures for Covariate parity tests.

if nargin < 1 || isempty(outputFile)
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));
    outputDir = fullfile(repoRoot, 'tests', 'fixtures', 'Covariate');
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

time = [0; 0.2; 0.4; 0.6; 0.8; 1.0];
data = [ ...
    0.0  0.00  0.50; ...
    0.2  0.04  0.70; ...
    0.4  0.16  0.80; ...
    0.6  0.36  0.95; ...
    0.8  0.64  1.10; ...
    1.0  1.00  1.25];
labels = {'c1','c2','c3'};

cov = Covariate(time, data, 'stim', 'time', 's', 'u', labels);
base_data = cov.dataToMatrix();
base_time = cov.time;

sigrep_standard = cov.getSigRep('standard').dataToMatrix();
sigrep_zero_mean = cov.getSigRep('zero-mean').dataToMatrix();

sub_ind_data = cov.getSubSignal(2).dataToMatrix();
sub_name_data = cov.getSubSignal('c3').dataToMatrix();

mean_ci = cov.computeMeanPlusCI(0.10);
mean_ci_data = mean_ci.dataToMatrix();
mean_ci_interval = mean_ci.ci{1}.dataToMatrix();

covA = Covariate(time, data(:,1), 'a', 'time', 's', 'u', {'a'});
ciA = ConfidenceInterval(time, [data(:,1)-0.10 data(:,1)+0.20]);
covA.setConfInterval(ciA);

covB = Covariate(time, 0.5*ones(size(time)), 'b', 'time', 's', 'u', {'b'});
plus_scalar = covA + 0.5;
plus_scalar_data = plus_scalar.dataToMatrix();
plus_scalar_ci = plus_scalar.ci{1}.dataToMatrix();

minus_scalar = covA - 0.5;
minus_scalar_data = minus_scalar.dataToMatrix();
minus_scalar_ci = minus_scalar.ci{1}.dataToMatrix();

cov_no_ci_1 = Covariate(time, data(:,1), 'n1', 'time', 's', 'u', {'n1'});
cov_no_ci_2 = Covariate(time, data(:,1)+0.25, 'n2', 'time', 's', 'u', {'n2'});
plus_no_ci = cov_no_ci_1 + cov_no_ci_2;
plus_no_ci_data = plus_no_ci.dataToMatrix();
minus_no_ci = cov_no_ci_1 - cov_no_ci_2;
minus_no_ci_data = minus_no_ci.dataToMatrix();

is_ci_before = covB.isConfIntervalSet();
covB.setConfInterval(ciA);
is_ci_after = covB.isConfIntervalSet();

filt = cov.filtfilt([0.2 0.2], [1 -0.3]);
filt_data = filt.dataToMatrix();

cov_struct = covA.toStructure();
roundtrip = Covariate.fromStructure(cov_struct);
roundtrip_data = roundtrip.dataToMatrix();
roundtrip_ci = roundtrip.ci{1}.dataToMatrix();

save(outputFile, ...
    'time', ...
    'data', ...
    'base_data', ...
    'base_time', ...
    'sigrep_standard', ...
    'sigrep_zero_mean', ...
    'sub_ind_data', ...
    'sub_name_data', ...
    'mean_ci_data', ...
    'mean_ci_interval', ...
    'plus_scalar_data', ...
    'plus_scalar_ci', ...
    'minus_scalar_data', ...
    'minus_scalar_ci', ...
    'plus_no_ci_data', ...
    'minus_no_ci_data', ...
    'is_ci_before', ...
    'is_ci_after', ...
    'filt_data', ...
    'cov_struct', ...
    'roundtrip_data', ...
    'roundtrip_ci');

fprintf('Wrote Covariate fixtures to %s\n', outputFile);
end
