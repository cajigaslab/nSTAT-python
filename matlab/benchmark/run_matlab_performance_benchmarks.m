function run_matlab_performance_benchmarks(outputJson, outputCsv, nstatRoot)
%RUN_MATLAB_PERFORMANCE_BENCHMARKS Build MATLAB baseline performance report.
%
% Usage:
%   run_matlab_performance_benchmarks(outputJson, outputCsv, nstatRoot)
%
% Inputs:
%   outputJson - JSON report path
%   outputCsv  - CSV report path
%   nstatRoot  - path to MATLAB nSTAT source repo

if nargin < 1 || isempty(outputJson)
    outputJson = fullfile(pwd, 'output', 'performance', 'matlab_performance_report.json');
end
if nargin < 2 || isempty(outputCsv)
    outputCsv = fullfile(pwd, 'output', 'performance', 'matlab_performance_report.csv');
end
if nargin < 3 || isempty(nstatRoot)
    nstatRoot = getenv('NSTAT_MATLAB_ROOT');
end
if isempty(nstatRoot)
    error('nstatRoot is required (arg 3 or NSTAT_MATLAB_ROOT env var).');
end
if exist(nstatRoot, 'dir') ~= 7
    error('nSTAT root does not exist: %s', nstatRoot);
end

addpath(nstatRoot, '-begin');

[jsonDir, ~, ~] = fileparts(outputJson);
[csvDir, ~, ~] = fileparts(outputCsv);
if exist(jsonDir, 'dir') ~= 7
    mkdir(jsonDir);
end
if exist(csvDir, 'dir') ~= 7
    mkdir(csvDir);
end

cases = {'unit_impulse_basis', 'covariate_resample', 'history_design_matrix', 'simulate_cif_thinning', 'decoding_spike_rate_cis'};
tiers = {'S', 'M', 'L'};
repeats = 7;
warmup = 2;
seedBase = 20260303;
rows = {};

for iCase = 1:numel(cases)
    for iTier = 1:numel(tiers)
        caseName = cases{iCase};
        tierName = tiers{iTier};
        runtimesMs = zeros(1, repeats);
        memoryMb = zeros(1, repeats);
        summary = struct();

        for rep = 1:(warmup + repeats)
            rng(seedBase + rep, 'twister');
            tStart = tic;
            summary = run_case(caseName, tierName);
            elapsedMs = toc(tStart) * 1000;

            if rep > warmup
                idx = rep - warmup;
                runtimesMs(idx) = elapsedMs;
                if isfield(summary, 'memory_proxy_mb')
                    memoryMb(idx) = summary.memory_proxy_mb;
                else
                    memoryMb(idx) = NaN;
                end
            end
        end

        row = struct();
        row.case = caseName;
        row.tier = tierName;
        row.repeats = repeats;
        row.warmup = warmup;
        row.median_runtime_ms = median(runtimesMs);
        row.mean_runtime_ms = mean(runtimesMs);
        row.std_runtime_ms = std(runtimesMs);
        row.median_peak_memory_mb = median(memoryMb);
        row.summary = summary;
        row.samples_runtime_ms = runtimesMs;
        row.samples_peak_memory_mb = memoryMb;
        rows{end + 1} = row; %#ok<AGROW>
    end
end

report.schema_version = 1;
report.generated_at_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.implementation = 'matlab';
report.nstat_root = nstatRoot;
report.reference_sha = resolve_git_sha(nstatRoot);
report.tiers = tiers;
report.cases = rows;
report.environment = collect_environment();

jsonText = jsonencode(report, 'PrettyPrint', true);
fid = fopen(outputJson, 'w');
if fid < 0
    error('Failed to open output JSON for write: %s', outputJson);
end
fwrite(fid, jsonText, 'char');
fclose(fid);

write_csv(rows, outputCsv);

fprintf('Wrote MATLAB performance JSON: %s\n', outputJson);
fprintf('Wrote MATLAB performance CSV: %s\n', outputCsv);
fprintf('Benchmarked case-tier pairs: %d\n', numel(rows));
end

function summary = run_case(caseName, tier)
cfg = get_case_config(caseName, tier);

switch caseName
    case 'unit_impulse_basis'
        basis = nstColl.generateUnitImpulseBasis(cfg.basis_width_s, 0.0, cfg.max_time_s, cfg.sample_rate_hz);
        mat = basis.data;
        summary.rows = size(mat, 1);
        summary.cols = size(mat, 2);
        summary.total_mass = sum(mat(:));
        summary.memory_proxy_mb = bytes_to_mb(whos('mat'));

    case 'covariate_resample'
        t = linspace(0.0, cfg.duration_s, cfg.n_grid)';
        y = sin(2.0 * pi * 3.0 * t) + 0.2 * cos(2.0 * pi * 9.0 * t);
        stim = Covariate(t, y, 'Stimulus', 'time', 's', 'V', {'stim'});
        stimRes = stim.resample(cfg.sample_rate_hz);
        mat = stimRes.data;
        summary.rows = size(mat, 1);
        summary.cols = size(mat, 2);
        summary.signal_energy = mean(mat(:, 1) .^ 2);
        summary.memory_proxy_mb = bytes_to_mb(whos('mat'));

    case 'history_design_matrix'
        spikeTimes = deterministic_spike_times(cfg.n_spikes, cfg.duration_s);
        tn = linspace(0.0, cfg.duration_s, cfg.n_grid)';
        histObj = History([0.0, 0.01, 0.02, 0.05, 0.10], 0.0, cfg.duration_s);
        nst = nspikeTrain(spikeTimes);
        nst.setMinTime(0.0);
        nst.setMaxTime(cfg.duration_s);
        cov = histObj.computeHistory(nst, [], tn);
        mat = cov.dataToMatrix();
        summary.rows = size(mat, 1);
        summary.cols = size(mat, 2);
        summary.total_count = sum(mat(:));
        summary.memory_proxy_mb = bytes_to_mb(whos('mat'));

    case 'simulate_cif_thinning'
        t = linspace(0.0, cfg.duration_s, floor(cfg.duration_s * 1000) + 1)';
        lam = 12.0 + 8.0 * sin(2.0 * pi * 3.0 * t);
        lam(lam < 0.2) = 0.2;
        lambdaCov = Covariate(t, lam, 'Lambda', 'time', 's', 'Hz', {'lambda'});
        coll = CIF.simulateCIFByThinningFromLambda(lambdaCov, cfg.n_realizations, cfg.max_time_res_s);
        totalSpikes = 0;
        for i = 1:coll.numSpikeTrains
            totalSpikes = totalSpikes + numel(coll.getNST(i).getSpikeTimes());
        end
        summary.num_units = coll.numSpikeTrains;
        summary.total_spikes = totalSpikes;
        summary.mean_spikes_per_unit = totalSpikes / max(coll.numSpikeTrains, 1);
        summary.memory_proxy_mb = bytes_to_mb(whos('lam'));

    case 'decoding_spike_rate_cis'
        [xK, Wku, dN] = deterministic_decode_inputs(cfg);
        t0 = 0.0;
        tf = (cfg.n_bins - 1) * cfg.decode_delta_s;
        [spikeRateSig, probMat, sigMat] = DecodingAlgorithms.computeSpikeRateCIs( ...
            xK, Wku, dN, t0, tf, 'binomial', cfg.decode_delta_s, 0.0, [], cfg.mc_draws, 0.05);
        rate = spikeRateSig.data;
        summary.num_trials = size(probMat, 1);
        summary.prob_mean = mean(probMat(:));
        summary.sig_count = sum(sigMat(:));
        summary.rate_mean = mean(rate(:));
        summary.memory_proxy_mb = bytes_to_mb(whos('probMat'));

    otherwise
        error('Unknown benchmark case: %s', caseName);
end
end

function cfg = get_case_config(caseName, tier)
switch caseName
    case 'unit_impulse_basis'
        switch tier
            case 'S'
                cfg.max_time_s = 1.0; cfg.sample_rate_hz = 500.0;
            case 'M'
                cfg.max_time_s = 2.0; cfg.sample_rate_hz = 1000.0;
            case 'L'
                cfg.max_time_s = 4.0; cfg.sample_rate_hz = 1500.0;
            otherwise
                error('Unknown tier: %s', tier);
        end
        cfg.basis_width_s = 0.02;

    case 'covariate_resample'
        switch tier
            case 'S'
                cfg.duration_s = 2.0; cfg.n_grid = 2001; cfg.sample_rate_hz = 500.0;
            case 'M'
                cfg.duration_s = 4.0; cfg.n_grid = 4001; cfg.sample_rate_hz = 750.0;
            case 'L'
                cfg.duration_s = 6.0; cfg.n_grid = 6001; cfg.sample_rate_hz = 1000.0;
            otherwise
                error('Unknown tier: %s', tier);
        end

    case 'history_design_matrix'
        switch tier
            case 'S'
                cfg.n_spikes = 200; cfg.n_grid = 1000; cfg.duration_s = 2.0;
            case 'M'
                cfg.n_spikes = 1000; cfg.n_grid = 5000; cfg.duration_s = 2.0;
            case 'L'
                cfg.n_spikes = 3000; cfg.n_grid = 10000; cfg.duration_s = 2.0;
            otherwise
                error('Unknown tier: %s', tier);
        end

    case 'simulate_cif_thinning'
        switch tier
            case 'S'
                cfg.duration_s = 1.0; cfg.n_realizations = 5; cfg.max_time_res_s = 0.001;
            case 'M'
                cfg.duration_s = 2.0; cfg.n_realizations = 10; cfg.max_time_res_s = 0.001;
            case 'L'
                cfg.duration_s = 3.0; cfg.n_realizations = 20; cfg.max_time_res_s = 0.001;
            otherwise
                error('Unknown tier: %s', tier);
        end

    case 'decoding_spike_rate_cis'
        switch tier
            case 'S'
                cfg.num_basis = 4; cfg.num_trials = 6; cfg.n_bins = 120; cfg.mc_draws = 30;
            case 'M'
                cfg.num_basis = 6; cfg.num_trials = 8; cfg.n_bins = 200; cfg.mc_draws = 50;
            case 'L'
                cfg.num_basis = 8; cfg.num_trials = 12; cfg.n_bins = 320; cfg.mc_draws = 80;
            otherwise
                error('Unknown tier: %s', tier);
        end
        cfg.decode_delta_s = 0.01;

    otherwise
        error('Unknown benchmark case: %s', caseName);
end
end

function spikes = deterministic_spike_times(nSpikes, duration_s)
idx = (1:nSpikes)';
phi = 0.6180339887498949;
spikes = mod(idx .* phi, 1.0) .* duration_s;
spikes = sort(spikes);
end

function [xK, Wku, dN] = deterministic_decode_inputs(cfg)
[basisGrid, trialGrid] = ndgrid(1:cfg.num_basis, 1:cfg.num_trials);
xK = 0.06 * sin(0.37 * (basisGrid .* trialGrid)) + 0.04 * cos(0.19 * (basisGrid .* trialGrid));

Wku = zeros(cfg.num_basis, cfg.num_basis, cfg.num_trials, cfg.num_trials);
for r = 1:cfg.num_basis
    Wku(r, r, :, :) = 0.05 * eye(cfg.num_trials);
end

grid = reshape(0:(cfg.num_trials * cfg.n_bins - 1), cfg.num_trials, cfg.n_bins);
dN = double((sin(0.173 * grid) + cos(0.037 * grid)) > 1.15);
end

function value = bytes_to_mb(whosStruct)
if isempty(whosStruct)
    value = NaN;
else
    value = double(whosStruct.bytes) / (1024.0 * 1024.0);
end
end

function sha = resolve_git_sha(repoRoot)
sha = 'unknown';
[status, out] = system(sprintf('git -C "%s" rev-parse HEAD', repoRoot));
if status == 0
    sha = strtrim(out);
end
end

function env = collect_environment()
env.matlab_version = version;
env.matlab_release = version('-release');
env.os = computer;
try
    env.blas = version('-blas');
catch
    env.blas = '';
end
env.omp_num_threads = getenv('OMP_NUM_THREADS');
env.mkl_num_threads = getenv('MKL_NUM_THREADS');
env.openblas_num_threads = getenv('OPENBLAS_NUM_THREADS');
end

function write_csv(rows, outCsv)
fid = fopen(outCsv, 'w');
if fid < 0
    error('Failed to open CSV output: %s', outCsv);
end
fprintf(fid, 'case,tier,repeats,median_runtime_ms,mean_runtime_ms,std_runtime_ms,median_peak_memory_mb,summary\n');
for i = 1:numel(rows)
    row = rows{i};
    summaryText = strrep(jsonencode(row.summary), '"', '""');
    fprintf(fid, '%s,%s,%d,%.9f,%.9f,%.9f,%.9f,"%s"\n', ...
        row.case, row.tier, row.repeats, row.median_runtime_ms, ...
        row.mean_runtime_ms, row.std_runtime_ms, row.median_peak_memory_mb, summaryText);
end
fclose(fid);
end
