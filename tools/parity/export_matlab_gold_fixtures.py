#!/usr/bin/env python3
"""Export MATLAB-gold fixtures for canonical parity workflows.

This script runs MATLAB in batch mode to generate deterministic fixture files
for three workflow families:
- PPSimExample
- DecodingExampleWithHist
- HippocampalPlaceCellExample
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import yaml


MATLAB_SCRIPT_TEMPLATE = r"""
rng(2026,'twister');

out_dir = '{out_dir}';
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

% ------------------------------
% Fixture 1: PPSimExample (GLM)
% ------------------------------
n = 2500;
dt = 0.01;
X = randn(n,1);
true_intercept = log(10.0);
true_beta = 0.45;
lambda = exp(true_intercept + X*true_beta);
y = poissrnd(lambda*dt);

offset = log(dt) * ones(n,1);
b = glmfit(X, y, 'poisson', 'constant', 'on', 'offset', offset);
expected_rate = exp(b(1) + X*b(2));

save(fullfile(out_dir, 'PPSimExample_gold.mat'), ...
    'X', 'y', 'dt', 'true_intercept', 'true_beta', 'b', 'expected_rate', '-v7');

% ------------------------------------------
% Fixture 2: DecodingExampleWithHist (Bayes)
% ------------------------------------------
n_units = 12;
n_states = 18;
n_time = 180;
centers = linspace(0, n_states-1, n_units)';
widths = 2.0 * ones(n_units,1);
states = 0:(n_states-1);
tuning = zeros(n_units, n_states);
for i=1:n_units
    tuning(i,:) = 0.05 + 0.35*exp(-0.5*((states-centers(i))./widths(i)).^2);
end

transition = zeros(n_states, n_states);
for i=1:n_states
    if i>1
        transition(i,i-1) = 0.2;
    end
    transition(i,i) = 0.6;
    if i<n_states
        transition(i,i+1) = 0.2;
    end
    transition(i,:) = transition(i,:)/sum(transition(i,:));
end

latent = zeros(1, n_time);
latent(1) = floor(n_states/2) + 1;
for t=2:n_time
    cdf = cumsum(transition(latent(t-1),:));
    r = rand();
    latent(t) = find(r <= cdf, 1, 'first');
end

spike_counts = zeros(n_units, n_time);
for t=1:n_time
    spike_counts(:,t) = poissrnd(tuning(:, latent(t)));
end

log_emit = zeros(n_states, n_time);
for s=1:n_states
    r = tuning(:,s);
    log_emit(s,:) = sum(spike_counts .* log(r) - r - gammaln(spike_counts + 1), 1);
end
log_prior = log((1.0/n_states) * ones(n_states,1));

log_post = zeros(n_states, n_time);
log_post(:,1) = log_prior + log_emit(:,1);
log_post(:,1) = log_post(:,1) - log(sum(exp(log_post(:,1))));

for t=2:n_time
    pred = zeros(n_states,1);
    for s_next=1:n_states
        vals = log_post(:,t-1) + log(transition(:,s_next));
        maxv = max(vals);
        pred(s_next) = maxv + log(sum(exp(vals - maxv)));
    end
    log_post(:,t) = pred + log_emit(:,t);
    maxv = max(log_post(:,t));
    log_post(:,t) = log_post(:,t) - (maxv + log(sum(exp(log_post(:,t)-maxv))));
end

expected_posterior = exp(log_post);
[~, idx] = max(expected_posterior, [], 1);
expected_decoded = idx - 1; % zero-based to match python implementation

save(fullfile(out_dir, 'DecodingExampleWithHist_gold.mat'), ...
    'spike_counts', 'tuning', 'transition', 'expected_posterior', 'expected_decoded', '-v7');

% --------------------------------------------------
% Fixture 3: HippocampalPlaceCellExample (Weighted)
% --------------------------------------------------
n_units_pc = 25;
n_states_pc = 40;
n_time_pc = 200;
state_axis = 0:(n_states_pc-1);

centers_pc = linspace(0, n_states_pc-1, n_units_pc)';
widths_pc = 3.0 + rand(n_units_pc,1);
tuning_curves = zeros(n_units_pc, n_states_pc);
for i=1:n_units_pc
    tuning_curves(i,:) = 0.1 + 1.2*exp(-0.5*((state_axis-centers_pc(i))./widths_pc(i)).^2);
end

latent_pc = zeros(1, n_time_pc);
latent_pc(1) = floor(n_states_pc/2) + 1;
for t=2:n_time_pc
    latent_pc(t) = min(max(latent_pc(t-1) + randi([-1,1]), 1), n_states_pc);
end

spike_counts_pc = zeros(n_units_pc, n_time_pc);
for t=1:n_time_pc
    spike_counts_pc(:,t) = poissrnd(tuning_curves(:, latent_pc(t)));
end

expected_decoded_weighted = zeros(1, n_time_pc);
for t=1:n_time_pc
    weights = spike_counts_pc(:,t) .* tuning_curves;
    post = sum(weights,1);
    post = post / (sum(post) + 1e-12);
    expected_decoded_weighted(t) = sum(post .* state_axis);
end

save(fullfile(out_dir, 'HippocampalPlaceCellExample_gold.mat'), ...
    'spike_counts_pc', 'tuning_curves', 'expected_decoded_weighted', '-v7');

fprintf('MATLAB gold fixtures exported to %s\n', out_dir);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold"),
        help="Directory for exported MATLAB fixtures",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/parity/fixtures/matlab_gold/manifest.yml"),
        help="Output manifest path",
    )
    return parser.parse_args()



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()



def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script_content = MATLAB_SCRIPT_TEMPLATE.format(out_dir=str(out_dir).replace("'", "''"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False, encoding="utf-8") as tmp:
        tmp.write(script_content)
        tmp_path = Path(tmp.name)

    try:
        escaped_tmp = str(tmp_path).replace("'", "''")
        cmd = ["matlab", "-batch", f"run('{escaped_tmp}')"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError("MATLAB fixture export failed")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    fixtures = []
    for file_name in [
        "PPSimExample_gold.mat",
        "DecodingExampleWithHist_gold.mat",
        "HippocampalPlaceCellExample_gold.mat",
    ]:
        path = out_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"expected fixture missing: {path}")
        fixtures.append(
            {
                "name": file_name.replace("_gold.mat", ""),
                "path": str(path.relative_to(repo_root).as_posix()),
                "sha256": _sha256(path),
                "source": "matlab_batch_export",
            }
        )

    manifest = {
        "version": 1,
        "fixtures": fixtures,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
