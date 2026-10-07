#!/usr/bin/env bash
# Run the parity_core notebook group without touching the user's own Jupyter
# kernelspecs or installing anything into their environment.
#
# 2026-10 review finding: the original `make notebooks-check` ran
# `$(PIP) install -q ipykernel` and `ipykernel install --user --name python3`,
# which (a) silently installed a package into whatever environment $(PY)
# pointed at, and (b) overwrote the user-level `python3` Jupyter kernelspec
# in ~/Library/Jupyter/kernels/python3 -- clobbering whatever that name
# previously pointed to, with no restoration path if the run failed midway.
#
# This script instead:
#   - requires ipykernel to already be installed in $PY's environment
#     (fails with an actionable message otherwise -- never installs it);
#   - installs a kernelspec under a per-run `mktemp -d` directory via
#     JUPYTER_DATA_DIR, under a dedicated name ("nstat-check"), so it never
#     touches ~/Library/Jupyter/kernels/python3 or any other existing name;
#   - removes that temp directory on exit, success or failure (`trap`).
set -euo pipefail

PY="${1:?usage: notebooks_check.sh <python> [run_notebooks.py args...]}"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_NAME="nstat-check"

if ! "$PY" -c "import ipykernel" >/dev/null 2>&1; then
    echo "error: ipykernel is not installed in $PY." >&2
    echo "This script will not install it for you (it must not mutate your" >&2
    echo "environment without being asked). Install it yourself first:" >&2
    echo "  $PY -m pip install ipykernel" >&2
    exit 1
fi

TMP_JUPYTER_DATA_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_JUPYTER_DATA_DIR"
}
trap cleanup EXIT

export JUPYTER_DATA_DIR="$TMP_JUPYTER_DATA_DIR"

"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "nstat-check (isolated, temporary)" \
    >/dev/null

"$PY" "$REPO_ROOT/tools/notebook_build/run_notebooks.py" --kernel-name "$KERNEL_NAME" "$@"
