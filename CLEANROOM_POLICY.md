# Clean-Room Policy for nSTAT-python

This repository is a clean-room Python implementation of nSTAT.

## Rules
- No MATLAB runtime/build/test dependency is allowed.
- No code/docs/workflows/notebooks/config files are copied from MATLAB nSTAT.
- Only example data may be shared across repositories, and only when explicitly listed in the allowlist.

## Enforcement
- CI runs a hash-overlap compliance job against `cajigaslab/nSTAT`.
- Non-data hash collisions fail the build.
- Shared data files must be explicitly listed in `tools/compliance/shared_data_allowlist.yml`.
