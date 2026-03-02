# Shared Example Data Policy

This directory tracks only shared example datasets.

Rules:
- Data may be shared with MATLAB nSTAT when explicitly listed in `datasets_manifest.json`.
- Every dataset record must include immutable version and SHA256 checksum.
- No non-data files are shared across repositories.

Recommended flow:
1. Publish immutable data artifacts (for example, GitHub Release assets).
2. Add records to `datasets_manifest.json` with URL and checksum.
3. Update `tools/compliance/shared_data_allowlist.yml` with approved overlaps.

Current shared records:
- `mEPSC-epsc2` (source: `cajigaslab/nSTAT`, path `data/mEPSCs/epsc2.txt`)

MATLAB example-data mirror workflow:
1. Build source snapshot manifest:
   - `python tools/data_mirror/build_manifest.py --source-root <matlab_data_dir> --version <YYYYMMDD> --out data/shared/matlab_source_<YYYYMMDD>.manifest.json`
2. Sync exact mirrored copy into this repository:
   - `python tools/data_mirror/sync_matlab_data.py --source-root <matlab_data_dir> --version <YYYYMMDD> --dest-root data/shared --clean`
3. Regenerate clean-room allowlist entries from mirror manifest:
   - `python tools/compliance/update_shared_data_allowlist.py --manifest data/shared/matlab_gold_<YYYYMMDD>.manifest.json --allowlist tools/compliance/shared_data_allowlist.yml`
4. Verify the mirrored tree against manifest:
   - `python tools/data_mirror/verify_matlab_data.py --manifest data/shared/matlab_gold_<YYYYMMDD>.manifest.json --strict`
