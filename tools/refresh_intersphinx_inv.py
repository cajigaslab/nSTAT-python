#!/usr/bin/env python3
"""Refresh the vendored intersphinx fallback inventories under ``docs/_inv/``.

The Sphinx docs build (``docs/conf.py``) cross-links to numpy / scipy /
matplotlib / sympy / Python via ``sphinx.ext.intersphinx``.  Each mapping
entry carries a committed fallback ``objects.inv`` so the strict ``-W``
build / GitHub Pages deploy can't fail on a transient host outage (e.g. a
``docs.scipy.org`` timeout): Sphinx tries the live inventory first, then
the vendored copy.

This script re-downloads each live ``objects.inv`` into ``docs/_inv/`` so
the fallbacks don't drift too far from upstream.  Run it occasionally
(e.g. from the scheduled docs review) — the URLs are derived from
``intersphinx_mapping`` in ``docs/conf.py`` so this stays in sync with the
configured targets automatically.

Usage::

    python tools/refresh_intersphinx_inv.py        # or: make refresh-intersphinx-inv
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
INV_DIR = DOCS / "_inv"
TIMEOUT = 30  # seconds


def _load_mapping() -> dict:
    """Import ``intersphinx_mapping`` from docs/conf.py without a full build."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_nstat_docs_conf", DOCS / "conf.py")
    assert spec and spec.loader
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    return conf.intersphinx_mapping


def main() -> int:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    mapping = _load_mapping()
    failures = []
    for name, (uri, _fallback) in mapping.items():
        url = uri.rstrip("/") + "/objects.inv"
        dest = INV_DIR / f"{name}.inv"
        try:
            # Some doc hosts (e.g. numpy.org) 403 the default urllib agent.
            req = urllib.request.Request(url, headers={"User-Agent": "nstat-docs-inv-refresh"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  ! {name}: failed to fetch {url} ({exc})")
            failures.append(name)
            continue
        if not data.startswith(b"# Sphinx inventory version 2"):
            print(f"  ! {name}: {url} did not return a v2 inventory; skipping")
            failures.append(name)
            continue
        dest.write_bytes(data)
        print(f"  ✓ {name}: {len(data):,} bytes -> {dest.relative_to(REPO_ROOT)}")

    if failures:
        print(f"\nRefreshed with {len(failures)} failure(s): {', '.join(failures)}")
        print("Existing vendored copies for the failed targets were left untouched.")
        return 1
    print(f"\nAll {len(mapping)} intersphinx inventories refreshed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
