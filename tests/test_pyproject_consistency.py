"""Structural assertions about ``pyproject.toml``'s ``[project.optional-dependencies]``.

These tests catch drift that ``pip install`` cannot — e.g., a new
optional-deps group landing without being added to ``[all-extras]``
(silently breaking the README's promise that the all-extras key is
"install everything"), or a typo in a package name that only manifests
when a user actually tries to install.

These are pure-text checks against the parsed TOML; they do not perform
network installs.  Add new assertions here whenever an optional-deps
contract emerges that's not currently locked down.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _opt_deps() -> dict[str, list[str]]:
    return dict(_read_pyproject()["project"]["optional-dependencies"])


# Groups whose contents are aspirational placeholders — empty lists are
# deliberate (no concrete extras module shipped yet).  Adding to this
# set is allowed; removing requires the corresponding group to also
# acquire real dependencies.
PLACEHOLDER_GROUPS = frozenset({"spikeinterface", "deep-learning"})

# Meta-groups that aggregate other groups; not user-facing single
# extras.  ``dev`` is the developer toolkit (pytest, sphinx, …) and
# is intentionally excluded from the all-extras union.
META_GROUPS = frozenset({"all-extras", "dev"})

# Heavyweight groups deliberately excluded from ``[all-extras]`` to
# keep the one-shot install footprint reasonable.  Membership here is
# a documented architectural decision, not a TODO.  Notes:
# - ``dynamax`` pulls JAX (~200 MB).
# - ``clusterless`` (``replay_trajectory_classification``) pulls JAX too
#   (~200 MB) and is the same install-footprint hazard as dynamax.
# - ``spatial-gp`` pulls gpflow + TensorFlow (heavy); ``hawkes`` (``tick``)
#   and ``dpp`` (``DPPy``) back only the OPTIONAL bridges of the
#   pure-NumPy/SciPy ``nstat.extras.spatial`` module, so they are not part
#   of the always-installable core surface.
HEAVY_OPT_OUT_OF_ALL_EXTRAS = frozenset(
    {"dynamax", "clusterless", "spatial-gp", "hawkes", "dpp"}
)


# ----------------------------------------------------------------------
# Contract 1 — every group is either a placeholder or has real deps
# ----------------------------------------------------------------------


def test_every_non_placeholder_group_has_at_least_one_dep() -> None:
    """A non-empty optional-deps group must list at least one package.

    Empty lists are reserved for explicit placeholders (see
    ``PLACEHOLDER_GROUPS``).  This guards against accidentally shipping
    an empty group that silently no-ops on install.
    """
    offenders: list[str] = []
    for group_name, deps in _opt_deps().items():
        if group_name in PLACEHOLDER_GROUPS or group_name in META_GROUPS:
            continue
        if not deps:
            offenders.append(group_name)
    assert not offenders, (
        f"Non-placeholder optional-deps groups with empty dep lists: "
        f"{offenders}.  Either add real deps or move the group name "
        f"into PLACEHOLDER_GROUPS in this test."
    )


# ----------------------------------------------------------------------
# Contract 2 — [all-extras] is the true union of every functional group
# ----------------------------------------------------------------------


def _normalize_package(spec: str) -> str:
    """Extract bare package name from a PEP 508 spec (``foo>=1.2`` → ``foo``)."""
    # PEP 508: name [extras] [version-spec] [env-marker]
    m = re.match(r"^\s*([A-Za-z0-9._-]+)", spec)
    assert m is not None, f"Cannot parse package name from spec: {spec!r}"
    return m.group(1).lower().replace("_", "-")


def test_all_extras_is_union_of_every_functional_group() -> None:
    """``[all-extras]`` must contain at least every package found in any
    individually-installable group (excluding placeholders + meta).

    Catches the bug from deep-dive #3 of the extras-shipping PR
    (#93): ``[all-extras]`` claimed to be the one-shot install but
    silently omitted pykalman / statsmodels / nitime from the
    ``[test-parity]`` group.
    """
    opt = _opt_deps()
    all_extras_pkgs = {_normalize_package(s) for s in opt.get("all-extras", [])}

    missing_by_group: dict[str, list[str]] = {}
    for group_name, deps in opt.items():
        if (
            group_name in PLACEHOLDER_GROUPS
            or group_name in META_GROUPS
            or group_name in HEAVY_OPT_OUT_OF_ALL_EXTRAS
        ):
            continue
        for spec in deps:
            pkg = _normalize_package(spec)
            if pkg not in all_extras_pkgs:
                missing_by_group.setdefault(group_name, []).append(pkg)

    assert not missing_by_group, (
        "[all-extras] is missing packages declared in other groups — "
        "the README's 'install everything' promise is broken.  "
        f"Missing: {missing_by_group}.  Add them to the all-extras "
        "list in pyproject.toml."
    )


# ----------------------------------------------------------------------
# Contract 3 — README's install matrix names real groups
# ----------------------------------------------------------------------


def test_readme_extras_install_lines_reference_real_groups() -> None:
    """Every ``pip install nstat-toolbox[GROUP]`` mention in README.md
    must reference a group that exists in ``[project.optional-dependencies]``.

    Catches drift where the README advertises an extras key that was
    renamed or removed.  Excludes the literal placeholder ``[GROUP]``
    (used in template prose).
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    opt = _opt_deps()
    declared_groups = set(opt.keys())

    pattern = re.compile(r"pip install nstat-toolbox\[([a-zA-Z0-9_-]+)\]")
    referenced = set(pattern.findall(readme))

    unknown = referenced - declared_groups - {"GROUP"}
    assert not unknown, (
        f"README.md references extras groups that don't exist in "
        f"pyproject.toml: {sorted(unknown)}.  Either add the group "
        f"or fix the README."
    )


# ----------------------------------------------------------------------
# Contract 4 — every extras subpackage has a backing dep group
# ----------------------------------------------------------------------


def test_every_extras_subpackage_has_corresponding_deps_group() -> None:
    """If ``nstat/extras/interop/X.py`` exists, there must be a
    ``[project.optional-dependencies.X]`` group OR the module must be
    explicitly listed in this test's allowlist.

    Catches drift where a new extras module ships without anyone
    updating pyproject.toml — users would have no documented way to
    ``pip install`` what they need.
    """
    opt = _opt_deps()
    declared_groups = set(opt.keys())

    # Map: extras module stem → expected group name.
    # When a module's natural group name differs from its filename
    # (e.g., spike_distances → [metrics]), declare the mapping here.
    EXPECTED_GROUP_FOR_MODULE = {
        # interop/
        "neo": "neo",
        "pynapple": "pynapple",
        "nwb": "nwb",
        # validation/  — backed by [test-parity] which bundles everything
        "nemos_bridge": "test-parity",
        "pykalman_bridge": "test-parity",
        "statsmodels_bridge": "test-parity",
        "nitime_bridge": "test-parity",
        # metrics/
        "spike_distances": "metrics",
        # em/
        "dynamax": "dynamax",
        "dynamax_bridge": "dynamax",
        # decoding/
        "clusterless_bridge": "clusterless",
        # spatial/  — OPTIONAL bridges have a backing group; the pure-core
        # modules need none (see CORE_NO_DEP_MODULES below).
        "hawkes_bridge": "hawkes",
        "dpp_bridge": "dpp",
    }

    # Pure-NumPy/SciPy extras modules that depend only on the core
    # numpy/scipy already in [project.dependencies] — they correctly have
    # NO backing optional-deps group.  (The nstat.extras.spatial core:
    # lgcp / spatial_gof / marked_gof.)  The heavier optional GP path of
    # lgcp is gated behind the [spatial-gp] group at call time.
    CORE_NO_DEP_MODULES = frozenset({"lgcp", "spatial_gof", "marked_gof", "basis"})

    extras_root = REPO_ROOT / "nstat" / "extras"
    if not extras_root.exists():
        pytest.skip("nstat/extras/ not present")

    offenders: list[tuple[str, str]] = []
    for py_file in extras_root.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        # Internal helpers (leading underscore) are private utilities
        # with no optional dep — they don't need a [optional-deps] group.
        if py_file.stem.startswith("_"):
            continue
        stem = py_file.stem
        # Pure-core extras modules legitimately have no backing dep group.
        if stem in CORE_NO_DEP_MODULES:
            continue
        expected = EXPECTED_GROUP_FOR_MODULE.get(stem)
        if expected is None:
            # Unknown module — add it to the EXPECTED_GROUP_FOR_MODULE
            # map here OR to the pyproject.toml deps groups.
            offenders.append((str(py_file.relative_to(REPO_ROOT)), "(unmapped)"))
            continue
        if expected not in declared_groups:
            offenders.append((str(py_file.relative_to(REPO_ROOT)), expected))

    assert not offenders, (
        f"Extras modules without a backing [optional-dependencies] group: "
        f"{offenders}.  Either add the group to pyproject.toml or add the "
        f"module to EXPECTED_GROUP_FOR_MODULE in this test."
    )
