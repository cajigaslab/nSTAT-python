from __future__ import annotations

import nstat
from nstat.errors import DataNotFoundError


def test_dataset_manifest_and_checksums() -> None:
    names = nstat.list_datasets()
    assert names

    check = nstat.verify_checksums()
    assert set(check.keys()) == set(names)
    assert all(isinstance(v, bool) for v in check.values())


def test_get_dataset_path() -> None:
    name = nstat.list_datasets()[0]
    try:
        path = nstat.get_dataset_path(name)
    except DataNotFoundError:
        # Standalone checkouts may intentionally omit large datasets.
        return
    assert path.exists()
