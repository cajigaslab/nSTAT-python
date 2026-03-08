from __future__ import annotations

from pathlib import Path

import numpy as np

import nstat.data_manager as data_manager
import nstat.notebook_data as notebook_data


def test_notebook_example_data_dir_disables_download_in_ci(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_ensure_example_data(*, download: bool = True) -> Path:
        calls.append(download)
        raise FileNotFoundError("missing")

    monkeypatch.setattr(data_manager, "ensure_example_data", fake_ensure_example_data)
    monkeypatch.setattr(data_manager, "get_data_dir", lambda: Path("/tmp/nstat-synthetic"))
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("NSTAT_NOTEBOOK_DOWNLOAD_EXAMPLE_DATA", raising=False)
    monkeypatch.delenv("NSTAT_ALLOW_SYNTHETIC_DATA", raising=False)

    path = notebook_data.notebook_example_data_dir(allow_synthetic=True)

    assert path == Path("/tmp/nstat-synthetic")
    assert calls == [False]
    assert notebook_data.os.environ["NSTAT_ALLOW_SYNTHETIC_DATA"] == "1"


def test_load_glm_data_for_notebook_uses_synthetic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(notebook_data, "notebook_example_data_dir", lambda *, allow_synthetic=False: Path("/tmp/missing-data"))

    payload = notebook_data.load_glm_data_for_notebook()

    expected = {
        "T",
        "xN",
        "yN",
        "vxN",
        "vyN",
        "spikes_binned",
        "spiketimes",
        "x_at_spiketimes",
        "y_at_spiketimes",
    }
    assert expected <= set(payload)
    assert payload["T"].ndim == 1
    assert payload["spikes_binned"].shape == payload["T"].shape
    assert payload["spiketimes"].ndim == 1
    assert np.all(np.diff(payload["T"]) > 0.0)
