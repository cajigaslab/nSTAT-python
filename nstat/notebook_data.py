from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import data_manager


def _download_policy() -> bool:
    policy = os.environ.get("NSTAT_NOTEBOOK_DOWNLOAD_EXAMPLE_DATA", "").strip().lower()
    if policy in {"1", "true", "yes", "on", "always"}:
        return True
    if policy in {"0", "false", "no", "off", "never"}:
        return False
    return os.environ.get("CI", "").strip().lower() not in {"1", "true", "yes", "on"}


def notebook_example_data_dir(*, allow_synthetic: bool = False) -> Path:
    """Return a notebook-safe example-data root.

    Local runs still auto-download data when needed. CI defaults to cached-only
    behavior and can fall back to the synthetic dataset paths used by the paper
    example helpers.
    """

    try:
        return data_manager.ensure_example_data(download=_download_policy())
    except FileNotFoundError:
        if not allow_synthetic:
            raise
        os.environ.setdefault("NSTAT_ALLOW_SYNTHETIC_DATA", "1")
        return data_manager.get_data_dir()


def _is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:200]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _synthetic_glm_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1202)
    sample_rate = 1000.0
    dt = 1.0 / sample_rate
    duration_s = 6.0
    t = np.arange(0.0, duration_s + dt, dt, dtype=float)
    theta = 2.0 * np.pi * 0.08 * t
    x_n = 0.75 * np.cos(theta) + 0.15 * np.sin(0.35 * theta)
    y_n = 0.65 * np.sin(theta + 0.2) + 0.10 * np.cos(0.5 * theta)
    vx_n = np.gradient(x_n, dt)
    vy_n = np.gradient(y_n, dt)

    eta = 1.4 + 1.0 * x_n - 0.8 * y_n - 0.55 * x_n * x_n - 0.45 * y_n * y_n + 0.35 * x_n * y_n
    lam_per_bin = np.clip(np.exp(np.clip(eta, -8.0, 4.5)) * dt, 1e-6, 0.2)
    spikes_binned = (rng.random(t.shape[0]) < lam_per_bin).astype(float)
    spike_idx = np.flatnonzero(spikes_binned > 0.5)
    spiketimes = t[spike_idx]
    x_at_spiketimes = x_n[spike_idx]
    y_at_spiketimes = y_n[spike_idx]

    return {
        "T": t,
        "xN": x_n,
        "yN": y_n,
        "vxN": vx_n,
        "vyN": vy_n,
        "spikes_binned": spikes_binned,
        "spiketimes": spiketimes,
        "x_at_spiketimes": x_at_spiketimes,
        "y_at_spiketimes": y_at_spiketimes,
    }


def load_glm_data_for_notebook() -> dict[str, np.ndarray]:
    """Return the canonical GLM dataset or a deterministic synthetic fallback."""

    data_dir = notebook_example_data_dir(allow_synthetic=True)
    path = data_dir / "glm_data.mat"
    if path.exists() and not _is_lfs_pointer(path):
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in payload.items() if not key.startswith("__")}
    return _synthetic_glm_data()
