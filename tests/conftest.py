from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def matlab_to_python_indices(arr) -> np.ndarray:
    """Translate MATLAB-stored 1-based index arrays to Python 0-based at the test boundary.

    Gold fixtures in ``tests/parity/fixtures/matlab_gold/*.mat`` store
    MATLAB's 1-based index arrays (``neuronNumbers``, ``selectorArray``,
    etc.).  Production code is now uniformly 0-based; tests calling
    ``loadmat(...).get("neuronNumbers")`` should route through this
    helper so the test-boundary translation is explicit and centralized.
    """
    return np.asarray(arr, dtype=int) - 1
