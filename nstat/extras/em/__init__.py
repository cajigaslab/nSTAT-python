"""State-space EM bridges via Dynamax.

MATLAB nSTAT exposes ``KF_EM`` / ``PP_EM`` / ``mPPCO_EM`` — three
families of EM-trained linear-Gaussian, point-process, and hybrid
point-process / continuous-observation state-space models. AUDIT_REPORT.md
§3.2 catalogs 19 unported methods in these families (roughly 7,500 LOC of
MATLAB if ported verbatim).

This subpackage wraps `Dynamax <https://github.com/probml/dynamax>`_
(JAX-based, MIT-licensed) which implements the same family of models
with a modern computational backend. Rather than re-implement 7,500
lines of EM in NumPy, the bridge translates between nstat's
object-oriented API and Dynamax's pytree-parameter API.

Install:
    pip install nstat-toolbox[dynamax]
"""
from __future__ import annotations

__all__: list[str] = []
