"""State-space EM bridges via Dynamax.

MATLAB nSTAT exposes ``KF_EM`` / ``PP_EM`` / ``mPPCO_EM`` — three
families of EM-trained linear-Gaussian, point-process, and hybrid
point-process / continuous-observation state-space models. They are ported
natively, as MATLAB mirrors, in ``nstat.DecodingAlgorithms`` (``KF_EM``,
``PP_EM``, ``PPLFP_EM``; ``mPPCO_*`` are deprecated aliases of ``PPLFP_*``).
This subpackage is a separate, Python-only alternative, started when
AUDIT_REPORT.md §3.2 still listed those 19 methods as unported.

This subpackage wraps `Dynamax <https://github.com/probml/dynamax>`_
(JAX-based, MIT-licensed), which implements the same family of models
with a modern computational backend; the bridge translates between
nstat's object-oriented API and Dynamax's pytree-parameter API.

Install:
    pip install nstat-toolbox[dynamax]
"""
from __future__ import annotations

__all__: list[str] = []
