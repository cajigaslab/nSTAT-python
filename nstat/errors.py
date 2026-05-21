"""Exception hierarchy for the Python nSTAT package.

All errors raised by user-facing nSTAT APIs derive from
:class:`NSTATError`, so callers can catch every package-specific failure
with a single ``except NSTATError`` clause.  Specialised subclasses also
inherit from the most appropriate Python builtin (``FileNotFoundError``,
``NotImplementedError``, ``RuntimeError``) so generic ``except`` handlers
continue to work.

Exported classes
----------------
- :class:`NSTATError` — root exception type for the package.
- :class:`DataNotFoundError` — required dataset missing on disk.
- :class:`ParityValidationError` — MATLAB/Python parity check failed.
- :class:`UnsupportedWorkflowError` — legacy MATLAB workflow not yet ported.
- :class:`MatlabEngineError` — MATLAB Engine interop failure.

This module has no MATLAB counterpart; the MATLAB toolbox uses ``error``
identifiers strings instead of an exception hierarchy.
"""
from __future__ import annotations


class NSTATError(Exception):
    """Base exception type for the Python nSTAT package."""


class DataNotFoundError(NSTATError, FileNotFoundError):
    """Raised when a required dataset is missing from the local checkout."""


class ParityValidationError(NSTATError):
    """Raised when MATLAB/Python parity validation fails."""


class UnsupportedWorkflowError(NSTATError, NotImplementedError):
    """Raised when a legacy workflow has not yet been ported."""


class MatlabEngineError(NSTATError, RuntimeError):
    """Raised when MATLAB Engine interaction fails."""
