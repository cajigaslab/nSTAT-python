"""Compatibility namespaces for non-Pythonic import styles.

This subpackage exists to support readers who prefer MATLAB-style imports
or who are migrating a MATLAB workflow:

>>> from nstat.compat.matlab import CIF, Covariate, SignalObj, nspikeTrain, nstColl

The canonical, Pythonic imports remain ``from nstat import ...``.  The
classes exposed through ``nstat.compat.matlab`` are the same objects as
the top-level imports; they are not separate implementations.

Each compatibility submodule is intentionally thin so the canonical
package surface stays the authoritative source of truth.
"""
