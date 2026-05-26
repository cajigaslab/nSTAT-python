# `tests/parity/_third_party/` — clean-room oracles

This directory holds **independent, clean-room implementations** of
statistical procedures used as parity oracles against `nstat.*` methods.

## What "clean-room" means here

Each oracle here is implemented from the **published algorithm** in the
cited primary literature, not by copying or adapting any specific
open-source implementation. The goal is to triangulate `nstat.*`
results against an unrelated implementation written from the same
mathematical specification — so that an agreement to high precision is
strong evidence both implementations are correct.

## Why not vendor upstream?

Two reasons:

1. **License-attribution complexity.** Vendoring even a permissively-
   licensed (MIT/BSD) source into this repository requires preserving
   notices in every file and tracking upstream license changes. For
   ~50-100 LOC oracle implementations, a clean-room rewrite is
   cheaper.
2. **Upstream maintenance status.** Several candidate upstreams (e.g.,
   `time_rescale` v0.2.1 from 2021) are effectively unmaintained.
   Pinning to a stale version is brittle; a clean-room version we own
   doesn't drift.

## What lives here

| File | Cites | Used by |
|---|---|---|
| `time_rescale_oracle.py` | Brown, Barbieri, Ventura, Kass, Frank — "The time-rescaling theorem and its application to neural spike train data analysis" (Neural Computation 2002, 14:325–346) | `tests/test_time_rescale_oracle.py` |

## What this directory is NOT

- Not part of the published `nstat` package (it lives under `tests/`,
  not under `nstat/`).
- Not on the public API surface — no `__init__.py` re-exports.
- Not intended for production use — these oracles are deliberately
  minimal to be easy to audit, not fast or robust on real data.
