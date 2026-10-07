# Contributing to nstat-python

This is a short, focused note file — repo-wide conventions live in
`AGENT_GUIDE.md` (toolbox usage) and the project's own maintenance
playbook. Add to this file as other cross-cutting testing gotchas come up;
keep each note short and dated.

## Testing singular matrices (2026-10)

**Never build a "should be singular" matrix from a floating-point
computation whose cancellation depends on summation order.** The classic
trap: `x @ x.T` where every column of `x` is identical. Mathematically the
result has rank 1 (for a 2-row `x`), so it *looks* exactly singular — but
whether the Gram-matrix accumulation actually produces a bit-exact zero
pivot during LU factorization depends on the BLAS/LAPACK build's
summation order and rounding. Apple Accelerate hit an exact zero pivot on
this construction; Linux OpenBLAS computed a tiny *nonzero* pivot instead.
The test passed on macOS and failed on Linux CI (2026-10 EM / release
cycle incident).

**Fix:** make the structural zero explicit and literal, not an emergent
property of floating-point arithmetic. Use
`tests/linalg_fixtures.exactly_singular_gram(dim, n_samples, rng)` — it
zeroes an entire row of the underlying sample matrix *before* forming the
Gram product, so the corresponding row/column of the result is exactly
`0.0` by construction on every BLAS build. Pair it with
`tests/linalg_fixtures.assert_exactly_singular(matrix)` to self-check any
singular fixture (including a hand-rolled one) before trusting it in a
test — it rejects a matrix that is singular only up to rounding.

A hand-written matrix of small integers with a literally duplicated
row/column (e.g. `np.array([[4, 0, 0], [0, 1, 1], [0, 1, 1]])`) is fine
without the helper — the duplication is bit-exact in the input, not a
product of accumulated rounding, so no BLAS build can disagree about it.
The helper matters specifically for matrices built as `A @ A.T` or similar
reductions over many terms.
