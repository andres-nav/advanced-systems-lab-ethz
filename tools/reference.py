#!/usr/bin/env python3
"""Correctness check for any top-k Pearson correlation variant.

Usage:
    reference.py <input.bin> <cpp_stdout.txt>

Reads the same binary input the C++ binary consumed, recomputes top-k in
float64 NumPy, then compares against the C++ binary's parsed stdout.

Acceptance rule per feature row:
    (1) non-sentinel index sets match exactly, OR
    (2) every disagreement has |corr| within tolerance of the reference
        k-th |corr|  (i.e. it's a float32-vs-float64 tie at the boundary).

Exits 0 on pass, 1 on mismatch, 2 on usage/parse errors.
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

# Keep in sync with src/common/parameters.hpp.
STD_EPS = 1e-12
SENTINEL = 0xFFFFFFFF

# Tie tolerance at the k-th boundary.
RTOL, ATOL = 1e-4, 1e-6


def load_input(path: Path):
    with path.open("rb") as f:
        F, S = struct.unpack("<II", f.read(8))
        starts = np.frombuffer(f.read(4 * F), dtype=np.int32).copy()
        ends = np.frombuffer(f.read(4 * F), dtype=np.int32).copy()
        X = np.frombuffer(f.read(4 * F * S), dtype=np.float32).reshape(F, S).copy()
    return X, starts, ends


def compute_reference(X, starts, ends, K):
    """Float64 reference. Returns (topk, corr)."""
    F, S = X.shape
    Xd = X.astype(np.float64)

    # Per-feature mu/sigma over each feature's own interval. We build a
    # validity mask and use it to zero out invalid samples; mean/std are
    # computed by summing over the mask rather than per-row slicing so
    # the remaining compute is two matmuls.
    mask = np.zeros((F, S), dtype=np.float64)
    for i in range(F):
        mask[i, starts[i]:ends[i]] = 1.0
    n_own = mask.sum(axis=1)                      # == ends[i] - starts[i]
    mu = (Xd * mask).sum(axis=1) / np.maximum(n_own, 1)
    Xc = (Xd - mu[:, None]) * mask                # zero outside interval
    var = (Xc * Xc).sum(axis=1) / np.maximum(n_own - 1, 1)
    sig = np.sqrt(var + STD_EPS)

    # Normalized values, zero outside each feature's interval.
    Xn = Xc / sig[:, None]

    # Pairwise overlap count and dot product via masked matmul.
    # overlap[i, j] = |{t : starts[max(i,j)] <= t < ends[min(i,j)]}|.
    # We get this by multiplying the two masks.
    overlap = mask @ mask.T                       # (F, F), float64
    dot = Xn @ Xn.T                               # (F, F), float64

    # corr(i, j) = dot(i, j) / (overlap(i, j) - 1). Guard against n<2.
    denom = overlap - 1.0
    corr = np.where(denom >= 1.0, dot / np.maximum(denom, 1.0), 0.0)
    np.fill_diagonal(corr, 0.0)

    # Top-K by |corr|, matching the baseline's strict-positive filter.
    order = np.argsort(-np.abs(corr), axis=1)
    topk = np.full((F, K), SENTINEL, dtype=np.int64)
    for i in range(F):
        picked = [j for j in order[i] if j != i and abs(corr[i, j]) > 0.0]
        for k, j in enumerate(picked[:K]):
            topk[i, k] = j
    return topk, corr


_FEATURE_RE = re.compile(r"Feature\s+(\d+):\s*(.*)")


def parse_cpp_output(text, F):
    """Return (topk array, K). Reads 'Feature i: j0 j1 ...' lines."""
    rows = {}
    for line in text.splitlines():
        m = _FEATURE_RE.match(line.strip())
        if not m:
            continue
        ints = [int(t) for t in m.group(2).split() if t.isdigit()]
        if ints:
            rows[int(m.group(1))] = ints

    if len(rows) != F:
        raise ValueError(f"expected {F} feature lines, got {len(rows)}")

    K = len(rows[0])
    topk = np.zeros((F, K), dtype=np.int64)
    for i in range(F):
        topk[i] = rows[i][:K]
    return topk, K


def compare(cpp_topk, ref_topk, corr):
    """Number of feature rows that fail the tolerance check."""
    F, _ = cpp_topk.shape
    failures = 0
    for i in range(F):
        cpp = {int(x) for x in cpp_topk[i] if int(x) != SENTINEL}
        ref = {int(x) for x in ref_topk[i] if int(x) != SENTINEL}
        if cpp == ref:
            continue

        # The k-th reference |corr| sets the tolerance band; any
        # disagreeing index whose |corr| is within tolerance of it is a
        # tie, not a bug. If the reference row is empty, the threshold
        # is 0 and no disagreement is acceptable.
        ref_sorted = sorted(ref, key=lambda j: -abs(corr[i, j]))
        threshold = abs(corr[i, ref_sorted[-1]]) if ref_sorted else 0.0
        diff = cpp ^ ref
        if all(np.isclose(abs(corr[i, j]), threshold, rtol=RTOL, atol=ATOL)
               for j in diff):
            continue

        failures += 1
        if failures <= 5:
            print(f"  feature {i}: cpp={sorted(cpp)} ref={sorted(ref)} "
                  f"kth_ref_|corr|={threshold:.6f}", file=sys.stderr)
            for j in sorted(diff):
                where = "extra" if j in cpp else "missing"
                print(f"    j={j:4d} corr={corr[i, j]:+.6f} {where}",
                      file=sys.stderr)
    return failures


def main():
    if len(sys.argv) != 3:
        print("Usage: reference.py <input.bin> <cpp_stdout.txt>", file=sys.stderr)
        return 2

    X, starts, ends = load_input(Path(sys.argv[1]))
    text = Path(sys.argv[2]).read_text()
    F, S = X.shape

    cpp_topk, K = parse_cpp_output(text, F)
    ref_topk, corr = compute_reference(X, starts, ends, K)
    failures = compare(cpp_topk, ref_topk, corr)

    status = "OK" if failures == 0 else f"FAIL ({failures} rows)"
    print(f"[check] F={F} S={S} K={K}  {status}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
