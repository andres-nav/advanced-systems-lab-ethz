#!/usr/bin/env python3
"""Baseline flop count.

The flop count is dependent on the given 'baseline algorithm' and kept the same.

Conventions (state these in the report):
    add, sub, mul, div, sqrt : 1 flop each
    fabs, neg                 : 0 flops (bitwise)
    integer ops, comparisons  : 0 flops

Phase breakdown:
    phase 1 (mean + std per feature):        4*n_i + 4      per feature
    phase 2 (corr over overlap):             6*n_ij + 1     per valid pair
    phase 3 (top-k by |corr|):               0              (fabs + compares)
    
Phase breakdown for baseline optimized:
    phase 1 (normalized_X):                  5*n_i + 10     per feature
    phase 2 (corr over overlap):             2*n_ij + 1     per valid pair
    phase 3 (heap materilization):           0              
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def count_flops(starts: np.ndarray, ends: np.ndarray, variant: str = "baseline") -> int:
    """Exact flop count for intervals `starts[i]..ends[i]`."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    F = int(starts.shape[0])
    n = ends - starts
    assert np.all(n >= 0), "negative interval lengths"

    if variant == "baseline":
        # Phase 1: two passes (sum then variance): 4*n_i + 4 per feature
        phase1 = 4 * int(n.sum()) + 4 * F

        # Phase 2: full matrix, 6*n_ij + 1 per valid pair (i != j, n_ij >= 2)
        L = np.maximum(starts[:, None], starts[None, :])
        R = np.minimum(ends[:, None], ends[None, :])
        overlap = np.maximum(R - L, 0)
        np.fill_diagonal(overlap, 0)
        valid = overlap >= 2
        phase2 = int((6 * overlap[valid]).sum() + valid.sum())

    elif variant == "baseline_optimized":
        # Phase 1: single pass sum+sum_sq (2*n_i adds + n_i muls = 3*n_i),
        #          then mean(1 div) + var(7 ops) + std(2 ops) = 10,
        #          then normalize: 2*n_i (sub + mul) per element
        phase1 = 5 * int(n.sum()) + 10 * F

        # Phase 2: upper triangle only, dot product: 2*n_ij + 1 per valid pair
        L = np.maximum(starts[:, None], starts[None, :])
        R = np.minimum(ends[:, None], ends[None, :])
        overlap = np.maximum(R - L, 0)
        # Upper triangle only
        mask = np.triu(np.ones((F, F), dtype=bool), k=1)
        valid = (overlap >= 2) & mask
        phase2 = int((2 * overlap[valid]).sum() + valid.sum())

    else:
        raise ValueError(f"unknown variant: {variant}")

    return phase1 + phase2


def from_bin(path: Path, variant: str = "baseline") -> tuple[int, int, int]:
    """Load a test_input.bin and return (flops, F, S)."""
    with path.open("rb") as f:
        F, S = struct.unpack("<II", f.read(8))
        starts = np.frombuffer(f.read(4 * F), dtype=np.int32).astype(np.int64, copy=True)
        ends = np.frombuffer(f.read(4 * F), dtype=np.int32).astype(np.int64, copy=True)
    return count_flops(starts, ends, variant), F, S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="path to test_input.bin")
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "baseline_optimized"],
                    help="flop model to use (default: baseline)")
    args = ap.parse_args()

    flops, F, S = from_bin(args.input, args.variant)
    print(f"{F},{S},{flops}")
    return 0


if __name__ == "__main__":
    main()
