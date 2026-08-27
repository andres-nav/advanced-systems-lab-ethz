#!/usr/bin/env python3
"""Analytical memory traffic (bytes transferred) for roofline analysis.

Conventions:
    - float = 4 bytes, uint32 = 4 bytes
    - Count reads and writes of arrays (cold-cache model)
    - Integer index arrays (starts, ends) not counted (negligible)

Baseline:
    Phase 1: read X[i,start:end] once for sum, once for variance = 2*n_i*4 (compulsory misses only for first pass)
             write mean_vals[i] + std_sample_vals[i] = 2*4 per feature
    Compulsory Misses: n_i*4 + 2*4 per feature
    
    Phase 2: for each valid pair (i,j) with i!=j: read X[i] and X[j] over overlap = 2*n_ij*4 (non-compulsory misses)
             write corr_matrix[i,j] = 4
    Compulsory Misses: 4 per valid pair
    
    Phase 3: read corr_matrix row (F*4 per feature, K passes) = F*F*4
             write topK = F*K*4
    Compulsory Misses: F*K*4

Baseline optimized:
    Phase 1: read X[i,start:end] once = n_i*4
             write normalized_X[i,start:end] = n_i*4
    Compulsory Misses: 2*n_i*4
    
    Phase 2: for each valid pair (i<j): read normalized_X[i] and normalized_X[j] = 2*n_ij*4 (non-compulsory misses)
             heap insertions: F*K*8 in total (each HeapElement is 8 bytes)
    Compulsory Misses: F*K*8
    
    Phase 3: write topK = F*K*4 (heap already in cache)
    Compulsory Misses: F*K*4
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def count_bytes(starts: np.ndarray, ends: np.ndarray, variant: str = "baseline") -> int:
    """Analytical byte count for given intervals."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    F = int(starts.shape[0])
    K = 16 # constant
    n = ends - starts

    if variant == "baseline":
        # Phase 1: 2 passes over X + write mean/std
        phase1 = int(n.sum()) * 4 + 2 * F * 4 # compulsory misses only for first pass

        # Phase 2: full matrix (i != j), read 2 streams + write corr entry
        L = np.maximum(starts[:, None], starts[None, :])
        R = np.minimum(ends[:, None], ends[None, :])
        overlap = np.maximum(R - L, 0)
        np.fill_diagonal(overlap, 0)
        valid = overlap >= 2
        phase2 = int(valid.sum()) * 4 # compulsory misses only

        # Phase 3: read corr_matrix + write topK
        phase3 = F * K * 4 # compulsory misses only for topK

    elif variant == "baseline_optimized":
        # Phase 1: read X + write normalized_X
        phase1 = 2 * int(n.sum()) * 4 # compulsory misses

        # # Phase 2: upper triangle only, read 2 normalized streams
        # L = np.maximum(starts[:, None], starts[None, :])
        # R = np.minimum(ends[:, None], ends[None, :])
        # overlap = np.maximum(R - L, 0)
        # mask = np.triu(np.ones((F, F), dtype=bool), k=1)
        # valid = (overlap >= 2) & mask
        # phase2 = int((2 * overlap[valid]).sum()) * 4
        phase2 = F * K * 8 # compulsory misses only for heap insertions

        # Phase 3: write topK
        phase3 = F * K * 4 # compulsory misses only for topK

    else:
        raise ValueError(f"unknown variant: {variant}")

    return phase1 + phase2 + phase3


def from_bin(path: Path, variant: str = "baseline") -> tuple[int, int, int]:
    """Load test_input.bin and return (bytes, F, S)."""
    with path.open("rb") as f:
        F, S = struct.unpack("<II", f.read(8))
        starts = np.frombuffer(f.read(4 * F), dtype=np.int32).astype(np.int64, copy=True)
        ends = np.frombuffer(f.read(4 * F), dtype=np.int32).astype(np.int64, copy=True)
    return count_bytes(starts, ends, variant), F, S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="path to test_input.bin")
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "baseline_optimized"],
                    help="memory traffic model to use")
    args = ap.parse_args()

    nbytes, F, S = from_bin(args.input, args.variant)
    print(f"{F},{S},{nbytes}")
    return 0


if __name__ == "__main__":
    main()
