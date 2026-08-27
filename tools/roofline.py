#!/usr/bin/env python3
"""Roofline plot across multiple variants.

Reads the latest sweep results for each variant in results/, computes
operational intensity (flops/byte) and performance (flops/cycle), and
plots them on a roofline diagram.

Usage:
    roofline.py --results results/ [--F 128] [--S 128]
    roofline.py --results results/ --variants 0_baseline 1_baseline_optimized

Machine parameters (Kaby Lake i7-7700HQ, single core, turbo off):
    - Scalar peak: 2 flop/cycle (1 add + 1 mul)
    - AVX2 FMA peak: 32 flop/cycle (2 FMA units × 8 floats × 2 flops)
    - Memory bandwidth: ~8 bytes/cycle (measured STREAM ~34 GB/s @ 4.2 GHz)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

# Machine ceilings (adjust for your hardware)
PEAK_SCALAR = 2.0        # flop/cycle: 1 add + 1 mul
PEAK_SCALAR_FMA = 4.0    # flop/cycle: 1 fma + 1 fma = 4 flops
PEAK_AVX = 16.0          # flop/cycle: 2 AVX units × 8 floats (no FMA)
PEAK_AVX_FMA = 32.0      # flop/cycle: 2 FMA units × 8 floats × 2 flops
MEM_BW = 7.0             # bytes/cycle (adjust: measure with STREAM)

PDF_NAME = "roofline.pdf"


def load_csv_grouped(path: Path, key_cols, val_col, cast=float):
    out = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = tuple(int(row[c]) for c in key_cols)
            out[key].append(cast(row[val_col]))
    return out


def latest_run(variant_dir: Path) -> Path | None:
    runs = sorted(variant_dir.iterdir())
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"),
                    help="results directory")
    ap.add_argument("--variants", nargs="*",
                    help="specific variants to plot (default: all found)")
    ap.add_argument("--F", type=int, help="filter to specific F value")
    ap.add_argument("--S", type=int, help="filter to specific S value")
    ap.add_argument("-o", "--output", type=Path, help="output PDF path")
    args = ap.parse_args()

    if args.variants:
        variant_names = args.variants
    else:
        variant_names = sorted(
            d.name for d in args.results.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    data_points = []  # (variant_name, oi, perf)

    for vname in variant_names:
        vdir = args.results / vname
        if not vdir.is_dir():
            print(f"warning: {vdir} not found, skipping")
            continue
        run_dir = latest_run(vdir)
        if not run_dir:
            continue

        bench_csv = run_dir / "bench.csv"
        flops_csv = run_dir / "flops.csv"
        bytes_csv = run_dir / "bytes.csv"

        if not bench_csv.exists() or not flops_csv.exists():
            continue
        if not bytes_csv.exists():
            print(f"warning: no bytes.csv in {run_dir}, skipping")
            continue

        cycles = load_csv_grouped(bench_csv, ("F", "S"), "cycles")
        flops = {k: v[0] for k, v in load_csv_grouped(flops_csv, ("F", "S"), "flops", int).items()}
        nbytes = {k: v[0] for k, v in load_csv_grouped(bytes_csv, ("F", "S"), "bytes", int).items()}

        for key in cycles:
            f_val, s_val = key
            if args.F and f_val != args.F:
                continue
            if args.S and s_val != args.S:
                continue
            if key not in flops or key not in nbytes:
                continue

            fl = flops[key]
            by = nbytes[key]
            med_cycles = median(cycles[key])

            oi = fl / by  # flops/byte
            perf = fl / med_cycles  # flops/cycle

            data_points.append((vname, oi, perf))

    if not data_points:
        raise SystemExit("no data points found")

    # Plot
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(6, 4))

    # Roofline ceilings
    oi_range = np.logspace(-2, 4, 500)
    # Memory-bound ceiling: perf = BW * OI
    mem_ceiling = MEM_BW * oi_range
    # Compute-bound ceilings
    for peak, label, ls in [
        (PEAK_SCALAR, "scalar peak", "--"),
        (PEAK_SCALAR_FMA, "scalar peak (fma)", "-."),
        (PEAK_AVX, "AVX2 peak", ":"),
        (PEAK_AVX_FMA, "AVX2 peak (fma)", "-"),
    ]:
        roof = np.minimum(mem_ceiling, peak)
        ax.plot(oi_range, roof, color="grey", linestyle=ls, linewidth=1, alpha=0.7)
        # Label at the flat part
        ax.text(oi_range[-1], peak, f" {label} ({peak:g})",
                va="center", ha="left", color="grey", fontsize=7)

    # Plot data points grouped by variant
    cmap = plt.get_cmap("tab10")
    variants_seen = {}
    for vname, oi, perf in data_points:
        if vname not in variants_seen:
            variants_seen[vname] = len(variants_seen)
        idx = variants_seen[vname]
        color = cmap(idx % 10)
        ax.scatter(oi, perf, color=color, s=40, zorder=5,
                   label=vname if vname not in [d[0] for d in data_points[:data_points.index((vname, oi, perf))]] else "")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    if unique:
        ax.legend(*zip(*unique), frameon=False, fontsize=7, loc="lower right")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Operational Intensity [flops/byte]")
    ax.set_ylabel("Performance [flops/cycle]")
    ax.set_title("Roofline")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = args.output or (args.results / PDF_NAME)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
