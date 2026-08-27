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
    - DRAM bandwidth: ~7 bytes/cycle (single-core STREAM-triad measured
      19.8 GB/s with the clock locked at 2.8 GHz: 19.8e9 / 2.8e9 = 7.07)
    - L2: 32 bytes/cycle, L1: 64 bytes/cycle (architectural, per-cycle)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

PEAK_SCALAR = 2.0
PEAK_SCALAR_FMA = 4.0
PEAK_AVX = 16.0
PEAK_AVX_FMA = 32.0
MEM_BW = 7.0   # bytes/cycle: single-core STREAM-triad 19.8 GB/s @ 2.8 GHz
L2_BW = 32.0
L1_BW = 64.0

PDF_NAME = "roofline.pdf"


def load_csv_grouped(path: Path, key_cols, val_col, cast=float):
    out = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = tuple(int(row[c]) for c in key_cols)
            out[key].append(cast(row[val_col]))
    return out


def cyc_col(path: Path) -> str:
    """Name of the cycle column in a bench.csv. The PAPI sweep schema uses
    `papi_tot_cyc`; the older RDTSC schema used `cycles`. Prefer PAPI."""
    with path.open() as f:
        header = next(csv.reader(f))
    if "papi_tot_cyc" in header:
        return "papi_tot_cyc"
    if "cycles" in header:
        return "cycles"
    raise SystemExit(f"{path}: no cycle column (papi_tot_cyc / cycles)")


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
    ap.add_argument("--output-prefix", default="roofline_step",
                    help="prefix for output SVG filenames (default: roofline_step)")
    args = ap.parse_args()

    if args.variants:
        variant_names = args.variants
    else:
        variant_names = sorted(
            d.name for d in args.results.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    data_points = []

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

        cycles = load_csv_grouped(bench_csv, ("F", "S"), cyc_col(bench_csv))
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

            oi = fl / by
            perf = fl / med_cycles

            data_points.append((vname, key, oi, perf))

    if not data_points:
        raise SystemExit("no data points found")

    # Which dimension is being swept? The one with more distinct values. Each
    # dot is one input size along it; we connect dots in that order so the
    # progression (and which dot is which input) is legible.
    all_keys = [k for _, k, _, _ in data_points]
    n_f = len({k[0] for k in all_keys})
    n_s = len({k[1] for k in all_keys})
    sweep_i = 0 if n_f >= n_s else 1
    sweep_label = "F" if sweep_i == 0 else "S"

    data_by_variant = defaultdict(list)
    for vname, key, oi, perf in data_points:
        data_by_variant[vname].append((key[sweep_i], oi, perf))
    for v in data_by_variant:
        data_by_variant[v].sort()

    ordered_variants = [v for v in variant_names if v in data_by_variant]
    cmap = plt.get_cmap("tab10")

    oi_hi = max(oi for _, _, oi, _ in data_points)
    x_lo = 2.0 ** -5
    x_hi = 2.0 ** (np.ceil(np.log2(oi_hi)) + 2)

    all_perf = [perf for _, _, _, perf in data_points]
    y_lo = 2.0 ** np.floor(np.log2(min(all_perf)) - 0.5)
    y_hi = 2.0 ** np.ceil(np.log2(max(max(all_perf), PEAK_AVX_FMA)) + 0.5)

    for step in range(1, len(ordered_variants) + 1):
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
        fig, ax = plt.subplots(figsize=(6, 4))

        oi_range = np.logspace(np.log2(x_lo) - 1, np.log2(x_hi) + 1, 600, base=2)
        
        for bw, label, ls in [(MEM_BW, "DRAM BW", "-"), (L2_BW, "L2 BW", "--"), (L1_BW, "L1 BW", ":")]:
            mem_ceiling = np.minimum(bw * oi_range, PEAK_AVX_FMA)
            ax.plot(oi_range, mem_ceiling, color="grey", linestyle=ls, linewidth=1, alpha=0.5)
            text_x = 2**-4
            text_y = bw * text_x
            ax.text(text_x * 1.1, text_y, f"{label}", va="bottom", ha="left", color="grey", fontsize=7, rotation=35)

        mem_ceiling_max = L1_BW * oi_range
        for peak, label, ls in [
            (PEAK_SCALAR, "scalar peak", "--"),
            (PEAK_SCALAR_FMA, "scalar peak (fma)", "-."),
            (PEAK_AVX, "AVX2 peak", ":"),
            (PEAK_AVX_FMA, "AVX2 peak (fma)", "-"),
        ]:
            roof = np.minimum(mem_ceiling_max, peak)
            ax.plot(oi_range, roof, color="grey", linestyle=ls, linewidth=1, alpha=0.7)
            ax.text(0.99, peak, f"{label} ({peak:g}) ", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", color="grey", fontsize=7)

        for idx, vname in enumerate(ordered_variants[:step]):
            color = cmap(idx % 10)
            pts = data_by_variant[vname]          # sorted by swept input value
            svals = [p[0] for p in pts]
            ois = [p[1] for p in pts]
            perfs = [p[2] for p in pts]
            a = 1.0 if idx == step - 1 else 0.3
            n = len(ois)
            # Connect dots in sweep order so the trajectory is readable.
            ax.plot(ois, perfs, "-", color=to_rgba(color, a),
                    linewidth=1.0, zorder=4)
            ax.scatter(ois, perfs, s=40, zorder=5,
                       facecolors=[to_rgba(color, 0.55 * a)] * n,
                       edgecolors=[to_rgba(color, a)] * n, linewidths=0.8,
                       label=vname)
            # Label the endpoints of the highlighted variant with their input
            # size so each dot's input value is identifiable.
            if idx == step - 1 and svals:
                for j, dx in ((0, -5), (n - 1, 5)):
                    ax.annotate(f"{sweep_label}={svals[j]}", (ois[j], perfs[j]),
                                textcoords="offset points", xytext=(dx, 5),
                                fontsize=6, color=to_rgba(color, a),
                                ha="right" if dx < 0 else "left")

        ax.legend(frameon=False, fontsize=7, loc="upper left")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Operational Intensity [flops/byte]")
        ax.set_ylabel("Performance [flops/cycle]")
        ax.set_title(f"Roofline (Up to {ordered_variants[step-1]})")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        plt.tight_layout()
        out_path_svg = args.results / f"{args.output_prefix}{step}.svg"
        plt.savefig(out_path_svg, bbox_inches="tight", format="svg")
        if step == len(ordered_variants):
            plt.savefig(args.results / f"{args.output_prefix}{step}.pdf", bbox_inches="tight")
        print(f"wrote {out_path_svg}")
        plt.close(fig)


if __name__ == "__main__":
    main()
