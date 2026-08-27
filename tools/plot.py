#!/usr/bin/env python3
"""Plot for a sweep produced by `make sweep`.

Reads bench.csv (F, S, cycles, trial) and flops.csv (F, S, flops) from
the given run directory, picks F or S as x-axis (whichever varies),
writes performance.pdf into the same directory.

Usage:
    plot.py --run results/unoptimized/2026-05-01_12-00-36
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles

import matplotlib.pyplot as plt

# Scalar single-issue peak on Skylake-family without FMA: 1 addss + 1 mulss
# per cycle on separate ports = 2 flop/cycle. With FMA it's the same number
# (1 FMA = 2 flops per cycle); with AVX2+FMA it's 16.
SCALAR_PEAK = 2.0
PDF_NAME = "performance.pdf"


def load_csv(path: Path, key_cols, val_col, cast=float):
    """Group rows by (key_cols) tuple → list of val_col values."""
    out = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = tuple(int(row[c]) for c in key_cols)
            out[key].append(cast(row[val_col]))
    return out


def iqr(samples):
    """Return (p25, p75). For <2 samples, returns (x, x)."""
    if len(samples) < 2:
        return samples[0], samples[0]
    q = quantiles(samples, n=4, method="inclusive")
    return q[0], q[2]


def plot(run_dir: Path) -> None:
    bench_csv = run_dir / "bench.csv"
    flops_csv = run_dir / "flops.csv"
    for p in (bench_csv, flops_csv):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    cycles = load_csv(bench_csv, ("F", "S"), "cycles")
    flops = {k: v[0] for k, v in load_csv(flops_csv, ("F", "S"), "flops", int).items()}

    # Pick the varying axis for x; the other parameterizes the lines.
    fs = sorted({k[0] for k in cycles})
    ss = sorted({k[1] for k in cycles})
    x_axis, line_axis, xi, li = ("S", "F", 1, 0) if len(ss) >= len(fs) else ("F", "S", 0, 1)

    # {line_value: {x_value: [flop/cycle samples]}}
    lines = defaultdict(dict)
    for key, cyc_samples in cycles.items():
        lines[key[li]][key[xi]] = [flops[key] / c for c in cyc_samples]

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    _, ax = plt.subplots(figsize=(4.2, 3.2))
    cmap = plt.get_cmap("tab10")

    for i, lv in enumerate(sorted(lines)):
        xs = sorted(lines[lv])
        med = [median(lines[lv][x]) for x in xs]
        lo = [iqr(lines[lv][x])[0] for x in xs]
        hi = [iqr(lines[lv][x])[1] for x in xs]
        color = cmap(i % 10)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, med, "o-", color=color,
                label=f"{line_axis}={lv}", linewidth=1.8, markersize=4)

    ax.axhline(SCALAR_PEAK, color="grey", linestyle="--", linewidth=1)
    ax.text(
        ax.get_xlim()[1], SCALAR_PEAK,
        f" scalar peak = {SCALAR_PEAK:g} flop/cycle",
        va="center", ha="right", color="grey", fontsize=8,
    )

    ax.set_xlabel(x_axis)
    ax.set_xscale("log", base=2)
    x_ticks = sorted({k[xi] for k in cycles})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.minorticks_off()

    ax.set_ylabel("Performance [flop/cycle]")
    ax.set_ylim(bottom=0, top=max(SCALAR_PEAK * 1.1, ax.get_ylim()[1]))
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)

    if len(lines) > 1:
        ax.legend(frameon=False, fontsize=8)

    ax.set_title("Performance")
    plt.tight_layout()

    out_path = run_dir / PDF_NAME
    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    
    out_path_svg = run_dir / PDF_NAME.replace(".pdf", ".svg")
    plt.savefig(out_path_svg, bbox_inches="tight", format="svg")
    print(f"wrote {out_path_svg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path,
                    help="path to a run directory (results/<variant>/<ts>/)")
    args = ap.parse_args()
    if not args.run.is_dir():
        raise SystemExit(f"not a directory: {args.run}")
    plot(args.run)


if __name__ == "__main__":
    main()
