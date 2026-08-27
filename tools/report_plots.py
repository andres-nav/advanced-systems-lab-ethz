#!/usr/bin/env python3
"""Performance and roofline figures for the report.

Usage:
    tools/report_plots.py --data GAUSSIAN
    tools/report_plots.py --data BLOCK_CORR
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Machine constants: Kaby Lake i7-7700HQ, single core, 2.8 GHz, turbo off.
PEAK_SCALAR = 2.0  # 1 add + 1 mul / cycle
PEAK_AVX_FMA = 32.0  # 2 FMA units x 8 fp32 lanes x 2 flop
MEM_BW = 7.0  # B/cycle, from a 19.8 GB/s single-core STREAM triad
L2_BW = 32.0
L1_BW = 64.0

# Ladder variant dir -> short line label, in the order they should stack.
LADDER = [
    ("1_algorithm", "Logical baseline"),
    ("2.1_manual_ilp_no_vectorization", "Manual ILP (no vec)"),
    ("4_manual_ilp", "Manual ILP (auto-vec)"),
    ("5_manual_vectorization", "Explicit vectorization"),
    ("7_multi_accumulator", "Multi-accumulator"),
    ("8_sorting", "Sorting"),
]

COLORS = ["#7a7a7a", "#5a7fb0", "#4a9b8e", "#c98a3a", "#7b6aa8", "#b22222"]
BEST = len(LADDER) - 1


def load_grouped(path, val_col, cast=float):
    """{(F, S): [values]} for one column of a sweep CSV."""
    out = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(int(row["F"]), int(row["S"]))].append(cast(row[val_col]))
    return out


def cyc_col(path):
    # PAPI sweeps use papi_tot_cyc; the old RDTSC ones used cycles.
    with open(path) as f:
        header = next(csv.reader(f))
    if "papi_tot_cyc" in header:
        return "papi_tot_cyc"
    if "cycles" in header:
        return "cycles"
    raise SystemExit(f"{path}: no cycle column")


def resolve_cell(results, variant, regime, data, sweep):
    name = f"{variant}_{regime}_{data}"
    found = []
    for run in sorted(results.iterdir()):
        d = run / name
        if not (d / "flops.csv").exists():
            continue
        fs, ss = set(), set()
        for row in csv.DictReader(open(d / "flops.csv")):
            fs.add(int(row["F"]))
            ss.add(int(row["S"]))
        is_f_sweep = len(fs) > len(ss)
        if is_f_sweep == (sweep == "F"):
            found.append((run.name, d))
    if not found:
        return None
    d = sorted(found)[-1][1]  # run dirs are timestamp-prefixed
    return d / "bench.csv", d / "flops.csv", d / "bytes.csv"


def collect(results, regime, data, sweep):
    xi = 0 if sweep == "F" else 1
    series = []
    for i, (variant, label) in enumerate(LADDER):
        cell = resolve_cell(results, variant, regime, data, sweep)
        if cell is None:
            print(f"  missing: {variant} {regime} {data} ({sweep}-sweep)")
            continue
        bench, flops_csv, bytes_csv = cell
        cyc = load_grouped(bench, cyc_col(bench))
        flops = {k: v[0] for k, v in load_grouped(flops_csv, "flops", int).items()}
        nbytes = {k: v[0] for k, v in load_grouped(bytes_csv, "bytes", int).items()}

        pts = []
        for key, samples in cyc.items():
            if key not in flops:
                continue
            perf = flops[key] / median(samples)
            oi = flops[key] / nbytes[key] if key in nbytes else None
            pts.append((key[xi], oi, perf))
        pts.sort()
        if pts:
            series.append((label, COLORS[i], i == BEST, pts))
    return series


def style_axes(ax):
    ax.set_facecolor("#eaeaea")
    ax.grid(True, color="white", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(length=0, labelsize=11)


def frame(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.annotate(
        "[flops/cycle]",
        xy=(0, 1),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=26)


def label_lines(ax, entries):
    items = [
        (np.log2(ys[-1]), xs[-1], ys[-1], text, color, weight)
        for xs, ys, text, color, weight in entries
    ]
    items.sort()  # by line-end height, low to high

    min_gap = 0.50  # log2 units
    label_y = [ylog + 0.12 for ylog, *_ in items]
    for i in range(1, len(label_y)):
        if label_y[i] - label_y[i - 1] < min_gap:
            label_y[i] = label_y[i - 1] + min_gap

    xmax = max(x for _, x, *_ in items)
    for (_, xend, yend, text, color, weight), ly in zip(items, label_y):
        ax.plot(
            [xend, xmax],
            [yend, 2.0**ly],
            color=color,
            linewidth=0.6,
            alpha=0.6,
            zorder=3,
            clip_on=False,
        )
        ax.annotate(
            text,
            xy=(xmax, 2.0**ly),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=color,
            fontweight=weight,
            zorder=8,
            annotation_clip=False,
        )


def perf_plot(series, sweep, title_extra, out_base):
    if not series:
        return
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    style_axes(ax)

    entries = []
    for label, color, best, pts in series:
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            linewidth=2.8 if best else 1.6,
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=6 if best else 4,
        )
        entries.append((xs, ys, label, color, "bold" if best else "normal"))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(2**-6, 2**6)

    ax.axhline(PEAK_AVX_FMA, color="#555555", linestyle="--", linewidth=1, zorder=2)
    ax.annotate(
        "AVX2+FMA peak (32)",
        xy=(ax.get_xlim()[0], PEAK_AVX_FMA),
        xytext=(2, 2),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="#555555",
    )

    label_lines(ax, entries)
    xlabel = "F (number of features)" if sweep == "F" else "S (samples per feature)"
    frame(ax, xlabel, f"Performance {title_extra}")

    fig.subplots_adjust(left=0.10, right=0.74, top=0.84, bottom=0.13)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"  wrote {out_base}.pdf")
    plt.close(fig)


def roofline_plot(series, title_extra, out_base):
    if not series:
        return
    all_oi = [oi for _, _, _, pts in series for _, oi, _ in pts if oi]
    all_perf = [p for _, _, _, pts in series for _, _, p in pts]
    if not all_oi:
        print(f"  skip {out_base}: no OI data")
        return

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    style_axes(ax)

    x_lo = 2.0**-3
    x_hi = 2.0 ** np.ceil(np.log2(max(all_oi)) + 1)
    oi = np.logspace(np.log2(x_lo), np.log2(x_hi), 400, base=2)

    # Label each slanted memory roof on its own line, inside the visible area.
    # We aim for a low height y_target on the line (x = y_target / bw); if that
    # x would fall left of the axis we clamp it to the left edge x_lo, which
    # slides the label up its own line but keeps it on screen. This is robust
    # to the axis width changing between the F- and S-sweeps.
    for bw, lab, y_target in [
        (MEM_BW, "DRAM 7 B/cyc", 3.0),
        (L2_BW, "L2 32 B/cyc", 3.0),
        (L1_BW, "L1 64 B/cyc", 3.0),
    ]:
        ax.plot(
            oi,
            np.minimum(bw * oi, PEAK_AVX_FMA),
            color="#999999",
            linestyle=":",
            linewidth=1,
            zorder=2,
        )
        x_lab = max(y_target / bw, x_lo)
        ax.annotate(
            lab,
            xy=(x_lab, bw * x_lab),
            xytext=(2, 3),
            textcoords="offset points",
            rotation=38,
            rotation_mode="anchor",
            va="bottom",
            ha="left",
            fontsize=7.5,
            color="#888888",
            zorder=3,
        )

    # Horizontal compute peaks; emphasise the AVX2+FMA one.
    for peak, lab, ls, col in [
        (PEAK_SCALAR, "scalar peak (2)", "--", "#999999"),
        (PEAK_AVX_FMA, "AVX2+FMA peak (32)", "-", "#555555"),
    ]:
        ax.plot(
            oi,
            np.full_like(oi, peak),
            color=col,
            linestyle=ls,
            linewidth=1.2 if peak == PEAK_AVX_FMA else 1,
            zorder=2,
        )
        ax.annotate(
            lab,
            xy=(x_lo, peak),
            xytext=(2, 2),
            textcoords="offset points",
            va="bottom",
            ha="left",
            fontsize=8.5,
            color=col,
        )

    entries = []
    for label, color, best, pts in series:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            linewidth=2.8 if best else 1.6,
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=6 if best else 4,
        )
        entries.append((xs, ys, label, color, "bold" if best else "normal"))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(
        2.0 ** np.floor(np.log2(min(all_perf)) - 0.5),
        2.0 ** np.ceil(np.log2(PEAK_AVX_FMA) + 0.6),
    )

    label_lines(ax, entries)
    frame(
        ax,
        "Operational intensity [flops/byte]",
        f"Roofline {title_extra}",
    )

    fig.subplots_adjust(left=0.10, right=0.74, top=0.84, bottom=0.13)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"  wrote {out_base}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--outdir", type=Path, default=Path("report/figures/generated"))
    ap.add_argument("--data", default="GAUSSIAN", choices=["GAUSSIAN", "BLOCK_CORR"])
    args = ap.parse_args()

    plt.rcParams["font.family"] = "Helvetica"
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.data == "GAUSSIAN" else f"_{args.data}"
    titles = {"LONG": "LONG (80% valid)", "SHORT": "SHORT (30% valid)"}

    print(f"data generator: {args.data}")
    for regime in ("LONG", "SHORT"):
        name = titles[regime]
        out = args.outdir

        # F-sweep, S fixed at 512.
        s = collect(args.results, regime, args.data, "F")
        perf_plot(s, "F", f"{name}, S=512", out / f"perf_{regime}{suffix}")
        roofline_plot(s, f"{name}, S=512", out / f"roofline_{regime}{suffix}")

        # S-sweep, F fixed at 512.
        s = collect(args.results, regime, args.data, "S")
        perf_plot(s, "S", f"{name}, F=512", out / f"perf_S_{regime}{suffix}")
        roofline_plot(s, f"{name}, F=512", out / f"roofline_S_{regime}{suffix}")


if __name__ == "__main__":
    main()
