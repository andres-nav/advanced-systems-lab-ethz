#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

PEAK_SCALAR = 2.0
PEAK_AVX_FMA = 32.0

# Kaby Lake i7-7700HQ single-core cache hierarchy.
BYTES_PER_ELEM = 4  # fp32
CACHE_BYTES = [("L1", 32 * 1024), ("L2", 256 * 1024), ("LLC", 6 * 1024 * 1024)]


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
    ap.add_argument("--output-prefix", default="perf_step",
                    help="prefix for output SVG filenames (default: perf_step)")
    args = ap.parse_args()

    if args.variants:
        variant_names = args.variants
    else:
        variant_names = sorted(
            d.name for d in args.results.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    perf_by_variant: dict[str, dict[int, float]] = {}
    x_label = "S"
    fixed_val: int | None = None  # value of the non-swept dimension (F or S)

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
        if not bench_csv.exists() or not flops_csv.exists():
            continue

        cycles = load_csv_grouped(bench_csv, ("F", "S"), cyc_col(bench_csv))
        flops = {k: v[0] for k, v in
                 load_csv_grouped(flops_csv, ("F", "S"), "flops", int).items()}

        fs = sorted({k[0] for k in cycles})
        ss = sorted({k[1] for k in cycles})
        xi = 1 if len(ss) >= len(fs) else 0
        x_label = "S" if xi == 1 else "F"
        fixed_val = (fs[0] if xi == 1 else ss[0])  # the held-constant dimension

        line: dict[int, float] = {}
        for key, cyc in cycles.items():
            if key not in flops:
                continue
            line[key[xi]] = flops[key] / median(cyc)
        if line:
            perf_by_variant[vname] = line

    ordered = [v for v in variant_names if v in perf_by_variant]
    if not ordered:
        raise SystemExit("no data points found")

    cmap = plt.get_cmap("tab10")

    for step in range(1, len(ordered) + 1):
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
        fig, ax = plt.subplots(figsize=(6, 4))

        for idx, vname in enumerate(ordered[:step]):
            color = cmap(idx % 10)
            xs = sorted(perf_by_variant[vname])
            ys = [perf_by_variant[vname][x] for x in xs]
            alpha = 1.0 if idx == step - 1 else 0.3
            ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=4,
                    alpha=alpha, label=vname, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel(f"Problem size {x_label} (samples per feature)")
        ax.set_ylabel("Performance [flops/cycle]")
        ax.set_title(f"Performance Scaling (Up to {ordered[step - 1]})")
        ax.grid(True, color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(2**-5, 2**6)

        # peak reference lines (x in axes fraction, y in data coords)
        for peak, label in [(PEAK_SCALAR, "scalar peak (2)"),
                            (PEAK_AVX_FMA, "AVX2 FMA peak (32)")]:
            ax.axhline(peak, color="grey", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(0.99, peak, f"{label} ", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", color="grey", fontsize=7)

        # Vertical markers where the F*S working set (fp32) outgrows each cache
        # level. Working set = fixed_val * x * BYTES_PER_ELEM, so the crossing is
        # at x = cache_bytes / (fixed_val * BYTES_PER_ELEM).
        all_x = sorted({x for v in ordered[:step] for x in perf_by_variant[v]})
        if fixed_val and all_x:
            x_min, x_max = all_x[0], all_x[-1]
            for cname, cbytes in CACHE_BYTES:
                cross = cbytes / (fixed_val * BYTES_PER_ELEM)
                if not (x_min <= cross <= x_max):
                    continue
                ax.axvline(cross, color="0.55", linestyle=":", linewidth=1,
                           zorder=1)
                ax.text(cross, 2**6, f" {cname}", color="0.45", fontsize=6.5,
                        va="top", ha="left", rotation=90)

        ax.legend(frameon=False, fontsize=7, loc="upper left")
        plt.tight_layout()
        out_svg = args.results / f"{args.output_prefix}{step}.svg"
        plt.savefig(out_svg, bbox_inches="tight", format="svg")
        if step == len(ordered):
            plt.savefig(args.results / f"{args.output_prefix}{step}.pdf", bbox_inches="tight")
        print(f"wrote {out_svg}")
        plt.close(fig)


if __name__ == "__main__":
    main()
