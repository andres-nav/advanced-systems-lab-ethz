# Advanced Systems Lab (ETH Zurich) — Optimizing Interval-Masked Top-K Pearson Correlation

Team 41 project for the ETH Zurich *Advanced Systems Lab* course (263-2300). We optimize a
single-core numerical kernel through a structured sequence of optimizations and analyze it
against the machine's roofline. The final implementation reaches **10.4 flops/cycle (~33% of
the AVX2+FMA single-precision peak) on a Kaby Lake i7-7700HQ**, a **53×** speedup over the
logical baseline when features span 80% of each row (`LONG`) and **26×** when they span 30%
(`SHORT`).

Full write-up: [`report/report.pdf`](report/report.pdf) · slides: [`presentation/slides-export.pdf`](presentation/slides-export.pdf)

## What it computes

Given an input matrix `X` of shape `F × S` (F features, S samples per feature) plus a per-feature
valid interval `[start[f], end[f])`, for every feature we find the `K = 16` other features with
the highest absolute Pearson correlation, computed **only over the overlapping valid interval**
of each pair. Three phases:

1. **Preprocessing** — per-feature mean/std (numerically stable variance) and z-score
   normalization over the valid window.
2. **Overlap correlation** — for each pair `(f0, f1)` with `f1 > f0`, a dot product of z-scores
   over the overlap `[max(start), min(end))`. Only the upper triangle is computed (symmetry).
   This is ~96% of the runtime and is the optimization target.
3. **Streaming top-K** — a size-K min-heap per feature; the full `F × F` matrix is never materialized.

## Optimization variants

Each directory under `src/variants/` is an incremental step with its own compile flags
(`flags.mk`):

| Variant | Optimization | Flags |
|---|---|---|
| `0_baseline` | Naive reference implementation | `-O0` |
| `1_algorithm` | Logical baseline: matrix symmetry, stable variance, z-score precompute, min-heap | `-O0` |
| `2_auto_vectorization` | Compiler auto-vectorization | `-O3 -march=native` |
| `2.1_manual_ilp_no_vectorization` | Manual ILP, auto-vectorizer disabled (ablation) | `… -ffast-math -fno-tree-vectorize` |
| `3_ffast_math_ilp` | Fast-math + ILP | `-O3 -march=native -ffast-math` |
| `4_manual_ilp` | 4 independent scalar accumulators (breaks the loop-carried dependency) | `… -ffast-math` |
| `5_manual_vectorization` | Explicit AVX2/FMA (8-wide), zero-padded masked dot product | `… -mavx -mfma` |
| `6_blocking` | 4×2 register blocking for cache reuse | `… -mavx -mfma` |
| `7_multi_accumulator` | 8 accumulators (4×2 block) to saturate both FMA units | `… -mavx -mfma` |
| `8_sorting` | Sort features by interval start → early loop exit + contiguous access (new contribution) | `… -mavx -mfma` |

## Quick start

Requirements: `g++` (C++20), `libpapi`, `python3` (numpy/matplotlib for the tools), and a Linux
host with AVX2/FMA. `config.mk` is git-ignored — copy it from the template first.

```bash
cp config.mk.example config.mk        # then edit F/S/VARIANT/CORE as needed
make compile VARIANT=8_sorting
make run     VARIANT=8_sorting F=512 S=512 INTERVAL=LONG DATA=GAUSSIAN
make check   VARIANT=8_sorting        # diff top-K against the NumPy reference (tools/reference.py)
make bench   VARIANT=8_sorting REP=3  # PAPI-counter benchmark (pinned via taskset)
make sweep                            # full parameter sweep -> results/
make report                           # regenerate figures + build report/report.pdf
```

Run `make help` to list all targets (`remote-*` targets run a sweep on a benchmark host over ssh).

## Layout

```
src/         kernel (main.cpp, common/) and the optimization variants
tools/       benchmark, sweep, plotting, roofline, and NumPy reference scripts
report/      LaTeX source, figures, and report.pdf
presentation/  Slidev deck (slides.md) and PDF export
results/     benchmark data from the sweeps
docs/        project description and baseline pseudocode
```

## Authors

Oliver Bergqvist · Andres Navarro Pedregal · Luhao Liu — Team 41, Department of Computer
Science, ETH Zurich.
