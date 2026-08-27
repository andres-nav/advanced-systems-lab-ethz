# Benchmark configuration. 

# --- what gets built and what input is generated ---
VARIANT  = 8_sorting
# VARIANT  = 1_baseline_optimized
F        = 1024
S        = 8192
SEED     = 0
# INTERVAL: LONG | SHORT
# INTERVAL = LONG
INTERVAL = SHORT
# DATA: GAUSSIAN | BLOCK_CORR
DATA     = GAUSSIAN

# --- sweep variants ---
# SWEEP_VARIANTS = 0_baseline 1_algorithm 2_auto_vectorization 3_ffast_math_ilp 4_manual_ilp 5_manual_vectorization 6_blocking 7_multi_accumulator 8_sorting
SWEEP_VARIANTS = 0_baseline

# SWEEP_F  = 8 16 32 64 128 256 512 1024 2048 4096 8192
SWEEP_F  = 8 16 32 64 128 256 512 1024 2048
SWEEP_S  = 512

# SWEEP_F  = 512
# SWEEP_S  = 32 64 128 256 512 1024 2048 4096 8192 16384

# SWEEP_INTERVAL = LONG SHORT
SWEEP_INTERVAL = LONG SHORT
# SWEEP_DATA     = GAUSSIAN BLOCK_CORR
SWEEP_DATA     = GAUSSIAN BLOCK_CORR

# --- measurement ---
# Trials per (F, S) point. Low (~1) for quick dev sweeps; 10+ for
# plots that go in the final report.
REP      = 3

# --- benchmarking ---
# Core the benchmark binary is pinned to with taskset. BENCH_SIBLING is
# its SMT sibling, offlined so nothing else runs
# on the same physical core. Check topology with:
#   cat /sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list
CORE          ?= 3
BENCH_SIBLING ?= 7

# --- remote ---
# Host to ssh into for `make remote-*` targets.
REMOTE_HOST ?= user@host
REMOTE_DIR  ?= ~/asl-project/
