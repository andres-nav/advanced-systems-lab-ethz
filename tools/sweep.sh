#!/bin/bash
# Orchestrate a sweep across variants × intervals × data regimes × F × S
# into a single timestamped run directory under $RESULTS_DIR.
#
# Run dir name is auto-encoded from the sweep params:
#   <RESULTS_DIR>/<TS>_F<f_tag>_S<s_tag>_<I_tag>_<D_tag>_R<rep>/
# where f_tag/s_tag are "MIN-MAX" for >1 values or just the value, and
# I_tag/D_tag are the regimes joined with '-'.
#
# Inside that dir, each (variant, interval, data) cell is a flat
# subdirectory <variant>_<INTERVAL>_<DATA>/ containing bench.csv,
# flops.csv, bytes.csv, perf.csv, comp.cpp, flags.mk. The top-level dir
# holds env.txt and a copy of config.mk.
#
# All configuration comes via environment variables (set by the
# Makefile). See REQUIRED below.

set -euo pipefail

# -------------------------------------------------------------------
# Required env vars from the Makefile.
# -------------------------------------------------------------------
REQUIRED=(
    VARIANTS SWEEP_F SWEEP_S SWEEP_INTERVAL SWEEP_DATA
    REP SEED CORE
    PYTHON BUILD_DIR RESULTS_DIR LOAD_MAX
    BASE_FLAGS CXX MAKE
    BENCH_HEADER
)
for v in "${REQUIRED[@]}"; do
    if [ -z "${!v:-}" ]; then
        echo "sweep.sh: required env var \$$v is not set" >&2
        exit 2
    fi
done

# -------------------------------------------------------------------
# Pre-flight: bail if the box is busy.
# -------------------------------------------------------------------
load=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l>m)}'; then
    echo "[sweep] aborting: load $load > LOAD_MAX=$LOAD_MAX" >&2
    exit 1
fi

# -------------------------------------------------------------------
# Build the auto-encoded run dir name.
#   F-tag / S-tag: "MIN-MAX" for >1 values, "VALUE" for a single value.
#   I-tag / D-tag: regimes joined with '-'.
# -------------------------------------------------------------------
range_tag() {
    local label=$1; shift
    if [ "$#" -le 1 ]; then echo "${label}$1"; return; fi
    local min=$1 max=$1 v
    for v in "$@"; do
        [ "$v" -lt "$min" ] && min=$v
        [ "$v" -gt "$max" ] && max=$v
    done
    echo "${label}${min}-${max}"
}

list_tag() { local IFS='-'; echo "$*"; }

ts=$(date +%Y-%m-%d_%H-%M-%S)
# shellcheck disable=SC2086 # word-splitting is intentional
f_tag=$(range_tag F $SWEEP_F)
# shellcheck disable=SC2086
s_tag=$(range_tag S $SWEEP_S)
# shellcheck disable=SC2086
i_tag=$(list_tag $SWEEP_INTERVAL)
# shellcheck disable=SC2086
d_tag=$(list_tag $SWEEP_DATA)

run_dir="$RESULTS_DIR/${ts}_${f_tag}_${s_tag}_${i_tag}_${d_tag}_R${REP}"
mkdir -p "$run_dir"
cp config.mk "$run_dir/" 2>/dev/null || true

# Mirror this script's own stderr to run_dir/sweep.log so post-mortem
# debugging is possible after the tmux session is gone, while the
# original stderr (tmux pane / terminal) still sees the same output
# live.
exec 2> >(tee -a "$run_dir/sweep.log" >&2)

# -------------------------------------------------------------------
# env.txt — host/compiler/CPU/governor/temp at start, freq+temp at end.
# -------------------------------------------------------------------
env_file=$run_dir/env.txt

temp_c() {
    for n in /sys/class/hwmon/*/name; do
        if [ "$(cat "$n" 2>/dev/null)" = coretemp ]; then
            local t
            t=$(cat "$(dirname "$n")/temp1_input" 2>/dev/null || true)
            [ -n "$t" ] && { echo "$((t / 1000)) C"; return; }
        fi
    done
    echo "n/a"
}

{
    echo "started:  $(date -Iseconds)"
    echo "host:     $(uname -n)"
    "$CXX" --version | head -1
    echo "flags:    $BASE_FLAGS"
    echo "variants: $VARIANTS"
    echo "F:        $SWEEP_F"
    echo "S:        $SWEEP_S"
    echo "INTERVAL: $SWEEP_INTERVAL"
    echo "DATA:     $SWEEP_DATA"
    echo "REP:      $REP"
    echo "SEED:     $SEED"
    echo "CORE:     $CORE"
    (lscpu 2>/dev/null | head -10) || grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
    echo "governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
    echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)"
    echo "cpu${CORE} freq: $(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo n/a) kHz"
    echo "temp:     $(temp_c)"
} > "$env_file"

# -------------------------------------------------------------------
# Sweep loop: for each variant build once, then loop interval × data
# × F × S.
# -------------------------------------------------------------------
PIN="setarch $(uname -m) -R taskset -c $CORE"
TEST_INPUT="$BUILD_DIR/test_input.bin"

variant_count=$(echo "$VARIANTS" | wc -w | tr -d ' ')
cells_per_variant=$(( $(echo "$SWEEP_INTERVAL" | wc -w) * $(echo "$SWEEP_DATA" | wc -w) ))
points_per_cell=$(( $(echo "$SWEEP_F" | wc -w) * $(echo "$SWEEP_S" | wc -w) ))
total_cells=$(( variant_count * cells_per_variant ))
cell=0
v_idx=0

for v in $VARIANTS; do
    v_idx=$((v_idx + 1))
    echo "[sweep] ($v_idx/$variant_count) building $v ..." >&2
    # Build via the Makefile's existing rule. _rebuild forces a fresh
    # compile so we never reuse a stale binary.
    "$MAKE" compile VARIANT="$v" BASE_FLAGS="$BASE_FLAGS" >&2

    # Pull FLOPS_VARIANT out of the variant's flags.mk for tools/flops.py.
    flops_variant=$(grep -E '^[[:space:]]*FLOPS_VARIANT[[:space:]]*=' \
                       "src/variants/$v/flags.mk" | head -1 | \
                       sed -E 's/^[^=]*=[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$flops_variant" ]; then
        echo "[sweep] $v: FLOPS_VARIANT not set in src/variants/$v/flags.mk" >&2
        exit 1
    fi

    for interval in $SWEEP_INTERVAL; do
        for data in $SWEEP_DATA; do
            cell=$((cell + 1))
            cell_dir="$run_dir/${v}_${interval}_${data}"
            mkdir -p "$cell_dir"
            cp "src/variants/$v/comp.cpp" "src/variants/$v/flags.mk" "$cell_dir/"

            bench_csv="$cell_dir/bench.csv"
            flops_csv="$cell_dir/flops.csv"
            bytes_csv="$cell_dir/bytes.csv"
            log_txt="$cell_dir/log.txt"
            echo "$BENCH_HEADER" > "$bench_csv"
            echo "F,S,flops"     > "$flops_csv"
            echo "F,S,bytes"     > "$bytes_csv"
            : > "$log_txt"

            point=0
            for f in $SWEEP_F; do
                for s in $SWEEP_S; do
                    point=$((point + 1))
                    echo "[sweep cell $cell/$total_cells point $point/$points_per_cell $(date +%H:%M:%S)]" \
                         "$v $interval $data F=$f S=$s" >&2

                    # Tag the cell log with the (F, S) point so a
                    # post-mortem grep finds the failing step quickly.
                    echo "=== F=$f S=$s ===" >> "$log_txt"

                    "$PYTHON" tools/gen_input.py --out "$TEST_INPUT" --seed "$SEED" \
                        --F "$f" --S "$s" --interval "$interval" --data "$data" >/dev/null 2>>"$log_txt"
                    "$PYTHON" tools/flops.py "$TEST_INPUT" --variant "$flops_variant" >> "$flops_csv" 2>>"$log_txt"
                    "$PYTHON" tools/bytes.py "$TEST_INPUT" --variant "$flops_variant" >> "$bytes_csv" 2>>"$log_txt"

                    # shellcheck disable=SC2086 # PIN must word-split
                    $PIN "./$BUILD_DIR/$v" "$TEST_INPUT" --bench --rep "$REP" \
                        >> "$bench_csv" 2>>"$log_txt"
                done
            done
        done
    done

    # Cool-down between variants (keeps the existing sweep-all behaviour
    # of giving the CPU a chance to settle). Skip after the last variant.
    if [ "$v_idx" -lt "$variant_count" ]; then
        echo "[sweep] cooling down 30s before next variant ..." >&2
        sleep 30
    fi
done

{
    echo
    echo "finished: $(date -Iseconds)"
    echo "cpu${CORE} freq: $(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo n/a) kHz"
    echo "temp:     $(temp_c)"
} >> "$env_file"

echo "[sweep] done — wrote $run_dir/" >&2
echo "$run_dir"
