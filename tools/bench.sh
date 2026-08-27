#!/bin/bash
# Put the machine into (or out of) a stable benchmarking state.
# Linux-only, requires sudo.
#
# Usage:
#   bench.sh setup   [SIBLING]   # performance governor, turbo off, sibling offline
#   bench.sh restore [SIBLING]   # dynamic governor,    turbo on,  sibling online
#
# SIBLING is the logical CPU id of the SMT sibling of the benchmark core.
# Look it up with:
#   cat /sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list
#
# For passwordless invocation (so users without general sudo access can still run
# this), create an 'asl' group and add authorized users to the 'asl' group, then install
# /etc/sudoers.d/asl via 'sudo vi /etc/sudoers.d/asl' containing:
#   %asl ALL=(root) NOPASSWD: \
#       /usr/bin/tee /sys/devices/system/cpu/cpu[0-9]*/online, \
#       /usr/bin/tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor, \
#       /usr/bin/tee /sys/devices/system/cpu/intel_pstate/no_turbo, \
#       /usr/bin/tee /sys/devices/system/cpu/cpufreq/boost, \
#       /usr/bin/tee /proc/sys/kernel/perf_event_paranoid

set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
    echo "bench.sh: Linux-only (current: $(uname -s))" >&2
    exit 1
fi

MODE="${1:-}"
SIBLING="${2:-}"

case "$MODE" in
    setup)
        GOV=performance
        TURBO_OFF=1       # intel_pstate/no_turbo
        BOOST=0           # cpufreq/boost
        SIBLING_ONLINE=0
        PARANOID=0        # /proc/sys/kernel/perf_event_paranoid (PAPI access)
        ;;
    restore)
        # Pick the best available dynamic governor.
        GOV=powersave
        AVAIL=/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
        if [ -f "$AVAIL" ]; then
            for g in schedutil ondemand powersave; do
                if grep -qw "$g" "$AVAIL"; then GOV=$g; break; fi
            done
        fi
        TURBO_OFF=0
        BOOST=1
        SIBLING_ONLINE=1
        PARANOID=2        # kernel default
        ;;
    *)
        echo "Usage: $0 {setup|restore} [SIBLING]" >&2
        exit 1
        ;;
esac

if [ -n "$SIBLING" ] && [ -f /sys/devices/system/cpu/cpu$SIBLING/online ]; then
    echo "$SIBLING_ONLINE" | sudo tee /sys/devices/system/cpu/cpu$SIBLING/online >/dev/null
fi

for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cpu_dir=${gov%/cpufreq/scaling_governor}
    # cpu0 has no 'online' file on most kernels (always online by design).
    if [ ! -f "$cpu_dir/online" ] || [ "$(cat "$cpu_dir/online")" = "1" ]; then
        echo "$GOV" | sudo tee "$gov" >/dev/null
    fi
done

if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "$TURBO_OFF" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
fi

if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo "$BOOST" | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
fi

if [ -f /proc/sys/kernel/perf_event_paranoid ]; then
    echo "$PARANOID" | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null
fi
