// Modes:
//   default : compute top-K once, print indices to stdout.
//   --bench : run benchmarking with pre-warming, CSV to stdout.
//
// Bench mode brackets each kernel call with PAPI_read() and accumulates
// per-event deltas per batch. The eventset is multiplexed so events
// that compete for the same PMC slots can still all be scheduled (PAPI
// time-shares the counters and scales the values). Default 10-event
// set covers IPC + full cache hierarchy + TLB + branches + FLOPs +
// load/store traffic. Override via PAPI_EVENTS env var (comma-separated
// PAPI preset names); PAPI_TOT_CYC must be present (used as the cycle
// reference for batch calibration and CSV output).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <papi.h>

#include "common/comp.hpp"
#include "common/microbench.hpp"
#include "common/parameters.hpp"
#include "common/parser.hpp"

// A measurement batch must cover at least this many cycles. With ~1
// cycle counter resolution and per-run noise of a few thousand cycles,
// batches need to be big to reduce variability.
static constexpr uint64_t CYCLES_REQUIRED = 100'000'000ULL; // 1e8
static constexpr long WARMUP_START = 100;                   // initial batch
static constexpr double WARMUP_GROW = 2.0;                  // min scale-up

// 10 events. PAPI_TOT_CYC + PAPI_TOT_INS are fixed-counter (free); the
// others contend for 4 GP slots, so the eventset is multiplexed.
static const char *DEFAULT_PAPI_EVENTS =
    "PAPI_TOT_CYC,PAPI_TOT_INS,"
    "PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM,"
    "PAPI_RES_STL,"
    "PAPI_BR_INS,PAPI_BR_MSP,"
    "PAPI_SP_OPS,PAPI_LST_INS";

struct PapiContext {
  int eventset = PAPI_NULL;
  int cyc_idx = -1; // index of PAPI_TOT_CYC inside `names`
  std::vector<std::string> names;
  std::vector<long long> sums;
  std::vector<long long> pre;
  std::vector<long long> post;
};

static bool papi_init(PapiContext &ctx) {
  int ret = PAPI_library_init(PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI_library_init failed: %s\n", PAPI_strerror(ret));
    return false;
  }
  ret = PAPI_multiplex_init();
  if (ret != PAPI_OK) {
    fprintf(stderr, "PAPI_multiplex_init failed: %s\n", PAPI_strerror(ret));
    return false;
  }
  ret = PAPI_create_eventset(&ctx.eventset);
  if (ret != PAPI_OK) {
    fprintf(stderr, "PAPI_create_eventset failed: %s\n", PAPI_strerror(ret));
    return false;
  }
  // Multiplexing must be configured on an empty eventset that has been
  // assigned to a component (0 = the CPU PMU).
  ret = PAPI_assign_eventset_component(ctx.eventset, 0);
  if (ret != PAPI_OK) {
    fprintf(stderr, "PAPI_assign_eventset_component failed: %s\n",
            PAPI_strerror(ret));
    return false;
  }
  ret = PAPI_set_multiplex(ctx.eventset);
  if (ret != PAPI_OK) {
    fprintf(stderr, "PAPI_set_multiplex failed: %s\n", PAPI_strerror(ret));
    return false;
  }

  const char *env = std::getenv("PAPI_EVENTS");
  const char *events = env ? env : DEFAULT_PAPI_EVENTS;
  char buf[1024];
  std::strncpy(buf, events, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  char *save = nullptr;
  for (char *tok = strtok_r(buf, ",", &save); tok;
       tok = strtok_r(nullptr, ",", &save)) {
    while (*tok == ' ' || *tok == '\t')
      ++tok;
    char *end = tok + std::strlen(tok);
    while (end > tok && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n'))
      *--end = '\0';
    if (*tok == '\0')
      continue;
    ret = PAPI_add_named_event(ctx.eventset, tok);
    if (ret != PAPI_OK) {
      fprintf(stderr, "[papi] skipping %s: %s\n", tok, PAPI_strerror(ret));
      continue;
    }
    if (std::strcmp(tok, "PAPI_TOT_CYC") == 0) {
      ctx.cyc_idx = static_cast<int>(ctx.names.size());
    }
    ctx.names.emplace_back(tok);
  }
  if (ctx.cyc_idx < 0) {
    fprintf(stderr, "PAPI: PAPI_TOT_CYC must be in the event list\n");
    return false;
  }
  ctx.sums.assign(ctx.names.size(), 0);
  ctx.pre.assign(ctx.names.size(), 0);
  ctx.post.assign(ctx.names.size(), 0);

  ret = PAPI_start(ctx.eventset);
  if (ret != PAPI_OK) {
    fprintf(stderr, "PAPI_start failed: %s\n", PAPI_strerror(ret));
    return false;
  }
  return true;
}

static void papi_destroy(PapiContext &ctx) {
  if (ctx.eventset == PAPI_NULL)
    return;
  std::vector<long long> discard(ctx.names.size(), 0);
  PAPI_stop(ctx.eventset, discard.data());
  PAPI_cleanup_eventset(ctx.eventset);
  PAPI_destroy_eventset(&ctx.eventset);
  PAPI_shutdown();
}

// Run `n` back-to-back kernel calls. PAPI deltas accumulate into
// ctx.sums (zeroed on entry); returns the PAPI_TOT_CYC sum so the
// existing batch-calibration logic still works. Between iterations we
// memset topK, scratch_reset (which flushes the variant's scratch),
// and clflush every input buffer so each timed call starts with cold
// data caches; clflush_range issues MFENCE so all flushes have retired
// before the kernel runs.
static uint64_t time_batch(const RawInData *data, Scratch *scratch,
                           uint32_t *topK, size_t topK_bytes, long n,
                           PapiContext &ctx) {
  const size_t X_bytes = static_cast<size_t>(data->F) * data->S * sizeof(float);
  const size_t idx_bytes = static_cast<size_t>(data->F) * sizeof(uint32_t);
  std::fill(ctx.sums.begin(), ctx.sums.end(), 0);

  for (long i = 0; i < n; ++i) {
    std::memset(topK, 0, topK_bytes);
    scratch_reset(data, scratch);
    clflush_range(topK, topK_bytes);
    clflush_range(data->X, X_bytes);
    clflush_range(data->starts, idx_bytes);
    clflush_range(data->ends, idx_bytes);
    clflush_range(data, sizeof(*data));
    PAPI_read(ctx.eventset, ctx.pre.data());
    computation(data, scratch, topK);
    PAPI_read(ctx.eventset, ctx.post.data());
    for (size_t k = 0; k < ctx.sums.size(); ++k) {
      ctx.sums[k] += ctx.post[k] - ctx.pre[k];
    }
    PROFILE_BUMP_CALL();
  }
  return static_cast<uint64_t>(ctx.sums[ctx.cyc_idx]);
}

// Pick a batch size whose total cycles cover CYCLES_REQUIRED, then
// discard one batch at that size to reach steady state.
static long calibrate_batch(const RawInData *data, Scratch *scratch,
                            uint32_t *topK, size_t topK_bytes,
                            PapiContext &ctx) {
  long n = WARMUP_START;
  for (;;) {
    uint64_t cycles = time_batch(data, scratch, topK, topK_bytes, n, ctx);
    if (cycles >= CYCLES_REQUIRED)
      break;
    double scale =
        (double)CYCLES_REQUIRED / (double)std::max<uint64_t>(cycles, 1);
    if (scale < WARMUP_GROW)
      scale = WARMUP_GROW;
    n = std::max<long>(n + 1, (long)((double)n * scale));
  }
  // Discard one batch at the final size.
  time_batch(data, scratch, topK, topK_bytes, n, ctx);
  return n;
}

static void run_default(const RawInData *data, Scratch *scratch,
                        uint32_t *topK) {
  scratch_reset(data, scratch);
  computation(data, scratch, topK);
  PROFILE_BUMP_CALL();
#ifndef MICROBENCH
  printf("Top %u results:\n", K);
  for (uint32_t i = 0; i < data->F; ++i) {
    printf("Feature %u: ", i);
    for (uint32_t k = 0; k < K; ++k) {
      printf("%u ", topK[i * K + k]);
    }
    printf("\n");
  }
  printf("\n");
#endif
}

// Bench mode. One row per trial. Columns: F, S, trial, then every PAPI
// event in the order they were added (per-call averages). The sweep
// script's bench.csv header must match exactly.
static void run_bench(const RawInData *data, Scratch *scratch, uint32_t *topK,
                      size_t topK_bytes, int rep, PapiContext &ctx) {
  const long n = calibrate_batch(data, scratch, topK, topK_bytes, ctx);
  for (int t = 0; t < rep; ++t) {
    time_batch(data, scratch, topK, topK_bytes, n, ctx);
    printf("%u,%u,%d", data->F, data->S, t);
    for (size_t k = 0; k < ctx.sums.size(); ++k) {
      printf(",%.3f", (double)ctx.sums[k] / (double)n);
    }
    printf("\n");
  }
}

struct Args {
  const char *file_path = nullptr;
  bool bench = false;
  int rep = -1;
};

static const char *USAGE = "Usage: %s <input_file> [--bench --rep N]\n"
                           "  --bench requires --rep N (positive integer).\n";

static bool parse_args(int argc, char *argv[], Args *out) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bench") == 0) {
      out->bench = true;
    } else if (std::strcmp(argv[i], "--rep") == 0 && i + 1 < argc) {
      out->rep = std::atoi(argv[++i]);
      if (out->rep <= 0) {
        fprintf(stderr, "Error: --rep must be a positive integer\n");
        return false;
      }
    } else if (argv[i][0] != '-') {
      out->file_path = argv[i];
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      fprintf(stderr, USAGE, argv[0]);
      return false;
    }
  }
  if (!out->file_path) {
    fprintf(stderr, USAGE, argv[0]);
    return false;
  }
  if (out->bench && out->rep <= 0) {
    fprintf(stderr, "Error: --bench requires --rep N\n");
    fprintf(stderr, USAGE, argv[0]);
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    return 1;
  }

  RawInData data = parse_raw_in_data(args.file_path);
  if (!data.X) {
    fprintf(stderr, "Error: Failed to load input data from %s\n",
            args.file_path);
    return 1;
  }

  const size_t topK_n = static_cast<size_t>(data.F) * K;
  const size_t topK_bytes = topK_n * sizeof(uint32_t);
  uint32_t *topK = aligned_alloc_padded<uint32_t>(topK_n);
  if (!topK) {
    fprintf(stderr, "Error: topK allocation failed (F=%u, K=%u)\n", data.F, K);
    free_raw_in_data(&data);
    return 1;
  }

  Scratch *scratch = scratch_create(&data);
  if (!scratch) {
    fprintf(stderr, "Error: scratch_create failed (F=%u, S=%u)\n", data.F,
            data.S);
    free(topK);
    free_raw_in_data(&data);
    return 1;
  }

  int rc = 0;
  if (args.bench) {
    PapiContext ctx;
    if (!papi_init(ctx)) {
      rc = 1;
    } else {
      run_bench(&data, scratch, topK, topK_bytes, args.rep, ctx);
      papi_destroy(ctx);
    }
  } else {
    run_default(&data, scratch, topK);
  }

  scratch_destroy(scratch);
  free(topK);
  free_raw_in_data(&data);
  return rc;
}
