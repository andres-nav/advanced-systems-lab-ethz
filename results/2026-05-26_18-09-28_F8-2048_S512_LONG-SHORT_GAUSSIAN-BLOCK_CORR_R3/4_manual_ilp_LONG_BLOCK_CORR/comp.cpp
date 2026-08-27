#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/comp.hpp"
#include "common/microbench.hpp"
#include "common/parameters.hpp"
#include "common/parser.hpp"

// Min-Heap Element
struct HeapElement {
  float val;
  uint32_t idx;
};

static inline void heap_insert(HeapElement *root, float val, uint32_t idx) {
  if (val <= root[0].val) {
    return;
  }

  root[0].val = val;
  root[0].idx = idx;
  uint32_t i = 0;
  while (true) {
    uint32_t left = 2 * i + 1;
    uint32_t right = 2 * i + 2;
    uint32_t smallest = i;

    if (left < K && root[left].val < root[smallest].val) {
      smallest = left;
    }
    if (right < K && root[right].val < root[smallest].val) {
      smallest = right;
    }
    if (smallest == i) {
      break;
    }

    std::swap(root[i], root[smallest]);
    i = smallest;
  }
}

static inline void heap_to_topK(HeapElement *root, uint32_t *topK_row) {
  for (uint32_t k = 0; k < K; ++k) {
    uint32_t j = K - k - 1;

    topK_row[j] = root[0].idx;
    root[0] = root[j];
    uint32_t i = 0;
    while (true) {
      uint32_t left = 2 * i + 1;
      uint32_t right = 2 * i + 2;
      uint32_t smallest = i;

      if (left < j && root[left].val < root[smallest].val) {
        smallest = left;
      }
      if (right < j && root[right].val < root[smallest].val) {
        smallest = right;
      }
      if (smallest == i) {
        break;
      }

      std::swap(root[i], root[smallest]);
      i = smallest;
    }
  }
}

static inline void heap_init(HeapElement *root) {
  for (uint32_t k = 0; k < K; ++k) {
    root[k].val = 0.0f;
    root[k].idx = SENTINEL;
  }
}

struct Scratch {
  float *normalized_X = nullptr;
  HeapElement *topK_heaps = nullptr;
};

Scratch *scratch_create(const RawInData *data) {
  Scratch *s = new (std::nothrow) Scratch();
  if (!s)
    return nullptr;

  size_t f_size = static_cast<size_t>(data->F);
  size_t s_size = static_cast<size_t>(data->S);

  s->normalized_X = aligned_alloc_padded<float>(f_size * s_size);
  s->topK_heaps = (HeapElement *)malloc(sizeof(HeapElement) * f_size * K);

  if (!s->normalized_X || !s->topK_heaps) {
    scratch_destroy(s);
    return nullptr;
  }

  return s;
}

void scratch_reset(const RawInData *data, Scratch *scratch) {
  size_t f_size = static_cast<size_t>(data->F);
  size_t s_size = static_cast<size_t>(data->S);

  memset(scratch->normalized_X, 0, f_size * s_size * sizeof(float));

  for (uint32_t i = 0; i < f_size; ++i) {
    heap_init(&scratch->topK_heaps[i * K]);
  }

  clflush_range(scratch->normalized_X, f_size * s_size * sizeof(float));
  clflush_range(scratch->topK_heaps, f_size * K * sizeof(HeapElement));
  clflush_range(scratch, sizeof(*scratch));
}

void scratch_destroy(Scratch *scratch) {
  if (!scratch)
    return;
  free(scratch->normalized_X);
  free(scratch->topK_heaps);
  delete scratch;
}

void computation(const RawInData *data, Scratch *scratch, uint32_t *topK) {
  uint32_t F = data->F;
  uint32_t S = data->S;
  float *X = data->X;
  uint32_t *starts = data->starts;
  uint32_t *ends = data->ends;

  float *normalized_X = scratch->normalized_X;
  HeapElement *topK_heaps = scratch->topK_heaps;

  // Phase 1: normalize with 4 accumulators for ILP
  {
    PROFILE("phase1");
    for (uint32_t i = 0; i < F; i++) {
      uint32_t start = starts[i];
      uint32_t end = ends[i];
      uint32_t n = end - start;

      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;

      uint32_t j = start;
      uint32_t end4 = start + (n & ~3u);
      for (; j < end4; j += 4) {
        float x0 = X[i * S + j];
        float x1 = X[i * S + j + 1];
        float x2 = X[i * S + j + 2];
        float x3 = X[i * S + j + 3];
        sum0 += x0;
        sum1 += x1;
        sum2 += x2;
        sum3 += x3;
        sq0 += x0 * x0;
        sq1 += x1 * x1;
        sq2 += x2 * x2;
        sq3 += x3 * x3;
      }
      for (; j < end; j++) {
        sum0 += X[i * S + j];
        sq0 += X[i * S + j] * X[i * S + j];
      }

      float sum = sum0 + sum1 + sum2 + sum3;
      float sum_sq = sq0 + sq1 + sq2 + sq3;

      float mean = sum / n;
      float var =
          (((n * mean) * mean) - ((2.0f * sum) * mean) + sum_sq) / (n - 1);
      float std = sqrt(var + STD_EPS);

      for (uint32_t j = start; j < end; j++) {
        normalized_X[i * S + j] = (X[i * S + j] - mean) / std;
      }
    }
  }

  // Phase 2: pairwise correlation with 4 accumulators for ILP
  {
    PROFILE("phase2");
    for (uint32_t i = 0; i < F; i++) {
      for (uint32_t j = i + 1; j < F; j++) {
        uint32_t start = std::max(starts[i], starts[j]);
        uint32_t end = std::min(ends[i], ends[j]);

        uint32_t n = end - start;
        if (n < 2) {
          continue;
        }

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        uint32_t k = start;
        uint32_t end4 = start + (n & ~3u);
        for (; k < end4; k += 4) {
          acc0 += normalized_X[i * S + k] * normalized_X[j * S + k];
          acc1 += normalized_X[i * S + k + 1] * normalized_X[j * S + k + 1];
          acc2 += normalized_X[i * S + k + 2] * normalized_X[j * S + k + 2];
          acc3 += normalized_X[i * S + k + 3] * normalized_X[j * S + k + 3];
        }
        for (; k < end; k++) {
          acc0 += normalized_X[i * S + k] * normalized_X[j * S + k];
        }

        float corr = (acc0 + acc1 + acc2 + acc3) / (n - 1);
        float v = std::fabs(corr);

        heap_insert(&topK_heaps[i * K], v, j);
        heap_insert(&topK_heaps[j * K], v, i);
      }
    }
  }

  // Phase 3: materialize top-K from heaps.
  {
    PROFILE("phase3");
    for (uint32_t i = 0; i < F; ++i) {
      heap_to_topK(&topK_heaps[i * K], &topK[i * K]);
    }
  }
}
