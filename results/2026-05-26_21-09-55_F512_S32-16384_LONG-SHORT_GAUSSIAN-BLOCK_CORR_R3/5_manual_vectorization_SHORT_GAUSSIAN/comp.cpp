#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

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

static inline float hsum256(__m256 v) {
  alignas(32) float temp[8];
  _mm256_store_ps(temp, v);
  return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] +
         temp[7];
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

  // Phase 1: normalize with AVX vectorization
  {
    PROFILE("phase1");
    for (uint32_t i = 0; i < F; i++) {
      uint32_t start = starts[i];
      uint32_t end = ends[i];
      uint32_t n = end - start;

      __m256 v_sum = _mm256_setzero_ps();
      __m256 v_sum_sq = _mm256_setzero_ps();

      uint32_t j = start;
      uint32_t end8 = start + (n & ~7u);
      for (; j < end8; j += 8) {
        __m256 v_x = _mm256_loadu_ps(&X[i * S + j]);
        v_sum = _mm256_add_ps(v_sum, v_x);
        v_sum_sq = _mm256_fmadd_ps(v_x, v_x, v_sum_sq);
      }
      float sum = hsum256(v_sum);
      float sum_sq = hsum256(v_sum_sq);
      for (; j < end; j++) {
        float x = X[i * S + j];
        sum += x;
        sum_sq += x * x;
      }

      float mean = sum / n;
      float var =
          (((n * mean) * mean) - ((2.0f * sum) * mean) + sum_sq) / (n - 1);
      float std = sqrt(var + STD_EPS);
      float inv_std = 1.0f / std;

      __m256 v_mean = _mm256_set1_ps(mean);
      __m256 v_inv_std = _mm256_set1_ps(inv_std);

      j = start;
      for (; j < end8; j += 8) {
        __m256 v_x = _mm256_loadu_ps(&X[i * S + j]);
        __m256 v_norm = _mm256_mul_ps(_mm256_sub_ps(v_x, v_mean), v_inv_std);
        _mm256_storeu_ps(&normalized_X[i * S + j], v_norm);
      }
      for (; j < end; j++) {
        normalized_X[i * S + j] = (X[i * S + j] - mean) * inv_std;
      }
    }
  }

  // Phase 2: pairwise correlation with AVX, single accumulator
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

        __m256 v_acc = _mm256_setzero_ps();

        uint32_t k = start;
        uint32_t end8 = start + (n & ~7u);
        for (; k < end8; k += 8) {
          __m256 v_i = _mm256_loadu_ps(&normalized_X[i * S + k]);
          __m256 v_j = _mm256_loadu_ps(&normalized_X[j * S + k]);
          v_acc = _mm256_fmadd_ps(v_i, v_j, v_acc);
        }
        float acc = hsum256(v_acc);
        for (; k < end; k++) {
          acc += normalized_X[i * S + k] * normalized_X[j * S + k];
        }

        float corr = acc / (n - 1);
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
