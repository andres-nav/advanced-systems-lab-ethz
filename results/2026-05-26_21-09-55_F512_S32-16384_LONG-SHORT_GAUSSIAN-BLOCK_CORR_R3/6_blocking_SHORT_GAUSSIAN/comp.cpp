#include <algorithm>
#include <cassert>
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

// REQUIRES: F is a multiple of 4
// REQUIRES: S is a multiple of 8
// REQUIRES: starts[i] and ends[i] are multiples of 8 for all i
void computation(const RawInData *data, Scratch *scratch, uint32_t *topK) {
  assert(data->F % 4 == 0 && "F must be a multiple of 4 for block processing");
  assert(data->S % 8 == 0 && "S must be a multiple of 8 for AVX processing");

  uint32_t F = data->F;
  uint32_t S = data->S;
  float *X = data->X;
  uint32_t *starts = data->starts;
  uint32_t *ends = data->ends;

  float *normalized_X = scratch->normalized_X;
  HeapElement *topK_heaps = scratch->topK_heaps;

  // Phase 1: normalize with AVX + multiple accumulators
  {
    PROFILE("phase1");
    for (uint32_t i = 0; i < F; i++) {
      uint32_t start = starts[i];
      uint32_t end = ends[i];
      uint32_t n = end - start;
      uint32_t aligned8_start = start & ~7u;
      uint32_t aligned8_end = (end + 7u) & ~7u;

      __m256 v_sum = _mm256_setzero_ps();
      __m256 v_sum_sq = _mm256_setzero_ps();

      const float *base_ptr = &X[i * S + aligned8_start];
      for (uint32_t j = aligned8_start; j < aligned8_end;
           j += 8, base_ptr += 8) {
        __m256 v_x = _mm256_load_ps(base_ptr);
        v_sum = _mm256_add_ps(v_sum, v_x);
        v_sum_sq = _mm256_fmadd_ps(v_x, v_x, v_sum_sq);
      }
      float sum = hsum256(v_sum);
      float sum_sq = hsum256(v_sum_sq);

      float mean = sum / n;
      float var =
          (((n * mean) * mean) - ((2.0f * sum) * mean) + sum_sq) / (n - 1);
      float std = sqrt(var + STD_EPS);
      float inv_std = 1.0f / std;

      __m256 v_mean = _mm256_set1_ps(mean);
      __m256 v_inv_std = _mm256_set1_ps(inv_std);

      base_ptr = &X[i * S + aligned8_start];
      float *norm_ptr = &normalized_X[i * S + aligned8_start];
      for (uint32_t j = aligned8_start; j < aligned8_end;
           j += 8, base_ptr += 8, norm_ptr += 8) {
        __m256 v_x = _mm256_load_ps(base_ptr);
        __m256 v_norm = _mm256_mul_ps(_mm256_sub_ps(v_x, v_mean), v_inv_std);
        _mm256_store_ps(norm_ptr, v_norm);
      }
    }
  }

  // Phase 2: blocked by 4 i-features, single j at a time, 4 accumulators
  {
    PROFILE("phase2");
    for (uint32_t i = 0; i < F; i += 4) {
      // Diagonal block: pairs within the 4-feature block
      {
        uint32_t l01 = std::max(starts[i], starts[i + 1]);
        uint32_t l02 = std::max(starts[i], starts[i + 2]);
        uint32_t l03 = std::max(starts[i], starts[i + 3]);
        uint32_t l12 = std::max(starts[i + 1], starts[i + 2]);
        uint32_t l13 = std::max(starts[i + 1], starts[i + 3]);
        uint32_t l23 = std::max(starts[i + 2], starts[i + 3]);
        uint32_t r01 = std::min(ends[i], ends[i + 1]);
        uint32_t r02 = std::min(ends[i], ends[i + 2]);
        uint32_t r03 = std::min(ends[i], ends[i + 3]);
        uint32_t r12 = std::min(ends[i + 1], ends[i + 2]);
        uint32_t r13 = std::min(ends[i + 1], ends[i + 3]);
        uint32_t r23 = std::min(ends[i + 2], ends[i + 3]);
        uint32_t n01 = r01 - l01, n02 = r02 - l02, n03 = r03 - l03;
        uint32_t n12 = r12 - l12, n13 = r13 - l13, n23 = r23 - l23;

        uint32_t min_l = std::min({l01, l02, l03, l12, l13, l23});
        uint32_t max_r = std::max({r01, r02, r03, r12, r13, r23});

        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc02 = _mm256_setzero_ps();
        __m256 acc03 = _mm256_setzero_ps();
        __m256 acc12 = _mm256_setzero_ps();
        __m256 acc13 = _mm256_setzero_ps();
        __m256 acc23 = _mm256_setzero_ps();

        const float *base_ptr = &normalized_X[i * S + min_l];
        for (uint32_t t = min_l; t < max_r; t += 8, base_ptr += 8) {
          __m256 v_i0 = _mm256_load_ps(base_ptr);
          __m256 v_i1 = _mm256_load_ps(base_ptr + S);
          __m256 v_i2 = _mm256_load_ps(base_ptr + 2 * S);
          __m256 v_i3 = _mm256_load_ps(base_ptr + 3 * S);
          acc01 = _mm256_fmadd_ps(v_i0, v_i1, acc01);
          acc02 = _mm256_fmadd_ps(v_i0, v_i2, acc02);
          acc03 = _mm256_fmadd_ps(v_i0, v_i3, acc03);
          acc12 = _mm256_fmadd_ps(v_i1, v_i2, acc12);
          acc13 = _mm256_fmadd_ps(v_i1, v_i3, acc13);
          acc23 = _mm256_fmadd_ps(v_i2, v_i3, acc23);
        }

        if (n01 > 0) {
          float tmp = std::fabs(hsum256(acc01) / (n01 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 1);
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i);
        }
        if (n02 > 0) {
          float tmp = std::fabs(hsum256(acc02) / (n02 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 2);
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i);
        }
        if (n03 > 0) {
          float tmp = std::fabs(hsum256(acc03) / (n03 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i);
        }
        if (n12 > 0) {
          float tmp = std::fabs(hsum256(acc12) / (n12 - 1));
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i + 2);
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i + 1);
        }
        if (n13 > 0) {
          float tmp = std::fabs(hsum256(acc13) / (n13 - 1));
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i + 1);
        }
        if (n23 > 0) {
          float tmp = std::fabs(hsum256(acc23) / (n23 - 1));
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i + 2);
        }
      }

      // Off-diagonal: 4 i-features vs each j outside the block
      for (uint32_t j = i + 4; j < F; j++) {
        uint32_t l0 = std::max(starts[i], starts[j]);
        uint32_t l1 = std::max(starts[i + 1], starts[j]);
        uint32_t l2 = std::max(starts[i + 2], starts[j]);
        uint32_t l3 = std::max(starts[i + 3], starts[j]);
        uint32_t r0 = std::min(ends[i], ends[j]);
        uint32_t r1 = std::min(ends[i + 1], ends[j]);
        uint32_t r2 = std::min(ends[i + 2], ends[j]);
        uint32_t r3 = std::min(ends[i + 3], ends[j]);
        uint32_t n0 = r0 - l0, n1 = r1 - l1, n2 = r2 - l2, n3 = r3 - l3;

        uint32_t min_l = std::min({l0, l1, l2, l3});
        uint32_t max_r = std::max({r0, r1, r2, r3});

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const float *base_ptr_i = &normalized_X[i * S + min_l];
        const float *base_ptr_j = &normalized_X[j * S + min_l];
        for (uint32_t t = min_l; t < max_r;
             t += 8, base_ptr_i += 8, base_ptr_j += 8) {
          __m256 v_j = _mm256_load_ps(base_ptr_j);
          __m256 v_i0 = _mm256_load_ps(base_ptr_i);
          __m256 v_i1 = _mm256_load_ps(base_ptr_i + S);
          __m256 v_i2 = _mm256_load_ps(base_ptr_i + 2 * S);
          __m256 v_i3 = _mm256_load_ps(base_ptr_i + 3 * S);
          acc0 = _mm256_fmadd_ps(v_i0, v_j, acc0);
          acc1 = _mm256_fmadd_ps(v_i1, v_j, acc1);
          acc2 = _mm256_fmadd_ps(v_i2, v_j, acc2);
          acc3 = _mm256_fmadd_ps(v_i3, v_j, acc3);
        }

        if (n0 > 0) {
          float v = std::fabs(hsum256(acc0) / (n0 - 1));
          heap_insert(&topK_heaps[i * K], v, j);
          heap_insert(&topK_heaps[j * K], v, i);
        }
        if (n1 > 0) {
          float v = std::fabs(hsum256(acc1) / (n1 - 1));
          heap_insert(&topK_heaps[(i + 1) * K], v, j);
          heap_insert(&topK_heaps[j * K], v, i + 1);
        }
        if (n2 > 0) {
          float v = std::fabs(hsum256(acc2) / (n2 - 1));
          heap_insert(&topK_heaps[(i + 2) * K], v, j);
          heap_insert(&topK_heaps[j * K], v, i + 2);
        }
        if (n3 > 0) {
          float v = std::fabs(hsum256(acc3) / (n3 - 1));
          heap_insert(&topK_heaps[(i + 3) * K], v, j);
          heap_insert(&topK_heaps[j * K], v, i + 3);
        }
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
