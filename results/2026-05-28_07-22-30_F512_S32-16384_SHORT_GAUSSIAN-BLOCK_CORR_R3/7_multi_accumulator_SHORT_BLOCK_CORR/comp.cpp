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
// We expect a fixed size Min-Heap of K elements
static inline void heap_insert(HeapElement *root, float val, uint32_t idx) {
  if (val <= root[0].val) {
    return; // No need to insert if the new value is smaller than the current
            // min
  }

  root[0].val = val;
  root[0].idx = idx;
  // Heapify down from the root to restore Min-Heap property
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
      break; // Min-Heap property is satisfied
    }

    // Swap with the smaller child
    std::swap(root[i], root[smallest]);
    i = smallest; // Move down to the child index
  }
}
// Materialize the Min-Heap into topK output array
static inline void heap_to_topK(HeapElement *root, uint32_t *topK_row) {
  // top K needs to be sorted in descending order
  for (uint32_t k = 0; k < K; ++k) {
    uint32_t j = K - k - 1;

    topK_row[j] =
        root[0].idx;   // The root of the Min-Heap is the smallest element
    root[0] = root[j]; // Move the last element to the root
    // Heapify
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
        break; // Min-Heap property is satisfied
      }

      // Swap with the smaller child
      std::swap(root[i], root[smallest]);
      i = smallest; // Move down to the child index
    }
  }
}
// Initialize the Min-Heap with K sentinel values
static inline void heap_init(HeapElement *root) {
  for (uint32_t k = 0; k < K; ++k) {
    root[k].val = 0.0f;
    root[k].idx = SENTINEL;
  }
}

// Per-variant scratch. For the unoptimized baseline we keep exactly the
// three buffers the original in-function allocations produced:
//   mean_vals        — per-feature mean of X over its valid interval
//   std_sample_vals  — per-feature sample std over its valid interval
//   corr_matrix      — dense F x F correlation matrix
//   normalized_X     — normalized X values for correlation computation
struct Scratch {
  // float *mean_vals = nullptr;
  // float *std_sample_vals = nullptr;
  // float *corr_matrix = nullptr;
  float *normalized_X = nullptr;
  HeapElement *topK_heaps = nullptr;
};

Scratch *scratch_create(const RawInData *data) {
  Scratch *s = new (std::nothrow) Scratch();
  if (!s)
    return nullptr;

  size_t f_size = static_cast<size_t>(data->F);
  size_t s_size = static_cast<size_t>(data->S);

  // s->mean_vals = aligned_alloc_padded<float>(f_size);
  // s->std_sample_vals = aligned_alloc_padded<float>(f_size);
  // s->corr_matrix = aligned_alloc_padded<float>(f_size * f_size);
  s->normalized_X = aligned_alloc_padded<float>(f_size * s_size);
  s->topK_heaps = (HeapElement *)malloc(sizeof(HeapElement) * f_size * K);
  // if (!s->mean_vals || !s->std_sample_vals || !s->corr_matrix ||
  // !s->normalized_X) {
  if (!s->normalized_X || !s->topK_heaps) {
    scratch_destroy(s);
    return nullptr;
  }
  // memset(s->corr_matrix, 0, f_size * f_size * sizeof(float));
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
  // free(scratch->mean_vals);
  // free(scratch->std_sample_vals);
  // free(scratch->corr_matrix);
  free(scratch->normalized_X);
  free(scratch->topK_heaps);
  delete scratch;
}

static inline float hsum256_simple(__m256 v) {
  alignas(32) float temp[8];
  _mm256_store_ps(temp, v);
  return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] +
         temp[7];
}

// REQUIRES: F is a multiple of 4
// REQUIRES: S is a multiple of 8
// REQUIRES: starts[i] and ends[i] are multiples of 8 for all i
void computation(const RawInData *data, Scratch *scratch, uint32_t *topK) {

  PROFILE("total");
  assert(data->F % 4 == 0 && "F must be a multiple of 4 for block processing");
  assert(data->S % 8 == 0 && "S must be a multiple of 8 for AVX processing");
  // for (uint32_t i = 0; i < data->F; ++i) {
  //   uint32_t start = data->starts[i];
  //   uint32_t end = data->ends[i];
  //   assert(start % 8 == 0 && end % 8 == 0 &&
  //          "We expect starts/ends to be 8-float aligned for efficient AVX "
  //          "processing");
  // }

  // float *mean_vals = scratch->mean_vals;
  // float *std_sample_vals = scratch->std_sample_vals;
  // float *corr_matrix = scratch->corr_matrix;
  float *normalized_X = scratch->normalized_X;
  // Variable rewriting
  uint32_t F = data->F;
  uint32_t S = data->S;
  float *X = data->X;
  uint32_t *starts = data->starts;
  uint32_t *ends = data->ends;

  HeapElement *topK_heaps = scratch->topK_heaps;

  // Phase 1: per-feature mean/std + normalization (fused).
  {
    PROFILE("phase1");

    for (uint32_t i = 0; i < F; ++i) {
      uint32_t start = starts[i];
      uint32_t end = ends[i];
      uint32_t aligned32_start = start & ~31u;
      uint32_t aligned32_end = (end + 31u) & ~31u;
      uint32_t aligned8_start = start & ~7u;
      uint32_t aligned8_end = (end + 7u) & ~7u;
      uint32_t interval_size = end - start;

      // Math: variance_sum = sum((x - mean)^2) = sum(x^2) - 2*mean*sum(x) +
      // n*mean^2
      float sum = 0.0f;
      float sum_sq = 0.0f;
      if (S >= 32) {
        // Manual vectorization: 0.0 padding elements won't contribute to the
        // sum Multiple accumulators: remove dependency chain
        __m256 v_sum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                           _mm256_setzero_ps(), _mm256_setzero_ps()};
        __m256 v_sum_sq[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                              _mm256_setzero_ps(), _mm256_setzero_ps()};
        const float *base_ptr = &X[i * S + aligned32_start];
        for (uint32_t j = aligned32_start; j < aligned32_end;
             j += 32, base_ptr += 32) {
          __m256 v_x[4] = {
              _mm256_load_ps(base_ptr), _mm256_load_ps(base_ptr + 8),
              _mm256_load_ps(base_ptr + 16), _mm256_load_ps(base_ptr + 24)};
          // v_sum += v_x
          v_sum[0] = _mm256_add_ps(v_sum[0], v_x[0]);
          v_sum[1] = _mm256_add_ps(v_sum[1], v_x[1]);
          v_sum[2] = _mm256_add_ps(v_sum[2], v_x[2]);
          v_sum[3] = _mm256_add_ps(v_sum[3], v_x[3]);
          // v_sum_sq += v_x * v_x
          v_sum_sq[0] = _mm256_fmadd_ps(v_x[0], v_x[0], v_sum_sq[0]);
          v_sum_sq[1] = _mm256_fmadd_ps(v_x[1], v_x[1], v_sum_sq[1]);
          v_sum_sq[2] = _mm256_fmadd_ps(v_x[2], v_x[2], v_sum_sq[2]);
          v_sum_sq[3] = _mm256_fmadd_ps(v_x[3], v_x[3], v_sum_sq[3]);
        }
        v_sum[0] = _mm256_add_ps(v_sum[0], v_sum[1]);
        v_sum[2] = _mm256_add_ps(v_sum[2], v_sum[3]);
        v_sum_sq[0] = _mm256_add_ps(v_sum_sq[0], v_sum_sq[1]);
        v_sum_sq[2] = _mm256_add_ps(v_sum_sq[2], v_sum_sq[3]);
        v_sum[0] = _mm256_add_ps(v_sum[0], v_sum[2]);
        v_sum_sq[0] = _mm256_add_ps(v_sum_sq[0], v_sum_sq[2]);
        sum = hsum256_simple(v_sum[0]);
        sum_sq = hsum256_simple(v_sum_sq[0]);
      } else {
        // Fallback to scalar code for small S
        const float *base_ptr = &X[i * S];
        sum = 0.0f;
        sum_sq = 0.0f;
        for (uint32_t j = start; j < end; ++j) {
          float v = base_ptr[j];
          sum += v;
          sum_sq += v * v;
        }
      }

      float mean = sum / interval_size;
      // variance_sum = sum((x - mean)^2) = sum(x^2) - 2*mean*sum(x) + n*mean^2
      float variance_sum = ((interval_size * mean) * mean) +
                           ((-2.0f * sum) * mean + sum_sq); // FMA style
      float std_sample =
          std::sqrt(variance_sum / (interval_size - 1) + STD_EPS);
      float inv_std_sample = 1.0f / std_sample;

      // Precompute: normalized_X = (X - mean) / std_sample for correlation
      // computation
      if (S >= 32) {
        // Manual vectorization
        __m256 v_mean = _mm256_set1_ps(mean);
        __m256 v_inv_std_sample = _mm256_set1_ps(inv_std_sample);
        const float *base_ptr = &X[i * S + aligned8_start];
        float *normalized_ptr = &normalized_X[i * S + aligned8_start];
        for (uint32_t j = aligned8_start; j < aligned8_end;
             j += 8, base_ptr += 8, normalized_ptr += 8) {
          __m256 v_x = _mm256_load_ps(base_ptr);
          // v_normalized = (v_x - v_mean) * v_inv_std_sample
          __m256 v_normalized =
              _mm256_mul_ps(_mm256_sub_ps(v_x, v_mean), v_inv_std_sample);
          _mm256_store_ps(normalized_ptr, v_normalized);
        }
      } else {
        // Fallback to scalar code for small S
        for (uint32_t j = start; j < end; ++j) {
          normalized_X[i * S + j] = (X[i * S + j] - mean) * inv_std_sample;
        }
      }

      // mean_vals[i] = mean;
      // std_sample_vals[i] = std_sample;
    }
  } // end phase 1

  // Phase 2: pairwise correlation + top-K heap insertion (fused).
  {
    PROFILE("phase2");
    // Math: symmetric matrix, only calculate upper triangle
    // Manual vectorization: 0.0 padding elements won't contribute to the sum
    for (uint32_t i = 0; i < F; i += 4) {
      // Block processing: check 4 features at a time to have better locality

      {
        PROFILE("phase2_diag");

        // Edge case (the diagonal staircase)
        // i   <-> i+1, i+2, i+3
        // i+1 <->      i+2, i+3
        // i+2 <->           i+3
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
        uint32_t n01 = r01 - l01;
        uint32_t n02 = r02 - l02;
        uint32_t n03 = r03 - l03;
        uint32_t n12 = r12 - l12;
        uint32_t n13 = r13 - l13;
        uint32_t n23 = r23 - l23;

        uint32_t min_l = std::min({l01, l02, l03, l12, l13, l23});
        uint32_t max_r = std::max({r01, r02, r03, r12, r13, r23});

        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc02 = _mm256_setzero_ps();
        __m256 acc03 = _mm256_setzero_ps();
        __m256 acc12 = _mm256_setzero_ps();
        __m256 acc13 = _mm256_setzero_ps();
        __m256 acc23 = _mm256_setzero_ps();
        // min_l and max_r are multiples of 8
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

        // corr_matrix[i *       F + (i + 1)] = n01 == 0 ? 0.0f :
        // hsum256_simple(acc01) / (n01 - 1); corr_matrix[i *       F + (i + 2)]
        // = n02 == 0 ? 0.0f : hsum256_simple(acc02) / (n02 - 1); corr_matrix[i
        // * F
        // + (i + 3)] = n03 == 0 ? 0.0f : hsum256_simple(acc03) / (n03 - 1);
        // corr_matrix[(i + 1) * F + (i + 2)] = n12 == 0 ? 0.0f :
        // hsum256_simple(acc12) / (n12 - 1); corr_matrix[(i + 1) * F + (i + 3)]
        // = n13 == 0 ? 0.0f : hsum256_simple(acc13) / (n13 - 1); corr_matrix[(i
        // + 2) * F + (i + 3)] = n23 == 0 ? 0.0f : hsum256_simple(acc23) / (n23
        // - 1);

        // corr_matrix[(i + 1) * F + i      ] = corr_matrix[i * F + (i + 1)];
        // corr_matrix[(i + 2) * F + i      ] = corr_matrix[i * F + (i + 2)];
        // corr_matrix[(i + 3) * F + i      ] = corr_matrix[i * F + (i + 3)];
        // corr_matrix[(i + 2) * F + (i + 1)] = corr_matrix[(i + 1) * F + (i +
        // 2)]; corr_matrix[(i + 3) * F + (i + 1)] = corr_matrix[(i + 1) * F +
        // (i
        // + 3)]; corr_matrix[(i + 3) * F + (i + 2)] = corr_matrix[(i + 2) * F +
        // (i + 3)];

        if (n01 > 0) {
          float tmp = std::fabs(hsum256_simple(acc01) / (n01 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 1);
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i);
        }
        if (n02 > 0) {
          float tmp = std::fabs(hsum256_simple(acc02) / (n02 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 2);
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i);
        }
        if (n03 > 0) {
          float tmp = std::fabs(hsum256_simple(acc03) / (n03 - 1));
          heap_insert(&topK_heaps[i * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i);
        }
        if (n12 > 0) {
          float tmp = std::fabs(hsum256_simple(acc12) / (n12 - 1));
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i + 2);
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i + 1);
        }
        if (n13 > 0) {
          float tmp = std::fabs(hsum256_simple(acc13) / (n13 - 1));
          heap_insert(&topK_heaps[(i + 1) * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i + 1);
        }
        if (n23 > 0) {
          float tmp = std::fabs(hsum256_simple(acc23) / (n23 - 1));
          heap_insert(&topK_heaps[(i + 2) * K], tmp, i + 3);
          heap_insert(&topK_heaps[(i + 3) * K], tmp, i + 2);
        }

      } // end phase2_diag

      {
        PROFILE("phase2_jloop");

        for (uint32_t j = i + 4; j + 1 < F; j += 2) {
          // i, i+1, i+2, i+3 <-> (j, j+1). 8 accumulators (4 i-features × 2
          // j-features) to keep 8 independent FMA dependency chains in flight —
          // saturates the 2-per-cycle FMA throughput on Skylake/Kaby Lake
          // despite 4-cycle latency.
          uint32_t l0a = std::max(starts[i], starts[j]);
          uint32_t l1a = std::max(starts[i + 1], starts[j]);
          uint32_t l2a = std::max(starts[i + 2], starts[j]);
          uint32_t l3a = std::max(starts[i + 3], starts[j]);
          uint32_t l0b = std::max(starts[i], starts[j + 1]);
          uint32_t l1b = std::max(starts[i + 1], starts[j + 1]);
          uint32_t l2b = std::max(starts[i + 2], starts[j + 1]);
          uint32_t l3b = std::max(starts[i + 3], starts[j + 1]);
          uint32_t r0a = std::min(ends[i], ends[j]);
          uint32_t r1a = std::min(ends[i + 1], ends[j]);
          uint32_t r2a = std::min(ends[i + 2], ends[j]);
          uint32_t r3a = std::min(ends[i + 3], ends[j]);
          uint32_t r0b = std::min(ends[i], ends[j + 1]);
          uint32_t r1b = std::min(ends[i + 1], ends[j + 1]);
          uint32_t r2b = std::min(ends[i + 2], ends[j + 1]);
          uint32_t r3b = std::min(ends[i + 3], ends[j + 1]);
          uint32_t n0a = r0a - l0a, n1a = r1a - l1a, n2a = r2a - l2a,
                   n3a = r3a - l3a;
          uint32_t n0b = r0b - l0b, n1b = r1b - l1b, n2b = r2b - l2b,
                   n3b = r3b - l3b;

          uint32_t min_l = std::min({l0a, l1a, l2a, l3a, l0b, l1b, l2b, l3b});
          uint32_t max_r = std::max({r0a, r1a, r2a, r3a, r0b, r1b, r2b, r3b});

          __m256 acc0a = _mm256_setzero_ps(), acc0b = _mm256_setzero_ps();
          __m256 acc1a = _mm256_setzero_ps(), acc1b = _mm256_setzero_ps();
          __m256 acc2a = _mm256_setzero_ps(), acc2b = _mm256_setzero_ps();
          __m256 acc3a = _mm256_setzero_ps(), acc3b = _mm256_setzero_ps();
          // min_l and max_r are multiples of 8
          const float *base_ptr_i = &normalized_X[i * S + min_l];
          const float *base_ptr_ja = &normalized_X[j * S + min_l];
          const float *base_ptr_jb = &normalized_X[(j + 1) * S + min_l];
          for (uint32_t t = min_l; t < max_r;
               t += 8, base_ptr_i += 8, base_ptr_ja += 8, base_ptr_jb += 8) {
            __m256 v_i0 = _mm256_load_ps(base_ptr_i);
            __m256 v_i1 = _mm256_load_ps(base_ptr_i + S);
            __m256 v_i2 = _mm256_load_ps(base_ptr_i + 2 * S);
            __m256 v_i3 = _mm256_load_ps(base_ptr_i + 3 * S);
            __m256 v_ja = _mm256_load_ps(base_ptr_ja);
            __m256 v_jb = _mm256_load_ps(base_ptr_jb);
            acc0a = _mm256_fmadd_ps(v_i0, v_ja, acc0a);
            acc0b = _mm256_fmadd_ps(v_i0, v_jb, acc0b);
            acc1a = _mm256_fmadd_ps(v_i1, v_ja, acc1a);
            acc1b = _mm256_fmadd_ps(v_i1, v_jb, acc1b);
            acc2a = _mm256_fmadd_ps(v_i2, v_ja, acc2a);
            acc2b = _mm256_fmadd_ps(v_i2, v_jb, acc2b);
            acc3a = _mm256_fmadd_ps(v_i3, v_ja, acc3a);
            acc3b = _mm256_fmadd_ps(v_i3, v_jb, acc3b);
          }

          float s0a = hsum256_simple(acc0a), s0b = hsum256_simple(acc0b);
          float s1a = hsum256_simple(acc1a), s1b = hsum256_simple(acc1b);
          float s2a = hsum256_simple(acc2a), s2b = hsum256_simple(acc2b);
          float s3a = hsum256_simple(acc3a), s3b = hsum256_simple(acc3b);

          if (n0a > 0) {
            float v = std::fabs(s0a / (n0a - 1));
            heap_insert(&topK_heaps[i * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i);
          }
          if (n1a > 0) {
            float v = std::fabs(s1a / (n1a - 1));
            heap_insert(&topK_heaps[(i + 1) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 1);
          }
          if (n2a > 0) {
            float v = std::fabs(s2a / (n2a - 1));
            heap_insert(&topK_heaps[(i + 2) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 2);
          }
          if (n3a > 0) {
            float v = std::fabs(s3a / (n3a - 1));
            heap_insert(&topK_heaps[(i + 3) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 3);
          }
          if (n0b > 0) {
            float v = std::fabs(s0b / (n0b - 1));
            heap_insert(&topK_heaps[i * K], v, j + 1);
            heap_insert(&topK_heaps[(j + 1) * K], v, i);
          }
          if (n1b > 0) {
            float v = std::fabs(s1b / (n1b - 1));
            heap_insert(&topK_heaps[(i + 1) * K], v, j + 1);
            heap_insert(&topK_heaps[(j + 1) * K], v, i + 1);
          }
          if (n2b > 0) {
            float v = std::fabs(s2b / (n2b - 1));
            heap_insert(&topK_heaps[(i + 2) * K], v, j + 1);
            heap_insert(&topK_heaps[(j + 1) * K], v, i + 2);
          }
          if (n3b > 0) {
            float v = std::fabs(s3b / (n3b - 1));
            heap_insert(&topK_heaps[(i + 3) * K], v, j + 1);
            heap_insert(&topK_heaps[(j + 1) * K], v, i + 3);
          }
        }

        // Tail: if an odd number of j-features remain, handle the last one
        // with the original 4-accumulator path.
        if ((F - (i + 4)) % 2 == 1) {
          uint32_t j = F - 1;
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
            __m256 v_i0 = _mm256_load_ps(base_ptr_i);
            __m256 v_i1 = _mm256_load_ps(base_ptr_i + S);
            __m256 v_i2 = _mm256_load_ps(base_ptr_i + 2 * S);
            __m256 v_i3 = _mm256_load_ps(base_ptr_i + 3 * S);
            __m256 v_j = _mm256_load_ps(base_ptr_j);
            acc0 = _mm256_fmadd_ps(v_i0, v_j, acc0);
            acc1 = _mm256_fmadd_ps(v_i1, v_j, acc1);
            acc2 = _mm256_fmadd_ps(v_i2, v_j, acc2);
            acc3 = _mm256_fmadd_ps(v_i3, v_j, acc3);
          }

          float s0 = hsum256_simple(acc0), s1 = hsum256_simple(acc1);
          float s2 = hsum256_simple(acc2), s3 = hsum256_simple(acc3);

          if (n0 > 0) {
            float v = std::fabs(s0 / (n0 - 1));
            heap_insert(&topK_heaps[i * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i);
          }
          if (n1 > 0) {
            float v = std::fabs(s1 / (n1 - 1));
            heap_insert(&topK_heaps[(i + 1) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 1);
          }
          if (n2 > 0) {
            float v = std::fabs(s2 / (n2 - 1));
            heap_insert(&topK_heaps[(i + 2) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 2);
          }
          if (n3 > 0) {
            float v = std::fabs(s3 / (n3 - 1));
            heap_insert(&topK_heaps[(i + 3) * K], v, j);
            heap_insert(&topK_heaps[j * K], v, i + 3);
          }
        }

      } // end phase2_jloop
    }
  } // end phase 2

#ifdef DEBUG
  printf("Correlation matrix:\n");
  for (uint32_t i = 0; i < data->F; ++i) {
    for (uint32_t j = 0; j < data->F; ++j) {
      printf("%.4f ", corr_matrix[i * data->F + j]);
    }
    printf("\n");
  }
#endif

  // for (uint32_t i = 0; i < F; ++i) {
  //   // naive solution: K-passes to find top K indices

  //   for (uint32_t k = 0; k < K; ++k) {
  //     // Find k-th largest correlation index
  //     uint32_t kth_idx = SENTINEL;
  //     float kth_val = 0.0f;
  //     for (uint32_t j = 0; j < data->F; ++j) {
  //       // Skip already selected indices
  //       bool already_selected = false;
  //       for (uint32_t m = 0; m < k; ++m) {
  //         if (topK[i * K + m] == j) {
  //           already_selected = true;
  //           break;
  //         }
  //       }
  //       if (already_selected)
  //         continue;

  //       float v = std::fabs(corr_matrix[i * F + j]);
  //       if (v > kth_val) {
  //         kth_val = v;
  //         kth_idx = j;
  //       }
  //     }
  //     topK[i * K + k] = kth_idx;
  //   }
  // }

  // Phase 3: materialize top-K from heaps.
  {
    PROFILE("phase3");
    for (uint32_t i = 0; i < F; ++i) {
      heap_to_topK(&topK_heaps[i * K], &topK[i * K]);
    }
  }

#ifdef DEBUG
  printf("Top %u indices:\n", K);
  for (uint32_t i = 0; i < data->F; ++i) {
    printf("Feature %u: ", i);
    for (uint32_t k = 0; k < K; ++k) {
      printf("%u ", topK[i * K + k]);
      printf("(%.4f) ", corr_matrix[i * data->F + topK[i * K + k]]);
    }
    printf("\n");
  }
#endif
}
