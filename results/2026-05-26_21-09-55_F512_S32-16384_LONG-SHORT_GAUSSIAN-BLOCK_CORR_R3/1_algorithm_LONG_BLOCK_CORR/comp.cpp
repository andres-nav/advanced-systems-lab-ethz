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

// Per-variant scratch buffers:
//   mean_vals        — per-feature mean of X over its valid interval
//   std_sample_vals  — per-feature sample std over its valid interval
//   corr_matrix      — dense F x F correlation matrix
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

  // Evict scratch from every cache level so the next computation starts
  // cold on these buffers. Pairs with main.cpp's flush of data->X and
  // its _mm_mfence() right before start_tsc.
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
  // Variable rewriting
  uint32_t F = data->F;
  uint32_t S = data->S;
  float *X = data->X;
  uint32_t *starts = data->starts;
  uint32_t *ends = data->ends;

  float *normalized_X = scratch->normalized_X;

  HeapElement *topK_heaps = scratch->topK_heaps;

  {
    PROFILE("phase1");
    for (uint32_t i = 0; i < F; i++) {
      uint32_t start = starts[i];
      uint32_t end = ends[i];
      uint32_t n = end - start;

      float sum = 0.0f;
      float sum_sq = 0.0f;

      for (uint32_t j = start; j < end; j++) {
        float x = X[i * S + j];
        sum += x;
        sum_sq += x * x;
      }

      float mean = sum / n;
      float var =
          (((n * mean) * mean) - ((2.0f * sum) * mean) + sum_sq) / (n - 1);
      float std = sqrt(var + STD_EPS);

      for (uint32_t j = start; j < end; j++) {
        normalized_X[i * S + j] = (X[i * S + j] - mean) / std;
      }
    }
  }

  // Phase 2: pairwise correlation + top-K heap insertion (fused).
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

        float acc = 0.0f;

        for (uint32_t k = start; k < end; k++) {
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
