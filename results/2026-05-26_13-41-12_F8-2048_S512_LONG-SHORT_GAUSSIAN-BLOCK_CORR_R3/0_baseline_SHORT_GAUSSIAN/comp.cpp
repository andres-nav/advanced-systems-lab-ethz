#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/comp.hpp"
#include "common/parameters.hpp"
#include "common/parser.hpp"

// Per-variant scratch buffers:
//   mean_vals        — per-feature mean of X over its valid interval
//   std_sample_vals  — per-feature sample std over its valid interval
//   corr_matrix      — dense F x F correlation matrix
struct Scratch {
  float *mean_vals = nullptr;
  float *std_sample_vals = nullptr;
  float *corr_matrix = nullptr;
};

Scratch *scratch_create(const RawInData *data) {
  Scratch *s = new (std::nothrow) Scratch();
  if (!s)
    return nullptr;

  size_t f_size = static_cast<size_t>(data->F);

  s->mean_vals = aligned_alloc_padded<float>(f_size);
  s->std_sample_vals = aligned_alloc_padded<float>(f_size);
  s->corr_matrix = aligned_alloc_padded<float>(f_size * f_size);

  if (!s->mean_vals || !s->std_sample_vals || !s->corr_matrix) {
    scratch_destroy(s);
    return nullptr;
  }
  return s;
}

void scratch_reset(const RawInData *data, Scratch *scratch) {
  size_t f_size = static_cast<size_t>(data->F);

  memset(scratch->mean_vals, 0, f_size * sizeof(float));
  memset(scratch->std_sample_vals, 0, f_size * sizeof(float));
  memset(scratch->corr_matrix, 0, f_size * f_size * sizeof(float));

  clflush_range(scratch->mean_vals, f_size * sizeof(float));
  clflush_range(scratch->std_sample_vals, f_size * sizeof(float));
  clflush_range(scratch->corr_matrix, f_size * f_size * sizeof(float));
  clflush_range(scratch, sizeof(*scratch));
}

void scratch_destroy(Scratch *scratch) {
  if (!scratch)
    return;
  free(scratch->mean_vals);
  free(scratch->std_sample_vals);
  free(scratch->corr_matrix);
  delete scratch;
}

void computation(const RawInData *data, Scratch *scratch, uint32_t *topK) {
  float *mean_vals = scratch->mean_vals;
  float *std_sample_vals = scratch->std_sample_vals;
  float *corr_matrix = scratch->corr_matrix;

  for (uint32_t i = 0; i < data->F; ++i) {
    uint32_t start = data->starts[i];
    uint32_t end = data->ends[i];

    float sum = 0.0f;
    for (uint32_t j = start; j < end; ++j) {
      sum += data->X[i * data->S + j];
    }
    mean_vals[i] = sum / (end - start);

    float variance_sum = 0.0f;
    for (uint32_t j = start; j < end; ++j) {
      float diff = data->X[i * data->S + j] - mean_vals[i];
      variance_sum += diff * diff;
    }
    std_sample_vals[i] = std::sqrt(variance_sum / (end - start - 1) + STD_EPS);
  }

  for (uint32_t i = 0; i < data->F; ++i) {
    for (uint32_t j = 0; j < data->F; ++j) {
      if (j == i) {
        corr_matrix[i * data->F + j] = 0.0f;
        continue;
      }

      uint32_t l = std::max(data->starts[i], data->starts[j]);
      uint32_t r = std::min(data->ends[i], data->ends[j]);
      uint32_t n = r - l;
      if (n < 2) {
        corr_matrix[i * data->F + j] = 0.0f;
        continue;
      }

      float acc = 0.0f;
      for (uint32_t t = l; t < r; ++t) {
        float zi =
            (data->X[i * data->S + t] - mean_vals[i]) / std_sample_vals[i];
        float zj =
            (data->X[j * data->S + t] - mean_vals[j]) / std_sample_vals[j];
        acc += zi * zj;
      }
      corr_matrix[i * data->F + j] = acc / (n - 1);
    }
  }

#ifdef DEBUG
  printf("Correlation matrix:\n");
  for (uint32_t i = 0; i < data->F; ++i) {
    for (uint32_t j = 0; j < data->F; ++j) {
      printf("%.4f ", corr_matrix[i * data->F + j]);
    }
    printf("\n");
  }
#endif

  for (uint32_t i = 0; i < data->F; ++i) {
    // naive solution: K-passes to find top K indices

    for (uint32_t k = 0; k < K; ++k) {
      // Find k-th largest correlation index
      uint32_t kth_idx = SENTINEL;
      float kth_val = 0.0f;
      for (uint32_t j = 0; j < data->F; ++j) {
        // Skip already selected indices
        bool already_selected = false;
        for (uint32_t m = 0; m < k; ++m) {
          if (topK[i * K + m] == j) {
            already_selected = true;
            break;
          }
        }
        if (already_selected)
          continue;

        float v = std::fabs(corr_matrix[i * data->F + j]);
        if (v > kth_val) {
          kth_val = v;
          kth_idx = j;
        }
      }
      topK[i * K + k] = kth_idx;
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
