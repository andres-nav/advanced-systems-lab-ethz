#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>

// The top "K" indices to be returned for each feature.
constexpr uint32_t K = 16;

// Memory alignment in bytes. Must be a power of two.
constexpr size_t MEM_ALIGNMENT = 32;

// Regularizer added under the sqrt in the sample std to avoid division
// by zero on constant-valued intervals.
// See <docs/project-description.pdf>
constexpr float STD_EPS = 1e-12f;

// Padding/sentinel for top-K results. Emitted when a feature has fewer
// than K neighbours with strictly non-zero |correlation|. Kept in sync
// with SENTINEL in tools/reference.py.
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

// Round `n` up to the next multiple of `a`. `a` must be a power of two.
constexpr size_t align_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

// aligned_alloc wrapper: allocate space for `n_elems` of T, aligned to
// MEM_ALIGNMENT, with the allocation size rounded up to the alignment
// (required by aligned_alloc). Returns nullptr on failure.
template <typename T> inline T *aligned_alloc_padded(size_t n_elems) {
  return static_cast<T *>(aligned_alloc(
      MEM_ALIGNMENT, align_up(n_elems * sizeof(T), MEM_ALIGNMENT)));
}

// Evict `bytes` starting at `addr` from every data cache level using
// CLFLUSH, then issue _mm_mfence so all flushes have retired by the time
// this call returns.
inline void clflush_range(const void *addr, size_t bytes) {
  constexpr size_t LINE = 64;
  const uint8_t *p = static_cast<const uint8_t *>(addr);
  uintptr_t first = reinterpret_cast<uintptr_t>(p) & ~(LINE - 1);
  uintptr_t last = reinterpret_cast<uintptr_t>(p + bytes);
  for (uintptr_t a = first; a < last; a += LINE) {
    _mm_clflush(reinterpret_cast<const void *>(a));
  }
  _mm_mfence();
}
