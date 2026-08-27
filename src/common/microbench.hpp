#pragma once

#ifdef MICROBENCH

#include <cstdint>
#include <iomanip>
#include <iostream>

#include "common/tsc_x86.h"

static constexpr int MB_MAX = 16;

struct MbGlobals {
  int n = 0;
  char const *labels[MB_MAX]{};
  uint64_t cycles[MB_MAX]{};
  uint64_t calls = 0;

  __attribute__((noinline)) int id(char const *l) {
    for (int i = 0; i < n; ++i) {
      if (labels[i] == l) return i;
    }
    if (n >= MB_MAX) return 0;    
    int i = n++;
    labels[i] = l; 
    cycles[i] = 0;
    return i;
  }

  inline void add(int i, uint64_t c) { cycles[i] += c; }
  inline void bump() { calls++; }

  ~MbGlobals() {
    if (calls == 0) { return; }
    for (int i = 0; i < n; ++i) {
      double const avg = static_cast<double>(cycles[i]) / calls;
      std::cerr << "[microbench] " << labels[i] << ": " 
      << std::fixed << std::setprecision(1) << avg << " cycles/call "
      << "(" << cycles[i] << " total, " << calls << " calls)\n";
    }
  }
};

// 'inline' prevents ODR violations (multiple globals) if included in multiple .cpp files
inline MbGlobals &g() { 
    static MbGlobals x; 
    return x; 
}

struct Scoped {
  int id_; 
  uint64_t start_;
  
  __attribute__((always_inline)) explicit Scoped(int id) 
      : id_(id), start_(start_tsc()) {}
      
  __attribute__((always_inline)) ~Scoped() { 
      g().add(id_, stop_tsc(start_)); 
  }
};

#define PROFILE(label) \
    static int _mb_ln_##__LINE__ = -1; \
    if (__builtin_expect(_mb_ln_##__LINE__ < 0, 0)) { \
        _mb_ln_##__LINE__ = g().id(label); \
    } \
    Scoped _mb_s_##__LINE__(_mb_ln_##__LINE__); \
    (void)_mb_s_##__LINE__

#define PROFILE_BUMP_CALL() g().bump()

#else

#define PROFILE(label) ((void)0)
#define PROFILE_BUMP_CALL() ((void)0)

#endif
