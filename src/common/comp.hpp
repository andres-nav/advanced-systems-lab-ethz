#pragma once
// Contract that every variant in src/variants/<name>/comp.cpp must implement.

#include <cstdint>

#include "common/parser.hpp"

// Opaque per-variant scratch. Defined in each variant's comp.cpp.
struct Scratch;

// Allocate + initialize scratch sized for `data`. Returns nullptr on
// allocation failure. Must be paired with scratch_destroy.
Scratch *scratch_create(const RawInData *data);

// Reset the scatch for multiple runs
void scratch_reset(const RawInData *data, Scratch *scratch);

// Release a Scratch previously returned by scratch_create. No-op on
// nullptr.
void scratch_destroy(Scratch *scratch);

// Compute top-K for `data` using `scratch` as preallocated working
// memory; writes results into `topK`. Does not allocate or free.
void computation(const RawInData *data, Scratch *scratch, uint32_t *topK);
