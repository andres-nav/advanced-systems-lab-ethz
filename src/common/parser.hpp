#pragma once

#include <cstdint>

struct RawInData {
  uint32_t F;
  uint32_t S;
  uint32_t *starts;
  uint32_t *ends;
  float *X;
};

void free_raw_in_data(RawInData *data);
RawInData parse_raw_in_data(const char *path);
void visualize_raw_in_data(const RawInData *data);
