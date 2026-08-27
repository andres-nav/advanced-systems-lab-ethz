#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common/parameters.hpp"
#include "common/parser.hpp"

void free_raw_in_data(RawInData *data) {
  free(data->starts);
  free(data->ends);
  free(data->X);
  data->starts = nullptr;
  data->ends = nullptr;
  data->X = nullptr;
}

RawInData parse_raw_in_data(const char *path) {
  RawInData data = {0, 0, nullptr, nullptr, nullptr};

  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "Error: Could not open file %s\n", path);
    return data;
  }

  if (fread(&data.F, sizeof(uint32_t), 1, file) != 1 ||
      fread(&data.S, sizeof(uint32_t), 1, file) != 1) {
    fprintf(stderr, "Error: Failed to read header.\n");
    fclose(file);
    return data;
  }

  data.starts = aligned_alloc_padded<uint32_t>(data.F);
  data.ends = aligned_alloc_padded<uint32_t>(data.F);
  const size_t n_x = static_cast<size_t>(data.F) * data.S;
  data.X = aligned_alloc_padded<float>(n_x);

  if (!data.starts || !data.ends || !data.X) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    free_raw_in_data(&data);
    fclose(file);
    return data;
  }

  // Read data into arrays. A short read here means the file was
  // truncated or the header lied about (F, S); fail loudly rather
  // than computing on uninitialized memory.
  if (fread(data.starts, sizeof(uint32_t), data.F, file) != data.F ||
      fread(data.ends, sizeof(uint32_t), data.F, file) != data.F ||
      fread(data.X, sizeof(float), n_x, file) != n_x) {
    fprintf(stderr, "Error: Truncated input file %s.\n", path);
    free_raw_in_data(&data);
    fclose(file);
    return data;
  }

  fclose(file);
  return data;
}

void visualize_raw_in_data(const RawInData *data) {
  if (!data || !data->X || !data->starts || !data->ends) {
    fprintf(stderr, "Error: Invalid data for visualization.\n");
    return;
  }

  printf("=== Header ===\n");
  printf("F (Rows):          %u\n", data->F);
  printf("S (Columns):       %u\n\n", data->S);

  printf("=== Starts Array ===\n");
  for (uint32_t i = 0; i < data->F; ++i) {
    printf("%u ", data->starts[i]);
  }
  printf("\n\n");

  printf("=== Ends Array ===\n");
  for (uint32_t i = 0; i < data->F; ++i) {
    printf("%u ", data->ends[i]);
  }
  printf("\n\n");

  printf("=== X Matrix ===\n");
  for (uint32_t i = 0; i < data->F; ++i) {
    for (uint32_t j = 0; j < data->S; ++j) {
      size_t index = static_cast<size_t>(i) * data->S + j;
      printf("%.2f ", data->X[index]);
    }
    printf("\n");
  }
  printf("\n");
}
