#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#define STDIN_BUFFER_BYTES (1024 * 1024)
#define MALLOC_PADDING 128

#include "common.cuh"

/* returns size of mallocd memory */
uint32_t fl_stdin_mapped_malloc(uint8_t** host_mapped_ptr) {
  cudaMallocHost(host_mapped_ptr, STDIN_BUFFER_BYTES);
  uint32_t bytes_read = fread(*host_mapped_ptr, 1, STDIN_BUFFER_BYTES, stdin);
  ASSERT(bytes_read > 0);
  return bytes_read;
}


uint32_t fl_file_mapped_malloc(const char* filename,
                             uint8_t** host_mapped_ptr) {
  FILE* file = fopen(filename, "rb");
  ASSERT(file != NULL);

  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  cudaMallocHost(host_mapped_ptr, file_size);
  size_t bytes_read = fread(*host_mapped_ptr, 1, file_size, file);
  fclose(file);

  ASSERT(bytes_read == file_size);
  return (uint32_t)file_size;
}


/* returns size of mallocd memory */
uint32_t fl_stdin_dev_malloc(uint8_t** host_mapped_ptr, uint8_t** dev_ptr) {
  uint32_t mappedmem_size = fl_stdin_mapped_malloc(host_mapped_ptr);
  uint32_t aligned_size = ROUND_UP(mappedmem_size, MALLOC_PADDING);
  cudaMalloc(dev_ptr, aligned_size);
  cudaMemcpy(*dev_ptr, *host_mapped_ptr, mappedmem_size, cudaMemcpyHostToDevice);
  if (aligned_size > mappedmem_size) {
    cudaMemset((uint8_t*)*dev_ptr + mappedmem_size, 0, aligned_size - mappedmem_size);
  }
  return (uint32_t)aligned_size;
}


uint32_t fl_file_dev_malloc(const char* filename, uint8_t** host_mapped_ptr, uint8_t** dev_ptr) {
  uint32_t mappedmem_size = fl_file_mapped_malloc(filename, host_mapped_ptr);
  uint32_t aligned_size = ROUND_UP(mappedmem_size, MALLOC_PADDING);
  cudaMalloc(dev_ptr, aligned_size);
  cudaMemcpy(*dev_ptr, *host_mapped_ptr, mappedmem_size, cudaMemcpyHostToDevice);
  if (aligned_size > mappedmem_size) {
    cudaMemset((uint8_t*)*dev_ptr + mappedmem_size, 0, aligned_size - mappedmem_size);
  }
  return mappedmem_size;
}
