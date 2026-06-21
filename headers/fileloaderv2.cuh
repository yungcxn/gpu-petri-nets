#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#define STDIN_BUFFER_BYTES (1024 * 1024)
#define MALLOC_PADDING 128

#include "common.cuh"

/* returns size of mallocd memory */
uint32_t fl_stdin_managed_malloc(uint8_t** host_managed_ptr) {
  cudaMallocManaged(host_managed_ptr, STDIN_BUFFER_BYTES);
  uint32_t bytes_read = fread(*host_managed_ptr, 1, STDIN_BUFFER_BYTES, stdin);
  ASSERT(bytes_read > 0);
  return bytes_read;
}

uint32_t fl_file_managed_malloc(const char* filename,
                                uint8_t** host_managed_ptr) {

  FILE* file = fopen(filename, "rb");
  ASSERT(file != NULL);
  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);
  cudaMallocManaged(host_managed_ptr, file_size);
  fread(*host_managed_ptr, 1, file_size, file);
  fclose(file);
  return (uint32_t)file_size;
}

uint32_t fl_stdin_dev_malloc(uint8_t** host_managed_ptr, uint8_t** dev_ptr) {
  uint32_t managedmem_size = fl_stdin_managed_malloc(host_managed_ptr);
  uint32_t aligned_size = ROUND_UP(managedmem_size, MALLOC_PADDING);
  cudaMalloc(dev_ptr, aligned_size);
  cudaMemcpy(*dev_ptr, *host_managed_ptr,
             managedmem_size, cudaMemcpyHostToDevice);
  if (aligned_size > managedmem_size) {
    cudaMemset((uint8_t*)*dev_ptr + managedmem_size, 0, 
               aligned_size - managedmem_size);
  }
  return aligned_size;
}

uint32_t fl_file_dev_malloc(const char* filename, uint8_t** host_managed_ptr,
                            uint8_t** dev_ptr) {

  uint32_t managedmem_size = fl_file_managed_malloc(filename, host_managed_ptr);
  uint32_t aligned_size = ROUND_UP(managedmem_size, MALLOC_PADDING);
  cudaMalloc(dev_ptr, aligned_size);
  cudaMemcpy(*dev_ptr, *host_managed_ptr, managedmem_size, 
             cudaMemcpyHostToDevice);
  if (aligned_size > managedmem_size) {
    cudaMemset((uint8_t*)*dev_ptr + managedmem_size, 0, 
               aligned_size - managedmem_size);
  }
  return aligned_size;
}