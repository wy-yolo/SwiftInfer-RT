#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "swiftinfer/kv_cache_kernels.h"

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void run(int batch, int sequence) {
  constexpr int layers = 24;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 64;
  constexpr int block_size = 16;
  const int max_blocks = (sequence + block_size - 1) / block_size;
  const int total_blocks = batch * max_blocks;
  const std::size_t pool_elements = static_cast<std::size_t>(total_blocks) * layers * 2 *
                                    kv_heads * block_size * head_dim;
  const std::size_t dense_elements = static_cast<std::size_t>(batch) * layers * 2 *
                                     kv_heads * sequence * head_dim;
  std::vector<std::int32_t> tables(static_cast<std::size_t>(batch) * max_blocks);
  std::vector<std::int32_t> lengths(batch, sequence);
  for (int b = 0; b < batch; ++b) {
    for (int block = 0; block < max_blocks; ++block) {
      tables[b * max_blocks + block] = b * max_blocks + block;
    }
  }
  __half *pool{}, *dense{};
  std::int32_t *device_tables{}, *device_lengths{};
  check(cudaMalloc(&pool, pool_elements * sizeof(__half)), "pool allocation");
  check(cudaMalloc(&dense, dense_elements * sizeof(__half)), "dense allocation");
  check(cudaMalloc(&device_tables, tables.size() * sizeof(std::int32_t)), "table allocation");
  check(cudaMalloc(&device_lengths, lengths.size() * sizeof(std::int32_t)), "length allocation");
  check(cudaMemset(pool, 0, pool_elements * sizeof(__half)), "pool initialization");
  check(cudaMemcpy(device_tables, tables.data(), tables.size() * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice), "table copy");
  check(cudaMemcpy(device_lengths, lengths.data(), lengths.size() * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice), "length copy");
  for (int i = 0; i < 20; ++i) {
    check(swiftinfer::launchGatherKv(pool, device_tables, device_lengths, batch,
                                 max_blocks, layers, kv_heads, head_dim, block_size,
                                 sequence, dense, nullptr), "warmup launch");
  }
  check(cudaDeviceSynchronize(), "warmup sync");
  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start), "event create");
  check(cudaEventCreate(&stop), "event create");
  check(cudaEventRecord(start), "event record");
  constexpr int iterations = 100;
  for (int i = 0; i < iterations; ++i) {
    check(swiftinfer::launchGatherKv(pool, device_tables, device_lengths, batch,
                                 max_blocks, layers, kv_heads, head_dim, block_size,
                                 sequence, dense, nullptr), "benchmark launch");
  }
  check(cudaEventRecord(stop), "event record");
  check(cudaEventSynchronize(stop), "event sync");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
  const double bytes = static_cast<double>(dense_elements) * sizeof(__half) * 2.0;
  const double average_ms = elapsed_ms / iterations;
  const double bandwidth_gbps = bytes / (average_ms / 1000.0) / 1.0e9;
  std::cout << batch << ',' << sequence << ',' << average_ms << ','
            << bandwidth_gbps << ',' << (pool_elements * sizeof(__half)) / (1024.0 * 1024.0)
            << '\n';
  cudaEventDestroy(stop); cudaEventDestroy(start);
  cudaFree(device_lengths); cudaFree(device_tables); cudaFree(dense); cudaFree(pool);
}

}  // namespace

int main() {
  std::cout << "batch,sequence,gather_ms,effective_gbps,pool_mib\n";
  for (int batch : {1, 8, 16, 32}) {
    for (int sequence : {256, 1024, 2048, 3968}) run(batch, sequence);
  }
}

