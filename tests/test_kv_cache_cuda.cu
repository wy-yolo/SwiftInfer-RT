#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void runCase(int length, int batch) {
  constexpr int block_size = 16;
  constexpr int layers = 3;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 2;
  const int blocks_per_request = (length + block_size - 1) / block_size;
  const int total_blocks = blocks_per_request * batch;
  const int max_sequence = length + 1;
  const std::size_t pool_count = static_cast<std::size_t>(total_blocks) * layers * 2 *
                                 kv_heads * block_size * head_dim;
  const std::size_t dense_new_count = static_cast<std::size_t>(batch) * layers * 2 *
                                      kv_heads * length * head_dim;
  const std::size_t dense_out_count = static_cast<std::size_t>(batch) * layers * 2 *
                                      kv_heads * max_sequence * head_dim;

  std::vector<__half> pool(pool_count, __float2half(-1.0f));
  std::vector<__half> dense_new(dense_new_count);
  std::vector<__half> dense_out(dense_out_count);
  std::vector<std::int32_t> tables(static_cast<std::size_t>(batch) * blocks_per_request);
  std::vector<std::int32_t> starts(batch, 0);
  std::vector<std::int32_t> lengths(batch, length);
  for (int b = 0; b < batch; ++b) {
    for (int block = 0; block < blocks_per_request; ++block) {
      tables[b * blocks_per_request + block] = b * blocks_per_request + block;
    }
  }
  for (std::size_t i = 0; i < dense_new.size(); ++i) {
    dense_new[i] = __float2half(static_cast<float>((i % 997) + 1));
  }

  __half *d_pool{}, *d_new{}, *d_out{};
  std::int32_t *d_tables{}, *d_starts{}, *d_lengths{};
  check(cudaMalloc(&d_pool, pool.size() * sizeof(__half)), "cudaMalloc pool");
  check(cudaMalloc(&d_new, dense_new.size() * sizeof(__half)), "cudaMalloc dense_new");
  check(cudaMalloc(&d_out, dense_out.size() * sizeof(__half)), "cudaMalloc dense_out");
  check(cudaMalloc(&d_tables, tables.size() * sizeof(std::int32_t)), "cudaMalloc tables");
  check(cudaMalloc(&d_starts, starts.size() * sizeof(std::int32_t)), "cudaMalloc starts");
  check(cudaMalloc(&d_lengths, lengths.size() * sizeof(std::int32_t)), "cudaMalloc lengths");
  check(cudaMemcpy(d_pool, pool.data(), pool.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy pool");
  check(cudaMemcpy(d_new, dense_new.data(), dense_new.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy dense");
  check(cudaMemcpy(d_tables, tables.data(), tables.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "copy tables");
  check(cudaMemcpy(d_starts, starts.data(), starts.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "copy starts");
  check(cudaMemcpy(d_lengths, lengths.data(), lengths.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "copy lengths");

  check(swiftinfer::launchScatterKv(d_new, d_tables, d_starts, batch,
                                 blocks_per_request, layers, kv_heads, head_dim,
                                 block_size, length, d_pool, nullptr), "scatter launch");
  check(swiftinfer::launchGatherKv(d_pool, d_tables, d_lengths, batch,
                                blocks_per_request, layers, kv_heads, head_dim,
                                block_size, max_sequence, d_out, nullptr), "gather launch");
  check(cudaDeviceSynchronize(), "synchronize");
  check(cudaMemcpy(dense_out.data(), d_out, dense_out.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost), "copy result");

  for (int layer = 0; layer < layers; ++layer) {
    for (int kv = 0; kv < 2; ++kv) {
      for (int b = 0; b < batch; ++b) {
        for (int head = 0; head < kv_heads; ++head) {
          for (int token = 0; token < max_sequence; ++token) {
            for (int dim = 0; dim < head_dim; ++dim) {
          const std::size_t out_index =
              ((((((static_cast<std::size_t>(layer) * 2 + kv) * batch + b) *
                    kv_heads + head) * max_sequence + token) * head_dim + dim));
          if (token == length) {
            require(__half2float(dense_out[out_index]) == 0.0f, "padding is not zero");
          } else {
            const std::size_t input_index =
                ((((((static_cast<std::size_t>(layer) * 2 + kv) * batch + b) *
                      kv_heads + head) * length + token) * head_dim + dim));
            require(std::fabs(__half2float(dense_out[out_index]) -
                              __half2float(dense_new[input_index])) < 0.01f,
                    "gather/scatter mismatch");
          }
            }
          }
        }
      }
    }
  }
  cudaFree(d_lengths); cudaFree(d_starts); cudaFree(d_tables);
  cudaFree(d_out); cudaFree(d_new); cudaFree(d_pool);
}

}  // namespace

int main() {
  for (int length : {1, 15, 16, 17, 255, 256}) runCase(length, 1);
  runCase(4095, 32);
  std::cout << "kv_cache_cuda: PASS\n";
}
