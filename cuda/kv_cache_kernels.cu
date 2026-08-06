#include "minillm/kv_cache_kernels.h"

#include <cstddef>

namespace minillm {
namespace {

__global__ void gatherKernel(const __half* pool, const std::int32_t* tables,
                             const std::int32_t* lengths, int batch,
                             int max_blocks, int layers, int kv_heads,
                             int head_dim, int block_size, int max_sequence,
                             __half* dense) {
  const std::size_t total = static_cast<std::size_t>(batch) * layers * 2 *
                            kv_heads * max_sequence * head_dim;
  for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total; linear += blockDim.x * gridDim.x) {
    std::size_t value = linear;
    const int dim = value % head_dim; value /= head_dim;
    const int token = value % max_sequence; value /= max_sequence;
    const int head = value % kv_heads; value /= kv_heads;
    const int request = value % batch; value /= batch;
    const int kv = value % 2; value /= 2;
    const int layer = static_cast<int>(value);
    if (token >= lengths[request]) {
      dense[linear] = __float2half(0.0f);
      continue;
    }
    const int logical = token / block_size;
    const int offset = token % block_size;
    const int physical = logical < max_blocks
        ? tables[request * max_blocks + logical] : -1;
    if (physical < 0) {
      dense[linear] = __float2half(0.0f);
      continue;
    }
    const std::size_t pool_index =
        (((((static_cast<std::size_t>(physical) * layers + layer) * 2 + kv) *
             kv_heads + head) * block_size + offset) * head_dim + dim);
    dense[linear] = pool[pool_index];
  }
}

__global__ void scatterKernel(const __half* dense_new,
                              const std::int32_t* tables,
                              const std::int32_t* starts, int batch,
                              int max_blocks, int layers, int kv_heads,
                              int head_dim, int block_size, int token_count,
                              __half* pool) {
  const std::size_t total = static_cast<std::size_t>(batch) * layers * 2 *
                            kv_heads * token_count * head_dim;
  for (std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total; linear += blockDim.x * gridDim.x) {
    std::size_t value = linear;
    const int dim = value % head_dim; value /= head_dim;
    const int query_token = value % token_count; value /= token_count;
    const int head = value % kv_heads; value /= kv_heads;
    const int request = value % batch; value /= batch;
    const int kv = value % 2; value /= 2;
    const int layer = static_cast<int>(value);
    const int token = starts[request] + query_token;
    const int logical = token / block_size;
    if (logical >= max_blocks) return;
    const int physical = tables[request * max_blocks + logical];
    if (physical < 0) return;
    const int offset = token % block_size;
    const std::size_t pool_index =
        (((((static_cast<std::size_t>(physical) * layers + layer) * 2 + kv) *
             kv_heads + head) * block_size + offset) * head_dim + dim);
    pool[pool_index] = dense_new[linear];
  }
}

int gridFor(std::size_t total) {
  constexpr int block = 256;
  const auto grid = static_cast<int>((total + block - 1) / block);
  return grid > 65535 ? 65535 : grid;
}

}  // namespace

cudaError_t launchGatherKv(const __half* pool, const std::int32_t* block_tables,
                           const std::int32_t* lengths, int batch,
                           int max_blocks_per_request, int layers, int kv_heads,
                           int head_dim, int block_size, int max_sequence,
                           __half* dense, cudaStream_t stream) {
  if (!pool || !block_tables || !lengths || !dense || batch <= 0 || layers <= 0 ||
      kv_heads <= 0 || head_dim <= 0 || block_size <= 0 || max_sequence <= 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t total = static_cast<std::size_t>(batch) * layers * 2 *
                            kv_heads * max_sequence * head_dim;
  gatherKernel<<<gridFor(total), 256, 0, stream>>>(
      pool, block_tables, lengths, batch, max_blocks_per_request, layers,
      kv_heads, head_dim, block_size, max_sequence, dense);
  return cudaPeekAtLastError();
}

cudaError_t launchScatterKv(const __half* dense_new,
                            const std::int32_t* block_tables,
                            const std::int32_t* start_positions, int batch,
                            int max_blocks_per_request, int layers, int kv_heads,
                            int head_dim, int block_size, int token_count,
                            __half* pool, cudaStream_t stream) {
  if (!dense_new || !block_tables || !start_positions || !pool || batch <= 0 ||
      layers <= 0 || kv_heads <= 0 || head_dim <= 0 || block_size <= 0 ||
      token_count <= 0) return cudaErrorInvalidValue;
  const std::size_t total = static_cast<std::size_t>(batch) * layers * 2 *
                            kv_heads * token_count * head_dim;
  scatterKernel<<<gridFor(total), 256, 0, stream>>>(
      dense_new, block_tables, start_positions, batch, max_blocks_per_request,
      layers, kv_heads, head_dim, block_size, token_count, pool);
  return cudaPeekAtLastError();
}

}  // namespace minillm
