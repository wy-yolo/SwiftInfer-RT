#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace swiftinfer {

// Dense tensors use [layer, key_or_value, batch, kv_head, token, head_dim].
// Each TensorRT K/V tensor is therefore one contiguous slice.

cudaError_t launchGatherKv(const __half* pool, const std::int32_t* block_tables,
                           const std::int32_t* lengths, int batch,
                           int max_blocks_per_request, int layers, int kv_heads,
                           int head_dim, int block_size, int max_sequence,
                           __half* dense, cudaStream_t stream);

cudaError_t launchScatterKv(const __half* dense_new,
                            const std::int32_t* block_tables,
                            const std::int32_t* start_positions, int batch,
                            int max_blocks_per_request, int layers, int kv_heads,
                            int head_dim, int block_size, int token_count,
                            __half* pool, cudaStream_t stream);

}  // namespace swiftinfer
