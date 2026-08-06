#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace minillm {

cudaError_t launchGreedyArgmax(const __half* logits, int batch, int vocab_size,
                               std::int32_t* output_tokens,
                               cudaStream_t stream);

}  // namespace minillm
