#include "minillm/greedy_sampler_cuda.h"

#include <cfloat>

namespace minillm {
namespace {

struct Candidate {
  float value;
  int index;
};

__device__ Candidate better(Candidate left, Candidate right) {
  return right.value > left.value ||
                 (right.value == left.value && right.index < left.index)
             ? right
             : left;
}

__global__ void greedyArgmaxKernel(const __half* logits, int vocab_size,
                                   std::int32_t* output) {
  const int request = blockIdx.x;
  Candidate candidate{-FLT_MAX, 0};
  const __half* row = logits + static_cast<std::size_t>(request) * vocab_size;
  for (int index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    candidate = better(candidate, Candidate{__half2float(row[index]), index});
  }
  __shared__ float values[256];
  __shared__ int indices[256];
  values[threadIdx.x] = candidate.value;
  indices[threadIdx.x] = candidate.index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const Candidate merged = better(
          Candidate{values[threadIdx.x], indices[threadIdx.x]},
          Candidate{values[threadIdx.x + stride], indices[threadIdx.x + stride]});
      values[threadIdx.x] = merged.value;
      indices[threadIdx.x] = merged.index;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[request] = indices[0];
}

}  // namespace

cudaError_t launchGreedyArgmax(const __half* logits, int batch, int vocab_size,
                               std::int32_t* output_tokens,
                               cudaStream_t stream) {
  if (!logits || !output_tokens || batch <= 0 || vocab_size <= 0) {
    return cudaErrorInvalidValue;
  }
  greedyArgmaxKernel<<<batch, 256, 0, stream>>>(logits, vocab_size,
                                                output_tokens);
  return cudaPeekAtLastError();
}

}  // namespace minillm
