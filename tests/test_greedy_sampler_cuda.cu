#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "minillm/greedy_sampler_cuda.h"

namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

int main() {
  constexpr int batch = 8;
  constexpr int vocab = 151936;
  std::vector<__half> logits(static_cast<std::size_t>(batch) * vocab,
                             __float2half(-10.0f));
  std::vector<std::int32_t> expected(batch);
  for (int request = 0; request < batch; ++request) {
    expected[request] = (request * 19391 + 17) % vocab;
    logits[static_cast<std::size_t>(request) * vocab + expected[request]] =
        __float2half(4.0f);
  }
  // Equal values must select the smaller token ID.
  logits[5] = __float2half(7.0f);
  logits[9] = __float2half(7.0f);
  expected[0] = 5;

  __half* device_logits{};
  std::int32_t* device_tokens{};
  check(cudaMalloc(&device_logits, logits.size() * sizeof(__half)), "allocate logits");
  check(cudaMalloc(&device_tokens, batch * sizeof(std::int32_t)), "allocate tokens");
  check(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(__half),
                   cudaMemcpyHostToDevice),
        "copy logits");
  check(minillm::launchGreedyArgmax(device_logits, batch, vocab, device_tokens,
                                    nullptr),
        "launch argmax");
  std::vector<std::int32_t> actual(batch);
  check(cudaMemcpy(actual.data(), device_tokens, batch * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost),
        "copy tokens");
  if (actual != expected) throw std::runtime_error("CUDA greedy argmax mismatch");
  cudaFree(device_tokens);
  cudaFree(device_logits);
  std::cout << "greedy_sampler_cuda: PASS\n";
}
