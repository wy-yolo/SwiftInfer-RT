#include "swiftinfer/greedy_sampler.h"

#include <stdexcept>

namespace swiftinfer {

std::int32_t greedySample(const float* logits, std::size_t vocab_size) {
  if (!logits || vocab_size == 0) throw std::invalid_argument("empty logits");
  std::size_t best = 0;
  for (std::size_t i = 1; i < vocab_size; ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace swiftinfer

