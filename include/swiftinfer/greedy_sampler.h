#pragma once

#include <cstddef>
#include <cstdint>

namespace swiftinfer {

std::int32_t greedySample(const float* logits, std::size_t vocab_size);

}  // namespace swiftinfer

