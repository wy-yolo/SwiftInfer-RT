#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minillm {

enum class FinishReason { kNone, kEos, kLength, kCancelled, kError };

struct RequestState {
  std::string request_id;
  std::vector<std::int32_t> input_ids;
  std::vector<std::int32_t> output_ids;
  std::size_t max_new_tokens{32};
  FinishReason finish_reason{FinishReason::kNone};
  bool prefetched{false};

  std::size_t sequenceLength() const { return input_ids.size() + output_ids.size(); }
  bool finished() const { return finish_reason != FinishReason::kNone; }
};

}  // namespace minillm

