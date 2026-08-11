#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "swiftinfer/kv_block_manager.h"
#include "swiftinfer/model_spec.h"
#include "swiftinfer/request_state.h"
#include "swiftinfer/runtime_session.h"
#include "swiftinfer/scheduler.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class DeterministicBackend final : public swiftinfer::ModelBackend {
 public:
  void prefill(
      const std::vector<std::shared_ptr<swiftinfer::RequestState>>& requests) override {
    prefill_count += requests.size();
  }

  std::vector<std::int32_t> decode(
      const std::vector<std::shared_ptr<swiftinfer::RequestState>>& requests) override {
    std::vector<std::int32_t> tokens;
    tokens.reserve(requests.size());
    for (const auto& request : requests) {
      tokens.push_back(request->output_ids.empty() ? 7 : 8);
    }
    return tokens;
  }

  std::size_t prefill_count{0};
};

}  // namespace

int main() {
  swiftinfer::ModelSpec spec;
  spec.eos_token_id = 99;
  swiftinfer::KVBlockManager blocks(16, 16);
  swiftinfer::Scheduler scheduler(2, 4, blocks);
  for (const char* id : {"a", "b", "c"}) {
    swiftinfer::RequestState request;
    request.request_id = id;
    request.input_ids = {1, 2, 3};
    request.max_new_tokens = 2;
    scheduler.submit(std::move(request));
  }
  DeterministicBackend backend;
  swiftinfer::RuntimeSession session(spec, scheduler, backend, 2);
  session.run();
  auto completed = scheduler.takeCompleted();
  require(completed.size() == 3, "all requests must complete");
  require(backend.prefill_count == 3, "each request must prefill exactly once");
  require(blocks.freeBlockCount() == 16, "all KV blocks must be released");
  for (const auto& request : completed) {
    require(request->output_ids == std::vector<std::int32_t>({7, 8}),
            "unexpected generated sequence");
    require(request->finish_reason == swiftinfer::FinishReason::kLength,
            "request must finish by length");
    require(request->first_token_at >= request->submitted_at,
            "first-token timestamp is invalid");
    require(request->completed_at >= request->first_token_at,
            "completion timestamp is invalid");
  }
}
