#include "swiftinfer/runtime_session.h"

#include <stdexcept>

namespace swiftinfer {

RuntimeSession::RuntimeSession(ModelSpec spec, Scheduler& scheduler,
                               ModelBackend& backend,
                               std::size_t decode_batch_limit)
    : spec_(spec),
      scheduler_(scheduler),
      backend_(backend),
      decode_batch_limit_(decode_batch_limit) {
  if (decode_batch_limit == 0) throw std::invalid_argument("decode batch limit is zero");
}

void RuntimeSession::run() {
  while (!scheduler_.empty()) {
    auto batch = scheduler_.nextBatch(decode_batch_limit_);
    if (batch.empty()) throw std::runtime_error("scheduler made no progress");
    std::vector<std::shared_ptr<RequestState>> prefill;
    for (auto& request : batch) {
      if (!request->prefetched) prefill.push_back(request);
    }
    if (!prefill.empty()) {
      backend_.prefill(prefill);
      for (auto& request : prefill) request->prefetched = true;
    }
    auto tokens = backend_.decode(batch);
    if (tokens.size() != batch.size()) throw std::runtime_error("backend batch mismatch");
    for (std::size_t i = 0; i < batch.size(); ++i) {
      scheduler_.appendToken(batch[i]->request_id, tokens[i], spec_.eos_token_id);
    }
  }
}

}  // namespace swiftinfer

