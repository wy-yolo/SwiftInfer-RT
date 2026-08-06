#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "minillm/kv_block_manager.h"
#include "minillm/request_state.h"

namespace minillm {

class Scheduler {
 public:
  Scheduler(std::size_t max_active, std::size_t max_total,
            KVBlockManager& blocks);

  void submit(RequestState request);
  std::vector<std::shared_ptr<RequestState>> nextBatch(std::size_t limit = 0);
  void appendToken(const std::string& request_id, std::int32_t token,
                   std::int32_t eos_token_id);
  void cancel(const std::string& request_id);
  std::vector<std::shared_ptr<RequestState>> takeCompleted();
  bool empty() const { return waiting_.empty() && active_.empty(); }
  std::size_t activeCount() const { return active_.size(); }
  std::size_t waitingCount() const { return waiting_.size(); }

 private:
  void admit();
  void finishAndErase(std::size_t index, FinishReason reason);

  std::size_t max_active_;
  std::size_t max_total_;
  KVBlockManager& blocks_;
  std::deque<std::shared_ptr<RequestState>> waiting_;
  std::vector<std::shared_ptr<RequestState>> active_;
  std::vector<std::shared_ptr<RequestState>> completed_;
  std::unordered_set<std::string> ids_;
};

}  // namespace minillm
