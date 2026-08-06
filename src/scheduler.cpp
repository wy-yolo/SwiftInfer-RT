#include "minillm/scheduler.h"

#include <algorithm>
#include <stdexcept>

namespace minillm {

Scheduler::Scheduler(std::size_t max_active, std::size_t max_total,
                     KVBlockManager& blocks)
    : max_active_(max_active), max_total_(max_total), blocks_(blocks) {
  if (max_active == 0 || max_total < max_active) {
    throw std::invalid_argument("invalid scheduler capacity");
  }
}

void Scheduler::submit(RequestState request) {
  if (request.request_id.empty() || request.input_ids.empty() ||
      request.max_new_tokens == 0) {
    throw std::invalid_argument("invalid request");
  }
  if (ids_.count(request.request_id)) throw std::invalid_argument("duplicate request id");
  if (ids_.size() >= max_total_) throw std::runtime_error("scheduler queue is full");
  ids_.insert(request.request_id);
  waiting_.push_back(std::make_shared<RequestState>(std::move(request)));
  admit();
}

void Scheduler::admit() {
  while (active_.size() < max_active_ && !waiting_.empty()) {
    auto request = waiting_.front();
    blocks_.ensureCapacity(request->request_id, request->input_ids.size());
    waiting_.pop_front();
    active_.push_back(std::move(request));
  }
}

std::vector<std::shared_ptr<RequestState>> Scheduler::nextBatch(std::size_t limit) {
  admit();
  const auto count = limit == 0 ? active_.size() : std::min(limit, active_.size());
  return {active_.begin(), active_.begin() + static_cast<std::ptrdiff_t>(count)};
}

void Scheduler::finishAndErase(std::size_t index, FinishReason reason) {
  auto request = active_.at(index);
  request->finish_reason = reason;
  blocks_.release(request->request_id);
  ids_.erase(request->request_id);
  completed_.push_back(request);
  active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(index));
  admit();
}

void Scheduler::appendToken(const std::string& request_id, std::int32_t token,
                            std::int32_t eos_token_id) {
  auto it = std::find_if(active_.begin(), active_.end(), [&](const auto& item) {
    return item->request_id == request_id;
  });
  if (it == active_.end()) throw std::out_of_range("request is not active");
  const auto index = static_cast<std::size_t>(std::distance(active_.begin(), it));
  auto& request = *it;
  blocks_.ensureCapacity(request_id, request->sequenceLength() + 1);
  request->output_ids.push_back(token);
  if (token == eos_token_id) {
    finishAndErase(index, FinishReason::kEos);
  } else if (request->output_ids.size() >= request->max_new_tokens) {
    finishAndErase(index, FinishReason::kLength);
  }
}

void Scheduler::cancel(const std::string& request_id) {
  auto active_it = std::find_if(active_.begin(), active_.end(), [&](const auto& item) {
    return item->request_id == request_id;
  });
  if (active_it != active_.end()) {
    finishAndErase(static_cast<std::size_t>(std::distance(active_.begin(), active_it)),
                   FinishReason::kCancelled);
    return;
  }
  auto waiting_it = std::find_if(waiting_.begin(), waiting_.end(), [&](const auto& item) {
    return item->request_id == request_id;
  });
  if (waiting_it == waiting_.end()) throw std::out_of_range("unknown request");
  (*waiting_it)->finish_reason = FinishReason::kCancelled;
  completed_.push_back(*waiting_it);
  ids_.erase(request_id);
  waiting_.erase(waiting_it);
}

std::vector<std::shared_ptr<RequestState>> Scheduler::takeCompleted() {
  auto completed = std::move(completed_);
  completed_.clear();
  return completed;
}

}  // namespace minillm
