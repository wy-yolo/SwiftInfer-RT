#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "minillm/model_spec.h"
#include "minillm/request_state.h"
#include "minillm/scheduler.h"

namespace minillm {

class ModelBackend {
 public:
  virtual ~ModelBackend() = default;
  virtual void prefill(const std::vector<std::shared_ptr<RequestState>>& requests) = 0;
  virtual std::vector<std::int32_t> decode(
      const std::vector<std::shared_ptr<RequestState>>& requests) = 0;
};

class RuntimeSession {
 public:
  RuntimeSession(ModelSpec spec, Scheduler& scheduler, ModelBackend& backend,
                 std::size_t decode_batch_limit);
  void run();

 private:
  ModelSpec spec_;
  Scheduler& scheduler_;
  ModelBackend& backend_;
  std::size_t decode_batch_limit_;
};

}  // namespace minillm

