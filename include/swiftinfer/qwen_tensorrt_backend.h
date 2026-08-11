#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "swiftinfer/kv_block_manager.h"
#include "swiftinfer/model_spec.h"
#include "swiftinfer/runtime_session.h"

namespace swiftinfer {

class QwenTensorRTBackend final : public ModelBackend {
 public:
  QwenTensorRTBackend(ModelSpec spec, KVBlockManager& blocks,
                      const std::string& prefill_engine,
                      const std::string& decode_engine,
                      std::size_t max_decode_batch = 32);
  ~QwenTensorRTBackend() override;

  void prefill(
      const std::vector<std::shared_ptr<RequestState>>& requests) override;
  std::vector<std::int32_t> decode(
      const std::vector<std::shared_ptr<RequestState>>& requests) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace swiftinfer
