#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace minillm {

class TensorRTBackend {
 public:
  explicit TensorRTBackend(const std::string& engine_path);
  ~TensorRTBackend();
  TensorRTBackend(const TensorRTBackend&) = delete;
  TensorRTBackend& operator=(const TensorRTBackend&) = delete;

  void setInputShape(const std::string& name, const std::vector<std::int64_t>& dims);
  void setTensorAddress(const std::string& name, void* address);
  void enqueue();
  void synchronize();
  cudaStream_t stream() const noexcept { return stream_; }
  std::vector<std::string> inputNames() const;
  std::vector<std::string> outputNames() const;

 private:
  class Logger;
  std::unique_ptr<Logger> logger_;
  nvinfer1::IRuntime* runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};
  nvinfer1::IExecutionContext* context_{nullptr};
  cudaStream_t stream_{nullptr};
};

}  // namespace minillm

