#include "minillm/tensorrt_backend.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace minillm {

class TensorRTBackend::Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

namespace {

std::vector<char> readFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open TensorRT engine: " + path);
  const auto size = input.tellg();
  if (size <= 0) throw std::runtime_error("empty TensorRT engine: " + path);
  std::vector<char> data(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(data.data(), size);
  if (!input) throw std::runtime_error("failed to read TensorRT engine: " + path);
  return data;
}

void cudaCheck(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace

TensorRTBackend::TensorRTBackend(const std::string& engine_path)
    : logger_(std::make_unique<Logger>()) {
  auto data = readFile(engine_path);
  runtime_ = nvinfer1::createInferRuntime(*logger_);
  if (!runtime_) throw std::runtime_error("failed to create TensorRT runtime");
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) throw std::runtime_error("failed to deserialize TensorRT engine");
  context_ = engine_->createExecutionContext();
  if (!context_) throw std::runtime_error("failed to create TensorRT context");
  cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
}

TensorRTBackend::~TensorRTBackend() {
  if (stream_) cudaStreamDestroy(stream_);
  delete context_;
  delete engine_;
  delete runtime_;
}

void TensorRTBackend::setInputShape(const std::string& name,
                                    const std::vector<std::int64_t>& dims) {
  if (dims.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw std::invalid_argument("too many tensor dimensions");
  }
  nvinfer1::Dims trt_dims;
  trt_dims.nbDims = static_cast<int>(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) trt_dims.d[i] = dims[i];
  if (!context_->setInputShape(name.c_str(), trt_dims)) {
    throw std::runtime_error("failed to set shape for tensor " + name);
  }
}

void TensorRTBackend::setTensorAddress(const std::string& name, void* address) {
  if (!address) throw std::invalid_argument("null tensor address for " + name);
  if (!context_->setTensorAddress(name.c_str(), address)) {
    throw std::runtime_error("failed to set address for tensor " + name);
  }
}

void TensorRTBackend::enqueue() {
  if (!context_->allInputDimensionsSpecified()) {
    throw std::runtime_error("not all TensorRT input shapes are specified");
  }
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
}

void TensorRTBackend::synchronize() {
  cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

std::vector<std::string> TensorRTBackend::inputNames() const {
  std::vector<std::string> names;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) names.emplace_back(name);
  }
  return names;
}

std::vector<std::string> TensorRTBackend::outputNames() const {
  std::vector<std::string> names;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) names.emplace_back(name);
  }
  return names;
}

}  // namespace minillm

