#include "swiftinfer/qwen_tensorrt_backend.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "swiftinfer/greedy_sampler_cuda.h"
#include "swiftinfer/kv_cache_kernels.h"
#include "swiftinfer/tensorrt_backend.h"

namespace swiftinfer {
namespace {

void cudaCheck(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class CudaBuffer {
 public:
  explicit CudaBuffer(std::size_t bytes = 0) : bytes_(bytes) {
    if (bytes_) cudaCheck(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
  ~CudaBuffer() {
    if (data_) cudaFree(data_);
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  void* data() const noexcept { return data_; }
  template <typename T>
  T* as() const noexcept {
    return static_cast<T*>(data_);
  }

 private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

template <typename T>
void copyToDevice(CudaBuffer& destination, const std::vector<T>& source,
                  cudaStream_t stream) {
  cudaCheck(cudaMemcpyAsync(destination.data(), source.data(),
                            source.size() * sizeof(T), cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync host to device");
}

std::size_t denseSliceElements(const ModelSpec& spec, std::size_t batch,
                               std::size_t sequence) {
  return batch * static_cast<std::size_t>(spec.num_kv_heads) * sequence *
         static_cast<std::size_t>(spec.head_dim);
}

}  // namespace

struct QwenTensorRTBackend::Impl {
  Impl(ModelSpec model_spec, KVBlockManager& block_manager,
       const std::string& prefill_path, const std::string& decode_path,
       std::size_t batch_limit)
      : spec(model_spec),
        blocks(block_manager),
        max_batch(batch_limit),
        max_blocks_per_request(
            (static_cast<std::size_t>(spec.max_context) + blocks.blockSize() - 1) /
            blocks.blockSize()),
        prefill(prefill_path),
        decode_engine(decode_path),
        pool(blocks.capacityTokens() * spec.kvElementsPerToken() * sizeof(__half)),
        dense_kv(max_batch * static_cast<std::size_t>(spec.max_context) *
                 spec.kvElementsPerToken() * sizeof(__half)),
        dense_new(max_batch * spec.kvElementsPerToken() * sizeof(__half)),
        input_ids(std::max(max_batch, static_cast<std::size_t>(spec.max_context)) *
                  sizeof(std::int32_t)),
        position_ids(std::max(max_batch, static_cast<std::size_t>(spec.max_context)) *
                     sizeof(std::int32_t)),
        attention_mask(max_batch * static_cast<std::size_t>(spec.max_context) *
                       sizeof(std::int32_t)),
        block_tables(max_batch * max_blocks_per_request * sizeof(std::int32_t)),
        lengths(max_batch * sizeof(std::int32_t)),
        starts(max_batch * sizeof(std::int32_t)),
        logits(max_batch * static_cast<std::size_t>(spec.vocab_size) * sizeof(__half)),
        sampled_tokens(max_batch * sizeof(std::int32_t)) {
    if (max_batch == 0 || max_batch > 32) {
      throw std::invalid_argument("decode batch must be in [1, 32]");
    }
    cudaCheck(cudaMemset(pool.data(), 0,
                         blocks.capacityTokens() * spec.kvElementsPerToken() *
                             sizeof(__half)),
              "initialize KV pool");
    validateEngine(prefill);
    validateEngine(decode_engine);
  }

  void validateEngine(TensorRTBackend& engine) {
    if (engine.tensorDataType("logits") != nvinfer1::DataType::kHALF) {
      throw std::runtime_error("TensorRT logits must be FP16");
    }
    for (int layer = 0; layer < spec.num_layers; ++layer) {
      for (const char* prefix : {"present_key_", "present_value_"}) {
        const auto name = std::string(prefix) + std::to_string(layer);
        if (engine.tensorDataType(name) != nvinfer1::DataType::kHALF) {
          throw std::runtime_error(name + " must be FP16");
        }
      }
    }
  }

  void bindKvOutputs(TensorRTBackend& engine, CudaBuffer& buffer,
                     std::size_t batch, std::size_t sequence) {
    const auto slice = denseSliceElements(spec, batch, sequence);
    auto* base = buffer.as<__half>();
    for (int layer = 0; layer < spec.num_layers; ++layer) {
      engine.setTensorAddress("present_key_" + std::to_string(layer),
                              base + (static_cast<std::size_t>(layer) * 2) * slice);
      engine.setTensorAddress("present_value_" + std::to_string(layer),
                              base + (static_cast<std::size_t>(layer) * 2 + 1) * slice);
    }
  }

  std::vector<std::int32_t> paddedTable(const std::string& request_id) const {
    std::vector<std::int32_t> result(max_blocks_per_request, -1);
    const auto table = blocks.blockTable(request_id);
    std::copy(table.begin(), table.end(), result.begin());
    return result;
  }

  std::int32_t prefillOne(const std::shared_ptr<RequestState>& request) {
    const auto sequence = request->input_ids.size();
    if (sequence == 0 || sequence > static_cast<std::size_t>(spec.max_context)) {
      throw std::invalid_argument("invalid prefill sequence length");
    }
    auto table = paddedTable(request->request_id);
    std::vector<std::int32_t> mask(sequence, 1);
    std::vector<std::int32_t> positions(sequence);
    std::vector<std::int32_t> zero_start(1, 0);
    for (std::size_t index = 0; index < sequence; ++index) {
      positions[index] = static_cast<std::int32_t>(index);
    }
    const auto stream = prefill.stream();
    copyToDevice(input_ids, request->input_ids, stream);
    copyToDevice(attention_mask, mask, stream);
    copyToDevice(position_ids, positions, stream);
    copyToDevice(block_tables, table, stream);
    copyToDevice(starts, zero_start, stream);

    const std::vector<std::int64_t> input_shape{1,
                                                static_cast<std::int64_t>(sequence)};
    prefill.setInputShape("input_ids", input_shape);
    prefill.setInputShape("attention_mask", input_shape);
    prefill.setInputShape("position_ids", input_shape);
    prefill.setTensorAddress("input_ids", input_ids.data());
    prefill.setTensorAddress("attention_mask", attention_mask.data());
    prefill.setTensorAddress("position_ids", position_ids.data());
    prefill.setTensorAddress("logits", logits.data());
    bindKvOutputs(prefill, dense_kv, 1, sequence);
    prefill.enqueue();
    cudaCheck(launchGreedyArgmax(logits.as<__half>(), 1, spec.vocab_size,
                                 sampled_tokens.as<std::int32_t>(), stream),
              "prefill greedy argmax");
    cudaCheck(launchScatterKv(
                  dense_kv.as<__half>(), block_tables.as<std::int32_t>(),
                  starts.as<std::int32_t>(), 1,
                  static_cast<int>(max_blocks_per_request), spec.num_layers,
                  spec.num_kv_heads, spec.head_dim,
                  static_cast<int>(blocks.blockSize()), static_cast<int>(sequence),
                  pool.as<__half>(), stream),
              "prefill KV scatter");
    std::int32_t token{};
    cudaCheck(cudaMemcpyAsync(&token, sampled_tokens.data(), sizeof(token),
                              cudaMemcpyDeviceToHost, stream),
              "copy prefill token");
    prefill.synchronize();
    return token;
  }

  std::vector<std::int32_t> decodeExisting(
      const std::vector<std::shared_ptr<RequestState>>& requests) {
    const auto batch = requests.size();
    if (batch == 0) return {};
    if (batch > max_batch) throw std::invalid_argument("decode batch exceeds limit");
    std::vector<std::int32_t> host_ids(batch);
    std::vector<std::int32_t> host_positions(batch);
    std::vector<std::int32_t> host_lengths(batch);
    std::vector<std::int32_t> host_starts(batch);
    std::vector<std::int32_t> host_tables(batch * max_blocks_per_request, -1);
    std::size_t max_history = 0;
    for (std::size_t row = 0; row < batch; ++row) {
      const auto& request = requests[row];
      if (request->output_ids.empty()) {
        throw std::logic_error("decode request has no generated token");
      }
      const auto history = request->sequenceLength() - 1;
      if (history == 0 || history >= static_cast<std::size_t>(spec.max_context)) {
        throw std::invalid_argument("decode history is outside profile");
      }
      max_history = std::max(max_history, history);
      host_ids[row] = request->output_ids.back();
      host_positions[row] = static_cast<std::int32_t>(history);
      host_lengths[row] = static_cast<std::int32_t>(history);
      host_starts[row] = static_cast<std::int32_t>(history);
      const auto table = blocks.blockTable(request->request_id);
      std::copy(table.begin(), table.end(),
                host_tables.begin() + static_cast<std::ptrdiff_t>(row * max_blocks_per_request));
    }
    std::vector<std::int32_t> host_mask(batch * (max_history + 1), 0);
    for (std::size_t row = 0; row < batch; ++row) {
      const auto base = row * (max_history + 1);
      std::fill(host_mask.begin() + static_cast<std::ptrdiff_t>(base),
                host_mask.begin() + static_cast<std::ptrdiff_t>(base + host_lengths[row]), 1);
      host_mask[base + max_history] = 1;
    }

    const auto stream = decode_engine.stream();
    copyToDevice(input_ids, host_ids, stream);
    copyToDevice(position_ids, host_positions, stream);
    copyToDevice(attention_mask, host_mask, stream);
    copyToDevice(block_tables, host_tables, stream);
    copyToDevice(lengths, host_lengths, stream);
    copyToDevice(starts, host_starts, stream);
    cudaCheck(launchGatherKv(
                  pool.as<__half>(), block_tables.as<std::int32_t>(),
                  lengths.as<std::int32_t>(), static_cast<int>(batch),
                  static_cast<int>(max_blocks_per_request), spec.num_layers,
                  spec.num_kv_heads, spec.head_dim,
                  static_cast<int>(blocks.blockSize()), static_cast<int>(max_history),
                  dense_kv.as<__half>(), stream),
              "decode KV gather");

    decode_engine.setInputShape("input_ids",
                                {static_cast<std::int64_t>(batch), 1});
    decode_engine.setInputShape("position_ids",
                                {static_cast<std::int64_t>(batch), 1});
    decode_engine.setInputShape(
        "attention_mask",
        {static_cast<std::int64_t>(batch),
         static_cast<std::int64_t>(max_history + 1)});
    decode_engine.setTensorAddress("input_ids", input_ids.data());
    decode_engine.setTensorAddress("position_ids", position_ids.data());
    decode_engine.setTensorAddress("attention_mask", attention_mask.data());
    const auto past_slice = denseSliceElements(spec, batch, max_history);
    auto* past_base = dense_kv.as<__half>();
    for (int layer = 0; layer < spec.num_layers; ++layer) {
      const auto shape = std::vector<std::int64_t>{
          static_cast<std::int64_t>(batch), spec.num_kv_heads,
          static_cast<std::int64_t>(max_history), spec.head_dim};
      const auto key_name = "past_key_" + std::to_string(layer);
      const auto value_name = "past_value_" + std::to_string(layer);
      decode_engine.setInputShape(key_name, shape);
      decode_engine.setInputShape(value_name, shape);
      decode_engine.setTensorAddress(
          key_name, past_base + (static_cast<std::size_t>(layer) * 2) * past_slice);
      decode_engine.setTensorAddress(
          value_name,
          past_base + (static_cast<std::size_t>(layer) * 2 + 1) * past_slice);
    }
    decode_engine.setTensorAddress("logits", logits.data());
    bindKvOutputs(decode_engine, dense_new, batch, 1);
    decode_engine.enqueue();
    cudaCheck(launchGreedyArgmax(logits.as<__half>(), static_cast<int>(batch),
                                 spec.vocab_size,
                                 sampled_tokens.as<std::int32_t>(), stream),
              "decode greedy argmax");
    cudaCheck(launchScatterKv(
                  dense_new.as<__half>(), block_tables.as<std::int32_t>(),
                  starts.as<std::int32_t>(), static_cast<int>(batch),
                  static_cast<int>(max_blocks_per_request), spec.num_layers,
                  spec.num_kv_heads, spec.head_dim,
                  static_cast<int>(blocks.blockSize()), 1, pool.as<__half>(),
                  stream),
              "decode KV scatter");
    std::vector<std::int32_t> result(batch);
    cudaCheck(cudaMemcpyAsync(result.data(), sampled_tokens.data(),
                              batch * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, stream),
              "copy decode tokens");
    decode_engine.synchronize();
    return result;
  }

  ModelSpec spec;
  KVBlockManager& blocks;
  std::size_t max_batch;
  std::size_t max_blocks_per_request;
  TensorRTBackend prefill;
  TensorRTBackend decode_engine;
  CudaBuffer pool;
  CudaBuffer dense_kv;
  CudaBuffer dense_new;
  CudaBuffer input_ids;
  CudaBuffer position_ids;
  CudaBuffer attention_mask;
  CudaBuffer block_tables;
  CudaBuffer lengths;
  CudaBuffer starts;
  CudaBuffer logits;
  CudaBuffer sampled_tokens;
  std::unordered_map<std::string, std::int32_t> staged_tokens;
};

QwenTensorRTBackend::QwenTensorRTBackend(
    ModelSpec spec, KVBlockManager& blocks, const std::string& prefill_engine,
    const std::string& decode_engine, std::size_t max_decode_batch)
    : impl_(std::make_unique<Impl>(spec, blocks, prefill_engine, decode_engine,
                                   max_decode_batch)) {}

QwenTensorRTBackend::~QwenTensorRTBackend() = default;

void QwenTensorRTBackend::prefill(
    const std::vector<std::shared_ptr<RequestState>>& requests) {
  for (const auto& request : requests) {
    if (impl_->staged_tokens.count(request->request_id)) {
      throw std::logic_error("duplicate staged prefill token");
    }
    impl_->staged_tokens.emplace(request->request_id,
                                 impl_->prefillOne(request));
  }
}

std::vector<std::int32_t> QwenTensorRTBackend::decode(
    const std::vector<std::shared_ptr<RequestState>>& requests) {
  std::vector<std::int32_t> result(requests.size(), -1);
  std::map<std::size_t, std::vector<std::pair<std::size_t,
                                              std::shared_ptr<RequestState>>>>
      groups;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto staged = impl_->staged_tokens.find(requests[index]->request_id);
    if (staged != impl_->staged_tokens.end()) {
      result[index] = staged->second;
      impl_->staged_tokens.erase(staged);
    } else {
      const auto history = requests[index]->sequenceLength() - 1;
      groups[history].emplace_back(index, requests[index]);
    }
  }
  // TensorRT's FP16 attention path can change a near-tie token when a short
  // history is right-padded by hundreds or thousands of positions.  Execute
  // each exact-history bucket separately: continuous batching is retained for
  // equal-length requests, while the production path remains numerically
  // equivalent to B1 instead of depending on unrelated active request lengths.
  for (const auto& [history, indexed_requests] : groups) {
    (void)history;
    std::vector<std::shared_ptr<RequestState>> bucket;
    bucket.reserve(indexed_requests.size());
    for (const auto& [index, request] : indexed_requests) {
      (void)index;
      bucket.push_back(request);
    }
    const auto decoded = impl_->decodeExisting(bucket);
    for (std::size_t row = 0; row < decoded.size(); ++row) {
      result[indexed_requests[row].first] = decoded[row];
    }
  }
  return result;
}

}  // namespace swiftinfer
