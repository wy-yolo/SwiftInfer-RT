#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "swiftinfer/kv_block_manager.h"
#include "swiftinfer/model_spec.h"
#ifdef SWIFTINFER_WITH_TENSORRT
#include "swiftinfer/qwen_tensorrt_backend.h"
#endif
#include "swiftinfer/request_state.h"
#include "swiftinfer/runtime_session.h"
#include "swiftinfer/scheduler.h"

namespace {

struct Options {
  bool validate_jsonl{false};
  bool generate_jsonl{false};
  bool flush_on_empty_line{false};
  std::size_t max_active{32};
  std::size_t max_total{64};
  std::string prefill_engine;
  std::string decode_engine;
  std::string model_spec;
};

std::string readFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open file: " + path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string stringField(const std::string& line, const std::string& name) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) {
    throw std::invalid_argument("missing or invalid " + name);
  }
  return match[1].str();
}

std::string bestEffortRequestId(const std::string& line) {
  try {
    return stringField(line, "request_id");
  } catch (...) {
    return {};
  }
}

std::size_t integerField(const std::string& line, const std::string& name) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) {
    throw std::invalid_argument("missing or invalid " + name);
  }
  return static_cast<std::size_t>(std::stoull(match[1].str()));
}

int optionalIntegerField(const std::string& text, const std::string& name,
                         int fallback) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) return fallback;
  return std::stoi(match[1].str());
}

std::vector<std::int32_t> integerArray(const std::string& line,
                                       const std::string& name) {
  // libstdc++'s ECMAScript regex executor is recursive.  Applying a repeated
  // capture to a 4K-token JSON array can recurse more than 100k frames and
  // overflow the host stack, so parse this flat integer array linearly.
  const auto key = "\"" + name + "\"";
  const auto key_position = line.find(key);
  if (key_position == std::string::npos) {
    throw std::invalid_argument("missing or invalid " + name);
  }
  const auto colon = line.find(':', key_position + key.size());
  const auto open = colon == std::string::npos ? std::string::npos
                                               : line.find('[', colon + 1);
  const auto close = open == std::string::npos ? std::string::npos
                                                : line.find(']', open + 1);
  if (open == std::string::npos || close == std::string::npos) {
    throw std::invalid_argument("missing or invalid " + name);
  }
  std::vector<std::int32_t> values;
  std::stringstream stream(line.substr(open + 1, close - open - 1));
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = trim(item);
    if (item.empty()) throw std::invalid_argument(name + " contains an empty item");
    std::size_t consumed = 0;
    const auto value = std::stoll(item, &consumed);
    if (consumed != item.size() || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument(name + " contains an invalid token");
    }
    values.push_back(static_cast<std::int32_t>(value));
  }
  if (values.empty()) throw std::invalid_argument("input_ids is empty");
  return values;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        } else {
          output << character;
        }
    }
  }
  return output.str();
}

std::string finishReason(swiftinfer::FinishReason reason) {
  switch (reason) {
    case swiftinfer::FinishReason::kEos: return "eos";
    case swiftinfer::FinishReason::kLength: return "length";
    case swiftinfer::FinishReason::kCancelled: return "cancelled";
    case swiftinfer::FinishReason::kError: return "error";
    case swiftinfer::FinishReason::kNone: return "none";
  }
  return "error";
}

double milliseconds(swiftinfer::RequestState::Clock::time_point begin,
                    swiftinfer::RequestState::Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void writeError(const std::string& request_id, const std::string& message) {
  std::cout << "{\"request_id\":\"" << jsonEscape(request_id)
            << "\",\"output_ids\":[],\"finish_reason\":\"error\",\"error\":\""
            << jsonEscape(message) << "\"}\n";
}

void writeResponse(const swiftinfer::RequestState& request) {
  const double ttft = milliseconds(request.submitted_at, request.first_token_at);
  const double tpot = request.output_ids.size() <= 1
      ? 0.0
      : milliseconds(request.first_token_at, request.completed_at) /
            static_cast<double>(request.output_ids.size() - 1);
  std::cout << "{\"request_id\":\"" << jsonEscape(request.request_id)
            << "\",\"output_ids\":[";
  for (std::size_t index = 0; index < request.output_ids.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << request.output_ids[index];
  }
  std::cout << "],\"finish_reason\":\"" << finishReason(request.finish_reason)
            << "\",\"ttft_ms\":" << std::fixed << std::setprecision(3) << ttft
            << ",\"tpot_ms\":" << tpot << "}\n";
}

swiftinfer::ModelSpec loadModelSpec(const std::string& path) {
  const auto text = readFile(path);
  swiftinfer::ModelSpec spec;
  spec.num_layers = optionalIntegerField(text, "num_layers", spec.num_layers);
  spec.hidden_size = optionalIntegerField(text, "hidden_size", spec.hidden_size);
  spec.num_attention_heads = optionalIntegerField(
      text, "num_attention_heads", spec.num_attention_heads);
  spec.num_kv_heads = optionalIntegerField(text, "num_kv_heads", spec.num_kv_heads);
  spec.head_dim = optionalIntegerField(text, "head_dim", spec.head_dim);
  spec.vocab_size = optionalIntegerField(text, "vocab_size", spec.vocab_size);
  spec.max_context = optionalIntegerField(text, "max_context", spec.max_context);
  spec.eos_token_id = optionalIntegerField(text, "eos_token_id", spec.eos_token_id);
  if (spec.num_layers <= 0 || spec.num_kv_heads <= 0 || spec.head_dim <= 0 ||
      spec.vocab_size <= 0 || spec.max_context <= 1) {
    throw std::invalid_argument("invalid model spec");
  }
  return spec;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--validate-jsonl") {
      options.validate_jsonl = true;
    } else if (argument == "--generate-jsonl") {
      options.generate_jsonl = true;
    } else if (argument == "--prefill-engine" && index + 1 < argc) {
      options.prefill_engine = argv[++index];
    } else if (argument == "--decode-engine" && index + 1 < argc) {
      options.decode_engine = argv[++index];
    } else if (argument == "--model-spec" && index + 1 < argc) {
      options.model_spec = argv[++index];
    } else if (argument == "--max-active" && index + 1 < argc) {
      options.max_active = static_cast<std::size_t>(std::stoull(argv[++index]));
    } else if (argument == "--max-total" && index + 1 < argc) {
      options.max_total = static_cast<std::size_t>(std::stoull(argv[++index]));
    } else if (argument == "--flush-on-empty-line") {
      options.flush_on_empty_line = true;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + argument);
    }
  }
  if (options.validate_jsonl == options.generate_jsonl) {
    throw std::invalid_argument("select exactly one of --validate-jsonl or --generate-jsonl");
  }
  if (options.generate_jsonl && (options.prefill_engine.empty() ||
      options.decode_engine.empty() || options.model_spec.empty())) {
    throw std::invalid_argument("generation requires both engines and --model-spec");
  }
  if (options.max_active == 0 || options.max_active > 32 ||
      options.max_total < options.max_active || options.max_total > 64) {
    throw std::invalid_argument("require 1 <= max-active <= 32 and max-active <= max-total <= 64");
  }
  return options;
}

swiftinfer::RequestState parseRequest(const std::string& line,
                                   const swiftinfer::ModelSpec& spec) {
  swiftinfer::RequestState request;
  request.request_id = stringField(line, "request_id");
  request.input_ids = integerArray(line, "input_ids");
  request.max_new_tokens = integerField(line, "max_new_tokens");
  if (request.max_new_tokens == 0 ||
      request.input_ids.size() + request.max_new_tokens >
          static_cast<std::size_t>(spec.max_context)) {
    throw std::invalid_argument("context exceeds model limit");
  }
  for (const auto token : request.input_ids) {
    if (token >= spec.vocab_size) throw std::invalid_argument("token id exceeds vocabulary");
  }
  return request;
}

int validateJsonl(const swiftinfer::ModelSpec& spec) {
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto request = parseRequest(line, spec);
      std::cout << "{\"request_id\":\"" << jsonEscape(request.request_id)
                << "\",\"accepted\":true,\"input_length\":"
                << request.input_ids.size() << ",\"max_new_tokens\":"
                << request.max_new_tokens << "}\n";
    } catch (const std::exception& error) {
      writeError(bestEffortRequestId(line), error.what());
    }
  }
  return 0;
}

int generateJsonl(const Options& options, const swiftinfer::ModelSpec& spec) {
#ifndef SWIFTINFER_WITH_TENSORRT
  (void)options;
  (void)spec;
  throw std::runtime_error("this binary was built without TensorRT support");
#else
  constexpr std::size_t kBlockSize = 16;
  constexpr std::size_t kNumBlocks = 8192;
  swiftinfer::KVBlockManager blocks(kNumBlocks, kBlockSize);
  swiftinfer::Scheduler scheduler(options.max_active, options.max_total, blocks);
  swiftinfer::QwenTensorRTBackend backend(spec, blocks, options.prefill_engine,
                                       options.decode_engine, options.max_active);
  swiftinfer::RuntimeSession session(spec, scheduler, backend, options.max_active);
  std::vector<std::string> accepted_ids;
  std::unordered_set<std::string> seen_ids;

  const auto flush = [&]() -> bool {
    if (accepted_ids.empty()) return true;
    try {
      session.run();
    } catch (const std::exception& error) {
      const auto completed = scheduler.takeCompleted();
      std::unordered_set<std::string> completed_ids;
      for (const auto& request : completed) {
        completed_ids.insert(request->request_id);
        writeResponse(*request);
      }
      for (const auto& id : accepted_ids) {
        if (!completed_ids.count(id)) writeError(id, error.what());
      }
      accepted_ids.clear();
      std::cout.flush();
      return false;
    }
    for (const auto& request : scheduler.takeCompleted()) writeResponse(*request);
    accepted_ids.clear();
    std::cout.flush();
    return true;
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    if (options.flush_on_empty_line && trim(line).empty()) {
      if (!flush()) return 1;
      continue;
    }
    try {
      auto request = parseRequest(line, spec);
      const auto request_id = request.request_id;
      if (seen_ids.count(request_id)) {
        throw std::invalid_argument("duplicate request id");
      }
      if (accepted_ids.size() == options.max_total && !flush()) return 1;
      scheduler.submit(std::move(request));
      accepted_ids.push_back(request_id);
      seen_ids.insert(request_id);
    } catch (const std::exception& error) {
      writeError(bestEffortRequestId(line), error.what());
    }
  }
  return flush() ? 0 : 1;
#endif
}

void usage() {
  std::cerr
      << "Usage:\n"
      << "  swiftinfer_cli --validate-jsonl [--model-spec FILE]\n"
      << "  swiftinfer_cli --prefill-engine FILE --decode-engine FILE "
         "--model-spec FILE --generate-jsonl [--max-active N] "
         "[--max-total N] [--flush-on-empty-line]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parseOptions(argc, argv);
    const auto spec = options.model_spec.empty()
        ? swiftinfer::ModelSpec{}
        : loadModelSpec(options.model_spec);
    return options.validate_jsonl ? validateJsonl(spec)
                                  : generateJsonl(options, spec);
  } catch (const std::exception& error) {
    usage();
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
