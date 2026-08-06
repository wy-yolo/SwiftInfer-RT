#include <iostream>
#include <cstdint>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/model_spec.h"

namespace {

std::string stringField(const std::string& line, const std::string& name) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) throw std::invalid_argument("missing " + name);
  return match[1].str();
}

std::size_t integerField(const std::string& line, const std::string& name) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) throw std::invalid_argument("missing " + name);
  return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::vector<std::int32_t> integerArray(const std::string& line,
                                       const std::string& name) {
  std::regex pattern("\\\"" + name + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(line, match, pattern)) throw std::invalid_argument("missing " + name);
  std::vector<std::int32_t> values;
  std::stringstream stream(match[1].str());
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) values.push_back(static_cast<std::int32_t>(std::stol(item)));
  }
  if (values.empty()) throw std::invalid_argument("input_ids is empty");
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "--validate-jsonl") {
    std::cerr << "Usage: minillm_cli --validate-jsonl\n";
    std::cerr << "The TensorRT generation adapter is configured by the Python CLI.\n";
    return 2;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto id = stringField(line, "request_id");
      const auto ids = integerArray(line, "input_ids");
      const auto max_new = integerField(line, "max_new_tokens");
      if (ids.size() + max_new > 4096) throw std::invalid_argument("context exceeds 4096");
      std::cout << "{\"request_id\":\"" << id
                << "\",\"accepted\":true,\"input_length\":" << ids.size()
                << ",\"max_new_tokens\":" << max_new << "}\n";
    } catch (const std::exception& error) {
      std::cout << "{\"accepted\":false,\"error\":\"" << error.what() << "\"}\n";
    }
  }
  return 0;
}
