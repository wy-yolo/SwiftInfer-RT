#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace minillm {

class KVBlockManager {
 public:
  KVBlockManager(std::size_t num_blocks, std::size_t block_size);

  void ensureCapacity(const std::string& request_id, std::size_t token_count);
  void release(const std::string& request_id);
  bool contains(const std::string& request_id) const;
  std::vector<std::int32_t> blockTable(const std::string& request_id) const;
  std::int32_t physicalBlock(const std::string& request_id,
                             std::size_t logical_block) const;
  std::size_t freeBlockCount() const;
  std::size_t requestBlockCount(const std::string& request_id) const;
  std::size_t blockSize() const noexcept { return block_size_; }
  std::size_t capacityTokens() const noexcept { return num_blocks_ * block_size_; }

 private:
  std::size_t requiredBlocks(std::size_t token_count) const;

  const std::size_t num_blocks_;
  const std::size_t block_size_;
  mutable std::mutex mutex_;
  std::vector<std::int32_t> free_blocks_;
  std::unordered_map<std::string, std::vector<std::int32_t>> tables_;
};

}  // namespace minillm

