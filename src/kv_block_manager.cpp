#include "minillm/kv_block_manager.h"

#include <stdexcept>

namespace minillm {

KVBlockManager::KVBlockManager(std::size_t num_blocks, std::size_t block_size)
    : num_blocks_(num_blocks), block_size_(block_size) {
  if (num_blocks == 0 || block_size == 0) {
    throw std::invalid_argument("KV block count and block size must be positive");
  }
  free_blocks_.reserve(num_blocks);
  for (std::size_t i = 0; i < num_blocks; ++i) {
    free_blocks_.push_back(static_cast<std::int32_t>(num_blocks - 1 - i));
  }
}

std::size_t KVBlockManager::requiredBlocks(std::size_t token_count) const {
  return token_count == 0 ? 0 : (token_count + block_size_ - 1) / block_size_;
}

void KVBlockManager::ensureCapacity(const std::string& request_id,
                                    std::size_t token_count) {
  if (request_id.empty()) throw std::invalid_argument("request id is empty");
  std::lock_guard<std::mutex> guard(mutex_);
  auto& table = tables_[request_id];
  const auto required = requiredBlocks(token_count);
  if (required <= table.size()) return;
  const auto additional = required - table.size();
  if (additional > free_blocks_.size()) {
    if (table.empty()) tables_.erase(request_id);
    throw std::runtime_error("KV block pool exhausted");
  }
  for (std::size_t i = 0; i < additional; ++i) {
    table.push_back(free_blocks_.back());
    free_blocks_.pop_back();
  }
}

void KVBlockManager::release(const std::string& request_id) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = tables_.find(request_id);
  if (it == tables_.end()) throw std::out_of_range("unknown or already released request");
  for (auto block : it->second) free_blocks_.push_back(block);
  tables_.erase(it);
}

bool KVBlockManager::contains(const std::string& request_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return tables_.find(request_id) != tables_.end();
}

std::vector<std::int32_t> KVBlockManager::blockTable(
    const std::string& request_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = tables_.find(request_id);
  if (it == tables_.end()) throw std::out_of_range("unknown request");
  return it->second;
}

std::int32_t KVBlockManager::physicalBlock(const std::string& request_id,
                                           std::size_t logical_block) const {
  auto table = blockTable(request_id);
  if (logical_block >= table.size()) throw std::out_of_range("logical block out of range");
  return table[logical_block];
}

std::size_t KVBlockManager::freeBlockCount() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return free_blocks_.size();
}

std::size_t KVBlockManager::requestBlockCount(const std::string& request_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = tables_.find(request_id);
  return it == tables_.end() ? 0 : it->second.size();
}

}  // namespace minillm

