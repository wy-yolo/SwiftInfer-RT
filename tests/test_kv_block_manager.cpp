#include <iostream>
#include <stdexcept>

#include "swiftinfer/kv_block_manager.h"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  swiftinfer::KVBlockManager blocks(4, 16);
  require(blocks.freeBlockCount() == 4, "initial free count");
  blocks.ensureCapacity("a", 1);
  require(blocks.requestBlockCount("a") == 1, "one block");
  blocks.ensureCapacity("a", 16);
  require(blocks.requestBlockCount("a") == 1, "boundary block");
  blocks.ensureCapacity("a", 17);
  require(blocks.requestBlockCount("a") == 2, "cross boundary");
  blocks.ensureCapacity("b", 32);
  require(blocks.freeBlockCount() == 0, "pool should be full");
  bool exhausted = false;
  try { blocks.ensureCapacity("c", 1); } catch (const std::runtime_error&) { exhausted = true; }
  require(exhausted && !blocks.contains("c"), "exhaustion rollback");
  const auto old = blocks.blockTable("a");
  blocks.release("a");
  require(blocks.freeBlockCount() == 2, "release count");
  blocks.ensureCapacity("c", 17);
  const auto reused = blocks.blockTable("c");
  require(reused.size() == 2 && old.size() == 2, "reuse count");
  bool double_release = false;
  try { blocks.release("a"); } catch (const std::out_of_range&) { double_release = true; }
  require(double_release, "double release detection");
  blocks.release("b");
  blocks.release("c");
  require(blocks.freeBlockCount() == 4, "final free count");
  std::cout << "kv_block_manager: PASS\n";
}
