#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "swiftinfer/scheduler.h"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  constexpr std::size_t block_size = 16;
  constexpr std::size_t blocks = 8192;
  swiftinfer::KVBlockManager pool(blocks, block_size);
  swiftinfer::Scheduler scheduler(32, 64, pool);
  for (int i = 0; i < 64; ++i) {
    swiftinfer::RequestState request;
    request.request_id = "request-" + std::to_string(i);
    const std::size_t prompt_length =
        static_cast<std::size_t>((i % 4 == 0) ? 3968 : (i % 4 == 1) ? 2048 :
                                 (i % 4 == 2) ? 1024 : 256);
    request.input_ids.assign(prompt_length, 42);
    request.max_new_tokens = 32;
    scheduler.submit(std::move(request));
  }
  require(scheduler.activeCount() == 32, "active capacity");
  require(scheduler.waitingCount() == 32, "waiting capacity");
  std::size_t completed_count = 0;
  while (!scheduler.empty()) {
    auto batch = scheduler.nextBatch(32);
    require(!batch.empty() && batch.size() <= 32, "batch bounds");
    for (auto& request : batch) {
      scheduler.appendToken(request->request_id, 7, 151645);
    }
    auto completed = scheduler.takeCompleted();
    completed_count += completed.size();
  }
  completed_count += scheduler.takeCompleted().size();
  require(completed_count == 64, "all requests completed");
  require(pool.freeBlockCount() == blocks, "stress test leaked blocks");
  std::cout << "scheduler_stress: PASS requests=" << completed_count << '\n';
}

