#include <iostream>
#include <stdexcept>

#include "swiftinfer/scheduler.h"

swiftinfer::RequestState makeRequest(const std::string& id, std::size_t max_new = 2) {
  swiftinfer::RequestState request;
  request.request_id = id;
  request.input_ids = {1, 2, 3};
  request.max_new_tokens = max_new;
  return request;
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  swiftinfer::KVBlockManager blocks(16, 16);
  swiftinfer::Scheduler scheduler(2, 4, blocks);
  scheduler.submit(makeRequest("a"));
  scheduler.submit(makeRequest("b", 4));
  scheduler.submit(makeRequest("c"));
  require(scheduler.activeCount() == 2 && scheduler.waitingCount() == 1, "initial admission");
  scheduler.appendToken("a", 7, 99);
  scheduler.appendToken("a", 8, 99);
  require(scheduler.activeCount() == 2 && scheduler.waitingCount() == 0, "continuous admission");
  auto batch = scheduler.nextBatch();
  require(batch[0]->request_id == "b" && batch[1]->request_id == "c", "FCFS order");
  scheduler.cancel("c");
  scheduler.appendToken("b", 99, 99);
  require(scheduler.empty(), "scheduler should be empty");
  require(blocks.freeBlockCount() == 16, "all blocks returned");
  auto completed = scheduler.takeCompleted();
  require(completed.size() == 3, "completed request accounting");
  std::cout << "scheduler: PASS\n";
}
