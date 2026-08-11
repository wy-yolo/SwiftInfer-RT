#pragma once

#include <cstddef>

namespace swiftinfer {

struct ModelSpec {
  int num_layers{24};
  int hidden_size{896};
  int num_attention_heads{14};
  int num_kv_heads{2};
  int head_dim{64};
  int vocab_size{151936};
  int max_context{4096};
  int eos_token_id{151645};

  std::size_t kvElementsPerToken() const {
    return static_cast<std::size_t>(num_layers) * 2 * num_kv_heads * head_dim;
  }
};

}  // namespace swiftinfer

