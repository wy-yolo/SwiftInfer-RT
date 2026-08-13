# MiniLLM-RT

MiniLLM-RT 是一个面向学习与复现实验的 C++17/TensorRT 大语言模型推理运行时。项目以
Qwen2.5-0.5B-Instruct 为验证模型，将推理图拆分为独立的 Prefill 与 Decode Engine，
实现请求级分块 KV Cache、CUDA Gather/Scatter、贪心采样，以及 FCFS Continuous
Batching。

项目已经在 RTX 5090 与 RTX 5060 Laptop GPU 上完成端到端验证。这里的实现重点是把
模型导出、数值验证、TensorRT Engine、C++ Runtime、内存检查和性能实验串成一条可复现
的工程链路，而不是替代成熟的生产级推理框架。

## 核心能力

- **双 Engine 推理**：Prefill 与 Decode 分别导出为 ONNX 并构建 FP16 TensorRT Engine。
- **分块 KV Cache**：以 16 token 为一个逻辑块，GPU 端预分配物理块池并支持分配、增长、
  释放与复用。
- **CUDA 数据搬运**：通过 Gather/Scatter 在分块 KV Cache 与 TensorRT 连续张量之间转换。
- **持续动态批处理**：Prefill 按请求顺序执行，Decode 支持请求完成后立即补位；RTX 5090
  最多支持 32 个活跃请求和 64 个排队请求。
- **完整正确性链路**：覆盖 Hugging Face → ONNX Runtime → TensorRT → C++ Runtime，
  并使用 Compute Sanitizer 检查 CUDA 内存和并发问题。
- **跨 GPU 复现**：相同源码与 ONNX 在 RTX 5090 和 RTX 5060 上分别重建专属 Engine。

## 功能边界

- 模型：Qwen2.5-0.5B-Instruct。
- 精度：FP16 生产路径，FP32 数值参考。
- 解码：Greedy Decoding，单 GPU。
- RTX 5090：最大上下文 4096，Decode batch 1–32。
- KV Cache：8192 个物理块，每块 16 token。
- 当前实现通过 Gather 将分块 KV 重组为连续 TensorRT 输入，**不是** PagedAttention 插件。
- 暂不包含量化、多 GPU、批量 Prefill、采样算法、HTTP 服务和纯 C++ Tokenizer。

## 环境要求

已验证的主要环境如下：

- Python 3.11 与 Conda；
- PyTorch 2.10（CUDA 12.8）；
- CUDA Toolkit 12.9；
- TensorRT 10.10.0.31；
- CMake、Ninja、GCC 13；
- NVIDIA Blackwell GPU，CUDA 架构 `sm_120`。

激活已有实验环境并执行自检：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate minillm-rt
python scripts/verify_environment.py
```

CUDA 编译器必须来自 Conda 环境中的 CUDA 12.9，不应使用系统的
`/usr/bin/nvcc`（本实验机上为 CUDA 12.2）。TensorRT C++ 构建还需要设置
`TENSORRT_ROOT`。

`configs/` 中的 Conda 锁文件与环境快照来自实际实验机器，其中的用户名和绝对路径仅用于
记录实验来源。复现时请将这些路径替换为自己的 Conda、项目和 TensorRT SDK 路径。

## 编译与测试

```bash
cmake -S . -B build-gpu-release2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DTENSORRT_ROOT="$PWD/artifacts/tensorrt-sdk"

cmake --build build-gpu-release2 -j
ctest --test-dir build-gpu-release2 --output-on-failure
PYTHONPATH=python python -m pytest -q
```

必须显式指定 CUDA Host Compiler。否则 CUDA 12.9 可能选择系统 GCC 14，并在本实验环境中
混用不兼容的系统与 Conda 头文件。

Compute Sanitizer 只应在 GPU 空闲时运行：

```bash
# 检查独立 CUDA Kernel
scripts/run_sanitizer.sh build-gpu-debug2

# 检查完整 TensorRT Runtime
scripts/run_runtime_sanitizer.sh build-gpu-debug2
```

## 模型导出与 Engine 构建

模型下载需要访问 Hugging Face。如果网络环境需要代理，可仅为下载命令临时设置代理；下方
地址只是示例，请按本机配置替换，不要写入长期 Shell 配置。

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
  python python/download_model.py

# 在 GPU 上分别导出 FP16 生产图与 FP32 参考图
python python/export_qwen.py --precision fp16 --device cuda:0
python python/rewrite_rope_lookup.py
python python/export_qwen.py --precision fp32 --device cuda:0

# ONNX 数值验证
python python/validate_onnx.py
python python/validate_onnx_matrix.py

# 完整生产语料门禁，需要 GPU 空闲且至少有 28 GiB 空闲显存
python python/validate_corpus.py --resume --require-exclusive --min-free-gb 28

# Prefill 与 Decode 必须在独立进程中依次构建
python python/build_engine.py --kind prefill --min-free-gb 28 \
  --fp32-gemm-accumulation
python python/build_engine.py --kind decode --min-free-gb 28
```

RTX 5090 Engine Profile：

- Prefill：B1，`S=1/256/4096`；
- Decode：`B1/H1`、`B8/H256`、`B32/H4095`。

RTX 5060 Engine Profile：

- Prefill：B1，`S=1/256/2048`；
- Decode：`B1/H1`、`B4/H256`、`B8/H2047`。

TensorRT `.plan` 与 GPU 架构相关，5090 Engine 不能直接复制到 5060 使用。项目会在构建前
检查 GPU 占用情况，并通过临时文件和原子替换避免留下残缺 Engine。

## 正确性验证

```bash
python python/generate_validation_corpus.py

PYTHONPATH=python python python/validate_runtime_corpus.py \
  --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
  --decode-engine artifacts/engines/rtx5090/fp16/decode.plan

PYTHONPATH=python python python/adjudicate_runtime_tokens.py
```

项目采用双轨数值策略：

- FP32：Hugging Face → ONNX Runtime CPU/CUDA 的逐张量门槛为
  `atol=rtol=1e-3`；
- FP16：要求 logits 余弦相似度不低于 0.999、NRMSE 不高于 0.02、Top-5
  重合率不低于 95%；边界分歧使用 FP32 参考结果裁决；
- 最终生成：Hugging Face、ONNX Runtime、TensorRT 与 C++ Runtime 的 Greedy Token
  序列必须一致。

完整验证覆盖 182 组配置、累计 1918 个请求。FP32 HF → ORT 最大 logits 误差约为
`1.27e-4`；1918 个请求均成功完成 C++ 推理，Python TensorRT 与 C++ 输出逐 Token
一致。详细策略与 Engine 哈希见 [`docs/numerical_gate.md`](docs/numerical_gate.md)。

## 性能实验

```bash
# 截图规格基线与完整 4K / Active Batch / Request Count 矩阵
PYTHONPATH=python python python/benchmark_runtime.py --mode all

python benchmarks/analyze_results.py \
  results/benchmarks/rtx5090/raw.jsonl \
  --output results/benchmarks/rtx5090/summary.json
```

正式实验要求 GPU 无其他用户计算进程且至少有 28 GiB 空闲显存。每组先预热 20 次，再进行
至少 5 次正式测量，记录 TTFT、TPOT p50/p95、聚合吞吐、单请求吞吐、峰值显存、GPU
利用率、功耗与温度。

### RTX 5090

截图规格为 Prompt 256、Output 32、共 16 个请求：

- 无 KV 重算基线 TPOT：5.375 ms；
- KV B1 TPOT：3.035 ms，降低 **43.54%**；
- KV B1 吞吐：330.6 tok/s；
- 动态 B8 吞吐：2034.9 tok/s，提升 **515.55%**。

扩展矩阵完成 44 组配置、220 次正式测量。峰值吞吐为 **4464.3 tok/s**
（Prompt 256、Active B32、32 个请求），峰值显存为 7875 MiB，最高温度为 78°C。
Compute Sanitizer 的 memcheck、initcheck、racecheck 与 synccheck 均通过，报告 0 错误、
0 泄漏和 0 hazard。

RTX 5090 Engine SHA256：

- Prefill：`053dd706a0db65f3331a8d97da258a88315019b65db52ba163049f81f65c2eaa`；
- Decode：`50b674cbbda37b9cfd5d59a8495c2786b69c56eee25ece9d81f73eb0efad585e`。

### RTX 5060 Laptop

RTX 5060 使用相同源码和 FP16 ONNX 重新构建 Engine，没有复用 RTX 5090 的 `.plan`。
完整验证结果如下：

- 6 个 HF → ORT → TRT Profile 端点全部通过；
- 48 个请求的可移植性语料逐 Token 一致，包含 `2016 + 32 = 2048` 上下文边界；
- 性能矩阵完成 18 组配置、54 次正式测量；
- 峰值吞吐为 **707.7 tok/s**（Prompt 256、Active B8、16 个请求）；
- 峰值显存为 5455 MiB，最高温度为 86°C。

RTX 5060 Engine SHA256：

- Prefill：`c38189ed78623d07c8941c72ced5b1947de4305a4e199184dbdaecdd6582c71b`；
- Decode：`627a0d20aadcaf6cc68193fbf0c26d4d8c810da9a0947dc5bfef2af7f04290c8`。

笔记本长时间运行正式实验时应加强散热。

## 本地产物

以下内容体积较大、与具体机器相关或可由脚本重新生成，因此已通过 `.gitignore` 排除，
**不会上传至 GitHub**：

- Hugging Face 模型权重；
- FP32/FP16 ONNX 与 External Data；
- TensorRT `.plan`；
- `artifacts/` 下的 TensorRT SDK 与元数据；
- `results/` 下的验证报告、Sanitizer 日志和性能结果；
- 本地 `build*` 构建目录。

复现实验时需要按前述流程在本地下载模型、导出 ONNX，并针对当前 GPU 重新构建 Engine。

## 已知限制

- 当前 TensorRT 后端只适配本项目导出的 Qwen2.5-0.5B 图与固定张量接口；
- Prefill 仍按请求串行执行；
- Decode 会按历史长度分组并 Gather 为连续 KV，尚未实现真正的 PagedAttention；
- 只支持 Greedy Decoding，没有 Top-k、Top-p 或温度采样；
- CLI 使用 JSONL Token 接口，Tokenizer 由 Python 侧负责；
- 未实现在线服务、多 GPU 和量化。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源，版权声明见 [NOTICE](NOTICE)。
