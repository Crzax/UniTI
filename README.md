# UniTI — 训推一体深度学习框架

**UniTI**（Unified Training & Inference）是一个仿照 Needle & Pytorch & vllm 构建的训推一体深度学习框架。  
核心目标：**用同一套代码完成从 MNIST 分类到 大语言模型推理的全流程**。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔧 **三种计算后端** | 自研 C++ CPU 后端（OpenBLAS 加速）、NumPy 封装后端（零编译即用）、自研 CUDA GPU 后端 |
| 🔄 **训推一体架构** | 训练和推理共享同一套算子与模型代码，通过 `no_grad()` / `detach` 机制无缝切换 |
| 🧠 **多种模型支持** | 两层全连接网络、ResNet9、RNN / LSTM / Transformer 语言模型、Qwen2 大语言模型 |
| 📄 **Paged Attention** | vLLM 风格分页式 KV Cache，按需分配物理块，避免连续预分配的内存浪费 |
| ⚡ **推理优化** | `no_grad()` 跳过计算图构建、`last_only` 仅计算末位 logits、KV Cache 增量解码 |
| 📦 **零外部依赖推理** | 自研 safetensors 加载器 + BPE Tokenizer，推理仅需 `numpy`，无需 torch / transformers |
| 🔤 **流式 UTF-8 解码** | 基于 `codecs.IncrementalDecoder` 的字节级增量解码，正确流式输出 emoji 等多字节字符 |

---

## 目录结构

```
UniTI/
├── python/uniti/              # 框架核心 Python 库
│   ├── __init__.py                # 包入口，延迟加载子模块
│   ├── autograd.py                # 自动微分引擎：Tensor、计算图、反向传播、no_grad
│   ├── backend_selection.py       # 后端选择逻辑（通过 UNITI_BACKEND 环境变量）
│   ├── backend_numpy.py           # NumPy 后端设备抽象
│   ├── backend_ndarray/           # 自研 NDArray 后端（统一 CPU/CUDA 接口）
│   │   ├── ndarray.py                 # NDArray 数据结构，统一的设备 API
│   │   └── ndarray_backend_numpy.py   # cpu_numpy 后端的纯 Python 实现
│   ├── ops/                       # 算子定义（前向 + 反向梯度）
│   │   ├── ops_mathematic.py          # 数学运算（加减乘除、矩阵乘、卷积、ReLU、softmax 等）
│   │   ├── ops_logarithmic.py         # 对数相关（log、exp、logsumexp 等）
│   │   └── ops_tuple.py              # TensorTuple 操作（split、stack 等）
│   ├── nn/                        # 神经网络模块
│   │   ├── nn_basic.py               # 基础：Module、Linear、ReLU、Dropout、BatchNorm、LayerNorm
│   │   ├── nn_conv.py                # 卷积：Conv、ConvBN
│   │   ├── nn_sequence.py            # 序列模型：RNN、LSTM、Embedding
│   │   ├── nn_transformer.py         # Transformer：MultiHeadAttention、TransformerLayer
│   │   ├── nn_qwen2.py              # Qwen2：RMSNorm、RoPE、GQA、SwiGLU MLP、完整 LM
│   │   └── paged_attention.py        # Paged Attention：vLLM 风格分页 KV Cache
│   ├── optim.py                   # 优化器：SGD（含动量）、Adam
│   ├── safetensors_loader.py      # 自研 safetensors 加载器（纯 numpy，零外部依赖）
│   ├── tokenizer.py               # 自研 BPE Tokenizer（支持 Qwen2/DeepSeek，含流式字节解码）
│   ├── init/                      # 参数初始化：Xavier、Kaiming、正态、均匀等
│   └── data/                      # 数据加载
│       ├── data_basic.py             # Dataset、DataLoader 基类
│       ├── data_transforms.py        # 数据增强变换
│       └── datasets/                # 内置数据集（MNIST、CIFAR-10、PTB）
├── src/                           # C++/CUDA 底层后端
│   ├── ndarray_backend_cpu.cc        # 自研 C++ CPU 后端（支持 OpenBLAS）
│   └── ndarray_backend_cuda.cu       # 自研 CUDA GPU 后端
├── apps/                          # 推理应用入口
│   ├── deepseek_inference.py         # DeepSeek-R1-Distill-Qwen-1.5B 推理脚本
│   ├── models.py                     # 训练用模型定义（ResNet9、LanguageModel）
│   └── simple_ml.py                  # 训练辅助函数（MNIST/CIFAR-10/PTB 训练循环）
├── tests/                         # 测试、训练脚本与性能基准
├── data/                          # 训练数据存放目录
├── CMakeLists.txt                 # CMake 构建配置
└── Makefile                       # 便捷构建入口
```

---

## 三种计算后端

UniTI 通过统一的 NDArray API 支持三种后端，所有模型代码和算子对后端完全透明：

| 后端 | 标识 | 实现 | 特点 |
|------|------|------|------|
| **cpu** | `uniti.cpu()` | 自研 C++ (`ndarray_backend_cpu.cc`) | OpenBLAS 加速矩阵乘法，性能最优的 CPU 方案 |
| **cpu_numpy** | `uniti.cpu_numpy()` | 纯 Python/NumPy | 零编译，开箱即用，适合快速验证 |
| **cuda** | `uniti.cuda()` | 自研 CUDA (`ndarray_backend_cuda.cu`) | GPU 并行计算，大模型推理必备 |

通过 `--device` 参数切换：

```bash
--device cpu          # 自研 C++ CPU 后端
--device cpu_numpy    # NumPy 后端（默认）
--device cuda         # 自研 CUDA GPU 后端
```

---

## 训推一体架构

UniTI 的核心设计理念：**训练和推理共享同一套代码路径**。

### 推理相对训练的优化手段

| 优化手段 | 说明 |
|----------|------|
| **`no_grad()` 上下文管理器** | 禁用梯度追踪，跳过计算图构建，大幅减少内存和计算开销 |
| **计算图 `detach`** | `no_grad()` 下自动 detach，不保留中间节点 |
| **`last_only` 优化** | 解码阶段仅对最后一个 token 计算 logits |
| **KV Cache 增量解码** | Prefill 一次处理 prompt，Decode 每步仅处理 1 个新 token |
| **`model.eval()`** | 关闭 Dropout/BatchNorm 的训练行为 |

```python
# 训练模式（构建计算图，支持反向传播）
model.train()
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()

# 推理模式（跳过计算图构建，节省内存和算力）
model.eval()
with no_grad():
    logits = model(x, start_pos=pos, last_only=True)
```

---

## Paged Attention

实现 [vLLM](https://arxiv.org/abs/2309.06180) 风格的分页式 KV Cache 管理：

- **物理块池**：KV Cache 划分为固定大小的 Page，全局统一管理
- **页表映射**：每个序列维护逻辑块到物理块的映射，支持非连续内存布局
- **按需分配**：动态分配新块，无需预估 `max_seq_len`
- **跨序列共享**：多序列共用物理块池，高效复用
- **设备统一**：三种后端使用完全相同的代码路径

```python
# Paged Attention
model.init_paged_cache(block_size=16, max_num_blocks=256)

# 传统连续 KV Cache
model.init_cache(batch_size=1, max_cache_len=2048)
```

---

## 零外部依赖推理

UniTI 自研实现了推理所需的全部组件，推理 Qwen2/DeepSeek 模型**仅需 `numpy`**：

| 功能 | 原外部依赖 | UniTI 自研实现 |
|------|-----------|---------------|
| 加载 `.safetensors` 权重 | `safetensors` + `torch` | `uniti.safetensors_loader` — 纯 `struct` + `json` + `numpy` 解析 |
| BPE Tokenizer | `transformers` | `uniti.tokenizer.UniTITokenizer` — 直接读取 `tokenizer.json` |
| BF16 → FP32 转换 | `torch.to(float32)` | 位操作 `uint16 << 16 → view(float32)` |
| 流式 UTF-8 解码 | - | `token_to_bytes()` + `codecs.IncrementalDecoder`，正确处理 emoji 等多字节字符 |

### 流式输出原理

Byte-level BPE 会将 emoji（如 🎉 = 4字节 `F0 9F 8E 89`）拆分为多个 token。逐 token 输出时，单个 token 只含部分字节，直接 decode 会出现 `�` 乱码。

UniTI 的解决方案：
1. `tokenizer.token_to_bytes(token_id)` — 将 token 转为原始字节（不做 UTF-8 decode）
2. `codecs.getincrementaldecoder('utf-8')` — 增量 decoder 自动缓冲不完整字节序列，只输出完整字符

---

## 支持的模型

| 模型 | 数据集 | 任务 | 网络结构 |
|------|--------|------|----------|
| 两层全连接网络 | MNIST | 手写数字分类 | Linear → ReLU → Linear |
| ResNet9 | CIFAR-10 | 图像分类 | ConvBN × 6 + Residual × 2 + Linear × 2 |
| RNN 语言模型 | PTB | 语言建模 | Embedding → RNN → Linear |
| LSTM 语言模型 | PTB | 语言建模 | Embedding → LSTM → Linear |
| Transformer 语言模型 | PTB | 语言建模 | Embedding → Transformer → Linear |
| Qwen2 (DeepSeek-R1-Distill-1.5B) | - | LLM 推理 | RMSNorm + GQA + RoPE + SwiGLU MLP |

其他模型可以基于提供的 Module 自行构建实现

---

## 快速开始

### 环境准备

#### 编译 C++ / CUDA 后端（可选，`cpu_numpy` 后端无需编译）

```bash
# 安装编译依赖
pip install pybind11 numpy

# 一键编译（自动检测 CUDA 和 OpenBLAS）
make

# 或手动：
mkdir build && cd build && cmake .. && make
```

编译产物自动输出到 `python/uniti/backend_ndarray/` 目录。

#### Python 依赖

```
numpy              # 数值计算基础（唯一必需依赖）
pybind11           # C++/Python 绑定（仅编译 cpu/cuda 后端需要）
regex              # （可选）更好的 Unicode 正则支持，tokenizer 会自动降级到 stdlib re
```

---

## 训练命令

训练脚本位于 `tests/` 目录，数据集会自动下载。

### MNIST 手写数字分类

```bash
python tests/test_train_mnist.py                              # NumPy 后端（默认）
python tests/test_train_mnist.py --device cpu                 # C++ CPU 后端
python tests/test_train_mnist.py --device cuda                # CUDA GPU 后端
python tests/test_train_mnist.py --hidden 128 --epochs 10 --lr 0.1 --batch 100
```

### CIFAR-10 图像分类（ResNet9）

```bash
python tests/test_train_cifar10.py                            # NumPy 后端
python tests/test_train_cifar10.py --device cuda --epochs 10 --lr 0.001
```

### PTB 语言模型（RNN / LSTM / Transformer）

```bash
python tests/test_train_ptb.py                                # 默认 LSTM
python tests/test_train_ptb.py --model rnn                    # RNN
python tests/test_train_ptb.py --model transformer            # Transformer
python tests/test_train_ptb.py --device cuda --model lstm --epochs 3 \
    --embedding_size 128 --hidden_size 128 --seq_len 40 --lr 4.0
```

---

## 推理命令

### DeepSeek-R1-Distill-Qwen-1.5B

```bash
# 基本推理（cpu_numpy 后端，连续 KV Cache）
python apps/deepseek_inference.py \
    --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "Hello, how are you?"

# CUDA 后端 + Paged Attention
python apps/deepseek_inference.py \
    --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "请解释什么是深度学习" \
    --device cuda \
    --paged --block_size 16 --max_blocks 256

# 贪心解码
python apps/deepseek_inference.py \
    --model_path /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt "What is machine learning?" \
    --device cpu --greedy
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | - | HuggingFace 格式模型路径 |
| `--prompt` | "Hello, how are you?" | 输入文本 |
| `--device` | cpu_numpy | 计算后端：`cpu` / `cpu_numpy` / `cuda` |
| `--max_new_tokens` | 1000 | 最大生成 token 数 |
| `--temperature` | 0.6 | 采样温度 |
| `--top_p` | 0.95 | Top-p 采样阈值 |
| `--greedy` | False | 贪心解码（不采样） |
| `--paged` | False | 启用 Paged Attention |
| `--block_size` | 16 | Page 大小（tokens/page） |
| `--max_blocks` | 256 | 最大物理块数 |
| `--no_chat` | False | 跳过 Chat Template |

---

## 测试与基准

```bash
# 功能测试
python tests/test_paged_attention.py          # Paged Attention 单元测试
python tests/test_paged_vs_contiguous.py      # Paged vs Contiguous KV Cache 对比
python tests/test_full_cpu.py                 # CPU 后端完整功能测试

# 性能基准
python tests/bench_cuda.py                    # CUDA 综合基准
python tests/bench_cuda_decode.py             # CUDA 解码性能
python tests/bench_ops.py                     # 算子性能基准
python tests/bench_paged_vs_contiguous.py     # Paged vs Contiguous 性能对比
```

---

## 技术架构

```
┌──────────────────────────────────────────────────┐
│                 用户代码 / 应用层                    │
│         (tests/test_train_*.py, apps/*.py)         │
├──────────────────────────────────────────────────┤
│                    nn 模块层                        │
│  Linear, Conv, RNN, LSTM, Transformer, Qwen2      │
│  PagedKVCacheManager, BatchNorm, LayerNorm         │
├──────────────────────────────────────────────────┤
│              autograd 自动微分引擎                   │
│        Tensor, Op, 计算图构建, 反向传播              │
│        no_grad(), detach(), backward()             │
├──────────────────────────────────────────────────┤
│                    ops 算子层                       │
│   EWiseAdd, MatMul, Conv, ReLU, Softmax, ...       │
│         每个算子定义 compute() + gradient()          │
├──────────────────────────────────────────────────┤
│              统一 NDArray API 层                    │
│         device.full(), .compact(), 切片赋值         │
│         .numpy(), .reshape(), .broadcast_to()      │
├──────────┬───────────────┬───────────────────────┤
│   cpu    │  cpu_numpy    │        cuda           │
│  (C++)   │  (Python)     │       (CUDA)          │
│ OpenBLAS │   NumPy       │  自研 GPU Kernel       │
└──────────┴───────────────┴───────────────────────┘
```

```
推理数据流（流式输出）:

Token ID ──→ tokenizer.token_to_bytes() ──→ 原始字节
                                              │
                        codecs.IncrementalDecoder('utf-8')
                                              │
                                         完整字符 ──→ print()
```
