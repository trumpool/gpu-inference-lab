# gpu-inference-lab

一个按周推进的学习仓库：从手写 CUDA 和 Triton kernel 开始，最后做到 LLM 推理系统（SGLang、vLLM）。计划见 [PLAN.md](PLAN.md)。

## 硬件

- 主力：RTX 5070 Ti（Blackwell，sm_120，16 GB），Windows + WSL2
- 副机：RTX 3070 Laptop（Ampere，sm_86，8 GB）

## 目录

| 目录 | 内容 | 对应计划 |
|---|---|---|
| `cuda/` | 手写 CUDA kernel：基础、matmul、reduction、softmax / norm | 第 0–6 周 |
| `triton/` | Triton kernel：softmax、Llama block 的融合 kernel、FlashAttention | 第 7–11 周 |
| `serving/` | SGLang 和 vLLM 的部署与实验 | 第 12–13 周 |
| `mini-engine/` | 迷你推理引擎：paged KV cache、continuous batching、prefix caching | 第 14–16 周 |
| `notes/` | 笔记和实验记录 | 全程 |
| `scripts/` | 工具脚本，比如环境检查 | — |
| `docs/` | 环境配置说明 | 第 0 周 |

## 第 0 周：开始

照 [docs/setup-windows-wsl2.md](docs/setup-windows-wsl2.md) 把环境配好，然后运行：

```bash
python scripts/check_env.py
cd cuda/00_hello && nvcc -arch=native hello.cu -o hello && ./hello
```

把输出记进 [notes/env.md](notes/env.md)。

## 规矩

- 正式的 benchmark 都在 5070 Ti 上测，并记下驱动、CUDA 和 PyTorch 的版本。
- 每个 kernel 都要有正确性测试：和 PyTorch 的结果对比，并写明容差。
