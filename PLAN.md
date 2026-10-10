# GPU / LLM 推理学习计划（2026-10-08 → 2027-03-07）

**目标**：春招前达到 new grad 推理 / ML Infra 岗位的面试水平，具体是三件事：

- 能写基础的 CUDA 和 Triton kernel，并讲清每一步优化的理由；
- 懂 SGLang 和 vLLM 的核心机制；
- 手上有 2–3 个带真实测量数字的项目。

**两条原则**

- 有面试的那一周，面试准备优先，学习计划让路。
- 项目跑出真实数字以后再写进简历。

## 硬件分工

| 机器 | 用途 |
|---|---|
| 台式机 RTX 5070 Ti（Blackwell sm_120，16 GB），装 WSL2 | 主力机，所有正式 benchmark 都在这台上测 |
| 笔记本 RTX 3070 Laptop（Ampere sm_86，8 GB） | 练 kernel，人在外面时用。有功耗墙，测出来的数不算正式结果 |
| 云 GPU | 只在需要多卡（tensor parallel）或 Hopper（H100）时按小时租 |

## 时间预算

| 时期 | 每周投入 |
|---|---|
| 正常上课周 | 10–12 小时 |
| 有面试的周、感恩节、期末 | 4–6 小时 |
| 寒假（约 12/21–1/10） | 25 小时左右 |

每周固定做的事：LeetCode 3–5 题；投递照常进行。

## 里程碑

| # | 截止 | 产出 |
|---|---|---|
| M1 | 11/22 | CUDA kernel 仓库：reduction / softmax / norm / matmul，附 Nsight 数据，以及和 cuBLAS、PyTorch 的对比 |
| M2 | 12/27 | Triton：Llama block 的融合 kernel 和 FlashAttention forward，附两张卡的 benchmark |
| M3a | 1/10 | SGLang 和 vLLM 在 5070 Ti 上的 serving 实验报告 |
| M3b | 1/31 | 迷你推理引擎：paged KV cache + continuous batching + prefix caching |
| M4 | 2/7 | 至少一个提给 SGLang 或 vLLM 的 PR |
| M5 | 3/7 | 简历更新、面试题库、模拟面试 |

---

## 第 0 周（10/8–10/11）环境搭建，约 3–4 小时

照 [docs/setup-windows-wsl2.md](docs/setup-windows-wsl2.md) 一步步做。

- [x] 台式机：在 Windows 上把 NVIDIA 驱动更新到最新，装 WSL2 + Ubuntu。
- [x] WSL 里只装 CUDA Toolkit（12.8 以上，选 WSL-Ubuntu 版）。**不要在 WSL 里装 Linux 驱动**，WSL 用的是 Windows 那边的驱动。
- [x] 装 PyTorch：在 pytorch.org 选 CUDA 12.8 或更新的 wheel。确认 `torch.cuda.get_device_capability()` 返回 `(12, 0)`。Triton 会随 PyTorch 一起装好。
- [x] 用 `nvcc` 编译并跑通一个 hello-world kernel。
- [x] 试一下 `ncu`（Nsight Compute）能不能读到硬件计数器。读不到就记下来，第 4 周再决定要不要给台式机装 Ubuntu 双系统。
- [ ] 配好从 Mac 远程连台式机：用 VS Code 或 Cursor 的 Remote-SSH；人不在家时用 Tailscale。
- [x] 建 GitHub 仓库 `gpu-inference-lab`（私有）和目录骨架。
- **交付**：`notes/env.md`，记下两台机器的 GPU 型号、compute capability，以及驱动、CUDA、PyTorch 的版本。

## 第 1 周（10/12–10/18）GPU 执行模型

面试窗口，面试准备优先，轻量，约 4–5 小时。

- [ ] 读 Horace He 的 *Making Deep Learning Go Brrrr From First Principles*，搞清 compute-bound、memory-bound、overhead-bound 三种瓶颈。
- [ ] 读 PMPP 前 3 章：host 和 device 的分工、kernel launch、grid / block / thread、多维索引。
- [ ] 写两个 kernel：vector add、RGB 转灰度（2D），用 `cudaEvent` 计时。
- [ ] 用 `cudaGetDeviceProperties` 把两张卡的 SM 数、显存位宽、频率打印出来，算出理论带宽。
- **交付**：`cuda/01_basics/`

## 第 2 周（10/19–10/25）硬件调度和内存

仍在面试窗口，轻量。

- [ ] 读 PMPP 第 4–5 章：SM、warp、调度、occupancy、内存层级、数据局部性。
- [ ] GPU MODE 的讲座挑前几讲看（profiling、PMPP 回顾）。
- [ ] 写一个 copy kernel 测带宽，对比合并访存和跨步访存的差别，再和两张卡的理论带宽比。
- **交付**：`notes/memory.md`，用一张图讲清楚合并访存为什么快。

## 第 3 周（10/26–11/1）共享内存和分块

恢复正常节奏，10–12 小时。

- [ ] 读 PMPP 第 5–6 章：shared memory、tiling、bank conflict、性能考量。
- [ ] 先写朴素 matmul，再写共享内存分块 matmul，用 GFLOPS 衡量，并和 cuBLAS（`torch.matmul`）对比。
- [ ] 学会画 roofline：算出两张卡的 FP32 峰值算力和带宽，把你的 kernel 标在图上。
- **交付**：`cuda/02_matmul_tiled/`，外加 roofline 图。

## 第 4 周（11/2–11/8）Reduction 和 warp 原语

- [ ] 读 PMPP 里讲 reduction 的那一章。
- [ ] 把 reduction 从朴素版一步步优化到 warp shuffle（`__shfl_down_sync`）版，每一版都记下带宽。
- [ ] 第一次认真用 Nsight Compute，看三样东西：memory throughput、occupancy、warp stall 的原因。
- **交付**：`cuda/03_reduction/`，每一版的带宽和它占理论峰值的百分比。

## 第 5 周（11/9–11/15）Softmax、LayerNorm 和数值正确性

- [ ] 读 online softmax 的论文（Milakov & Gimelshein, 2018）。
- [ ] 写按行的 softmax（先减最大值）和 online softmax；写 LayerNorm 和 RMSNorm。
- [ ] 写正确性测试：和 PyTorch 对比，fp32 和 fp16 分别定容差。你在 Upstart 做过 1e-5 的数值对齐，这里正好用上。
- **交付**：`cuda/04_softmax_norm/`，附测试和 benchmark。

## 第 6 周（11/16–11/22）Matmul 优化 worklog → M1

- [ ] 照着 Simon Boehm 的 matmul worklog 一步步优化：合并访存 → 共享内存 → 1D / 2D blocktiling → 向量化读取（warptiling 可选）。
- [ ] 每一步都在两张卡上记 GFLOPS。目标：在 5070 Ti 上达到 cuBLAS FP32 的 50% 以上。
- [ ] 把第 1–6 周的结果整理进仓库 README。
- **交付**：**M1** CUDA kernel 仓库。

## 第 7 周（11/23–11/29）Triton 入门

感恩节这周，轻量。

- [ ] 弄懂 Triton 的编程模型：program、block、`tl.load` / `tl.store` 加 mask、autotune。
- [ ] 做官方 tutorial 01（vector add）和 02（fused softmax）。
- [ ] 把第 5 周的 CUDA softmax、这周的 Triton softmax 和 PyTorch 放在一起比。
- **交付**：`triton/01_softmax/`

## 第 8 周（11/30–12/6）Triton matmul 和 torch.compile

- [ ] 做官方 tutorial 03（matmul）和 05（layer norm）。
- [ ] 用 `torch.compile` 编译一个小 MLP，设 `TORCH_LOGS=output_code` 看 Inductor 生成的 Triton 代码，和你自己写的对照。
- **交付**：`notes/inductor.md`，记下 Inductor 做了哪些融合。

## 第 9 周（12/7–12/13）Llama block 的融合 kernel

- [ ] 用 Triton 写三个 kernel：RMSNorm 和 residual 融合、SwiGLU 融合、RoPE。
- [ ] 用 `torch.library` 把它们注册成 custom op，替换进一个小模型（比如 Qwen2.5-0.5B）的 forward，确认输出和原版对齐。
- [ ] 做 benchmark：eager、`torch.compile`、你的 kernel 三者对比，两张卡都测。
- **交付**：`triton/02_llama_block/`

## 第 10 周（12/14–12/20）期末缓冲周

轻量，具体按你的考试安排调。

- [ ] 读 FlashAttention 1 和 2 的论文，搞懂分块和 online softmax 是怎么结合起来的。
- [ ] 开始写 Triton 版的 FlashAttention forward，先不加 causal mask。

## 第 11 周（12/21–12/27）FlashAttention → M2

寒假开始，约 25 小时。

- [ ] 写完 FlashAttention forward 并加上 causal mask。和 `scaled_dot_product_attention` 对齐数值，再比速度。
- [ ] 用 Nsight 看它为什么比朴素 attention 读写显存少。
- [ ] 读 Orca（continuous batching）和 vLLM（PagedAttention）的论文。
- **交付**：**M2** Triton 部分的 README，附两张卡的 benchmark 表。

## 第 12 周（12/28–1/3）部署 SGLang 和 vLLM

- [ ] 在 WSL2 里装最新版的 SGLang 和 vLLM。如果 sm_120 有兼容问题，先去翻它们的 issue。
- [ ] 在 5070 Ti 上部署 Llama-3.1-8B 或 Qwen2.5-7B（FP8 或 AWQ 量化）。用两边自带的 serving benchmark 测 TTFT、TPOT 和吞吐，看它们怎么随并发变化。
- [ ] 手算 KV cache 能放多少 token，再和引擎启动日志里报的数对一下。比如 FP8 权重大约占 8 GB，剩下约 6–7 GB 给 KV cache；fp16 下每个 token 占 128 KB，算下来能放约 5 万个 token。
- **交付**：`serving/01_baseline/`

## 第 13 周（1/4–1/10）Serving 实验报告 → M3a

- [ ] 读 SGLang 论文（RadixAttention）。
- [ ] 做四组对比实验：
  - prefix caching 开和关（用有共享前缀的负载，比如 few-shot、多轮对话）；
  - chunked prefill；
  - FP8 KV cache；
  - 量化方式（FP8 和 AWQ）。
- [ ] 写报告：每组实验的结论，以及 prefill 和 decode 各自的瓶颈在哪。
- **交付**：**M3a** 实验报告。

## 第 14 周（1/11–1/17）迷你推理引擎（一）

春季学期开学，回到每周 10–12 小时。

- [ ] 读 nano-vllm 的代码结构。
- [ ] 用纯 PyTorch 加载 Qwen2.5-0.5B 或 1.5B 的权重，写 forward、连续的 KV cache 和 greedy decoding。逐 token 和 HF transformers 的输出对齐。
- **交付**：`mini-engine/` 第一版。

## 第 15 周（1/18–1/24）迷你推理引擎（二）

- [ ] 实现 paged KV cache：block table，以及 block 的分配和回收。
- [ ] 实现 continuous batching：调度器把 prefill 和 decode 混在一起跑，长度不同的请求能放进同一个 batch。
- [ ] 和 HF 的 `generate` 比吞吐。
- **交付**：引擎第二版，附 benchmark。

## 第 16 周（1/25–1/31）迷你推理引擎（三）→ M3b

- [ ] 实现 prefix caching：给 KV block 做哈希，或者用 radix tree。
- [ ] decode 阶段用 CUDA Graphs 降低 kernel launch 的开销。
- [ ] 写 README：架构图、设计取舍、测量数字。
- **交付**：**M3b** 迷你推理引擎。

## 第 17 周（2/1–2/7）开源贡献 → M4

- [ ] 在 SGLang 或 vLLM 里找 good first issue，比如文档、benchmark、测试、小 bug。
- [ ] 你手上有消费级 Blackwell 卡（sm_120），这类卡的适配问题通常没多少人管。复现并修这类 bug 是很好的切入点。
- **交付**：**M4** 至少提一个 PR。

## 第 18 周（2/8–2/14）简历和 GitHub

- [ ] 从 M1–M4 里挑最强的 1–2 个项目，写进 MLE 或 AIE 底版。要用真实数字，让 Claude 按简历的写法规则一起改。
- [ ] 整理好 GitHub 仓库的 README，把仓库公开。
- [ ] 开始攒面试题库，覆盖 GPU 基础、kernel 优化、推理系统三块。

## 第 19 周（2/15–2/21）LLM serving 系统设计

- [ ] 练 2–3 道系统设计题：
  - 设计一个多租户的 LLM 推理服务，要讲到批处理、KV cache、按前缀路由、自动扩缩容；
  - 设计一个 RAG serving 系统。
- **交付**：每道题写一页笔记。

## 第 20 周（2/22–2/28）白板 kernel 和进阶主题

- [ ] 不看资料，在 30–45 分钟内写出 reduction、softmax、分块 matmul，并讲清每一步优化的理由。
- [ ] 下面三个方向各读一篇：
  - 量化：GPTQ、AWQ、FP8 / FP4；
  - speculative decoding，包括 EAGLE；
  - 并行：TP / PP / EP，以及 NCCL 的 all-reduce。
- **交付**：把下面的自测清单完整过一遍。

## 第 21 周（3/1–3/7）模拟面试和定向投递

- [ ] 做 2–3 次模拟面试，可以找同学，也可以在 GPU MODE 社区里找人。
- [ ] 定向投推理和 ML Infra 方向的岗位。
- [ ] 继续跟进提过的 PR。

---

## 自测清单（春招前都要能做到）

- [ ] 讲清 compute-bound、memory-bound、overhead-bound 的区别，会用 roofline 判断一个 kernel 卡在哪。
- [ ] 讲清合并访存、shared memory、bank conflict、occupancy、warp divergence。
- [ ] 白板写出 reduction、softmax、分块 matmul。
- [ ] 讲清 FlashAttention 为什么快：它是 IO 感知的，靠分块加 online softmax。
- [ ] 会算 KV cache 的大小。例如 Llama-3-8B 用 fp16，每个 token 是 2 × 32 层 × 8 个 KV 头 × 128 维 × 2 字节 = 128 KB。
- [ ] 讲清下面这几样各自解决什么问题：PagedAttention、RadixAttention、continuous batching、chunked prefill、speculative decoding。
- [ ] 讲清 prefill 和 decode 的瓶颈有什么不同，以及量化对两者分别有什么影响。
- [ ] 拿自己的项目讲清三件事：瓶颈在哪、用什么工具测的、提升了多少。

## 资源

- **PMPP**：*Programming Massively Parallel Processors*（第 4 版，Hwu / Kirk / El Hajj）
- **GPU MODE**：YouTube 讲座、Discord，讲义在 https://github.com/gpu-mode/lectures
- **Horace He**：*Making Deep Learning Go Brrrr From First Principles*，https://horace.io/brrr_intro.html
- **Simon Boehm**：*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*，https://siboehm.com/articles/22/CUDA-MMM
- **Triton 官方 tutorial**：https://triton-lang.org/main/getting-started/tutorials/index.html
- **NVIDIA 官方文档**：CUDA C++ Programming Guide、Nsight Compute / Nsight Systems 文档
- **论文**：
  - online softmax：arXiv 1805.02867
  - FlashAttention 1 / 2 / 3：arXiv 2205.14135、2307.08691、2407.08608
  - Orca：OSDI 2022
  - vLLM / PagedAttention：arXiv 2309.06180
  - SGLang：arXiv 2312.07104
  - speculative decoding：arXiv 2211.17192
  - EAGLE：arXiv 2401.15077
  - GPTQ：arXiv 2210.17323
  - AWQ：arXiv 2306.00978
- **代码**：
  - SGLang：https://github.com/sgl-project/sglang
  - vLLM：https://github.com/vllm-project/vllm
  - nano-vllm：https://github.com/GeeeekExplorer/nano-vllm
