# SGLang 快速上手（5070 Ti / WSL2）

SGLang 副线第 2–3 周的第一步：在台式机上把 SGLang 跑起来，用 OpenAI 兼容接口发请求。

## 环境（2026-10-23 装好）

| 项目 | 版本 |
|---|---|
| sglang | 0.5.22 |
| torch | 2.14.1+cu130（SGLang 现在只有 CUDA 13 这一条线，CUDA 12 的版本停在 0.5.19） |
| sglang-kernel / flashinfer-python | 0.4.9 / 0.7.1 |
| Windows 驱动 | 610.88（支持到 CUDA 13.3，所以能跑 cu130 的 wheel） |

**SGLang 装在单独的 venv：`~/venvs/sglang`**，不要装进仓库的 `.venv`。它会装自己的 torch 2.14，会把课程环境的 torch 2.11+cu128 换掉。

```bash
python3 -m venv ~/venvs/sglang
~/venvs/sglang/bin/pip install --upgrade pip uv
VIRTUAL_ENV=~/venvs/sglang ~/venvs/sglang/bin/uv pip install --prerelease=allow sglang
```

（命令来自 https://docs.sglang.io/get_started/install.html ，装完大约 10 GB。）

## 启动和测试

```bash
bash serving/00_sglang_quickstart/launch.sh          # 默认 Qwen/Qwen2.5-1.5B-Instruct，127.0.0.1:30000
~/venvs/sglang/bin/python serving/00_sglang_quickstart/client.py
```

第一次启动约 70 秒，大部分时间花在捕获 CUDA graph（prefill 39 s）。看到 `The server is fired up and ready to roll!` 就可以发请求了。

## 踩过的坑

- **用之前一定要激活 venv。** FlashInfer 第一次用到某个 kernel 时会用 nvcc 现场编译（JIT），编译要用 venv `bin/` 里的 `ninja`。不激活的话报 `FileNotFoundError: 'ninja'`。`launch.sh` 里已经激活了。
- **JIT 用的是系统的 CUDA 12.9 nvcc**（`/usr/local/cuda`），编出来是 `sm_120f`，和 torch 的 CUDA 13 混用目前没出问题。以后如果编译报错，可以试着把 `CUDA_HOME` 指到 venv 里 pip 装的 CUDA 13.4 nvcc（`~/venvs/sglang/lib/python3.12/site-packages/nvidia/cu13`）。
- **前一两个请求很慢**（1.6 s、4.2 s），也是在 JIT 编译，之后一个请求约 0.16 s。做 benchmark 前一定要先预热。
- **服务开着时会占掉约 15 GB 显存**（默认 `mem_fraction_static=0.751`，大部分给 KV cache）。这时候跑大一点的 CUDA 作业会显存不够，用完记得 Ctrl-C。

## 第一次运行看到的东西

**radix cache 复用前缀**：`client.py` 里两个请求共用一段约 1170 token 的 system prompt，服务器日志：

```
Prefill batch, #new-seq: 1, #new-token: 1165, #cached-token: 5,    ...   ← 第 1 次：前缀全部要算
Prefill batch, #new-seq: 1, #new-token: 13,   #cached-token: 1155, ...   ← 第 2 次：前缀命中缓存
```

**KV cache 的大小**：日志里是 `#tokens: 297095, K size: 3.97 GB, V size: 3.97 GB`。手算对得上：

```
Qwen2.5-1.5B：28 层，2 个 KV 头（GQA），head_dim = 1536 / 12 = 128，bf16 每个数 2 字节
每个 token = 2 (K 和 V) × 28 × 2 × 128 × 2 字节 = 28,672 字节 = 28 KiB
297,095 个 token × 28 KiB ≈ 7.93 GiB
```

对比 PLAN.md 自测清单里的 Llama-3-8B（每个 token 128 KB）：这个模型 KV 头少、层数少，所以每个 token 小很多。
