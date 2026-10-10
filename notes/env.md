# 环境记录

| 项目 | 台式机 | 笔记本 |
|---|---|---|
| GPU | RTX 5070 Ti | RTX 3070 Laptop |
| compute capability | sm_120 | |
| 显存 | 15.9 GiB（nvidia-smi：16303 MiB） | |
| SM 数 | 70 | |
| NVIDIA 驱动（Windows） | 610.88（CUDA UMD 13.3） | |
| CUDA Toolkit（`nvcc --version`） | 12.9（V12.9.86，`cuda-toolkit-12-9`，WSL-Ubuntu 源） | |
| PyTorch 版本 / 编译时的 CUDA 版本 | 2.11.0+cu128 / 12.8 | |
| Triton | 3.6.0（vector add ✓） | |
| `check_env.py` 测出的 fp16 matmul TFLOPS | 约 99.6（4096×4096，1.38 ms） | |
| Nsight Compute 能不能读硬件计数器 | 不能：ncu 2025.2.1 报 `ERR_NVGPUCTRPERM`，第 4 周再处理 | |

台式机系统：Windows 11 Pro 26200 + WSL 2.6.1，Ubuntu 24.04.5 LTS，内核 6.6.87.2-microsoft-standard-WSL2，Python 3.12.3。

## check_env.py 的输出

台式机（2026-10-09）：

```
Python      3.12.3  (Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39)
PyTorch     2.11.0+cu128  (编译时的 CUDA 12.8)
GPU         NVIDIA GeForce RTX 5070 Ti  sm_120  15.9 GiB  70 SMs
matmul      fp16 4096x4096：每次 1.38 ms，约 99.6 TFLOPS
Triton      3.6.0  vector add ✓ 结果正确

✓ 环境就绪。把上面的输出贴进 notes/env.md。
```

## cuda/00_hello 的输出

台式机（`nvcc -arch=native hello.cu -o hello && ./hello`）：

```
hello from block 0, thread 0
hello from block 0, thread 1
hello from block 0, thread 2
hello from block 0, thread 3
hello from block 1, thread 0
hello from block 1, thread 1
hello from block 1, thread 2
hello from block 1, thread 3
device: NVIDIA GeForce RTX 5070 Ti (sm_120)
```

## 踩过的坑

- **Triton 报 `Python.h: No such file or directory`**：Triton 第一次运行时要用 gcc 编译一个 C 扩展，需要 Python 头文件。`sudo apt install -y python3-dev` 之后就好了。（`docs/setup-windows-wsl2.md` 第 3 步的 apt 命令里还没有这个包。）
- **`ncu ./cuda/00_hello/hello` 报 `ERR_NVGPUCTRPERM`**：kernel 能跑，但读不了性能计数器。按文档先记下，第 4 周要用 Nsight 时再在 Windows 的 NVIDIA 控制面板里打开「允许所有用户访问 GPU 性能计数器」。
- venv 里没装 numpy，import torch 时会在 stderr 打一条 `Failed to initialize NumPy` 的警告，不影响上面的检查。
