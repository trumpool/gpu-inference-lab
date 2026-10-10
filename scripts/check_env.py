#!/usr/bin/env python3
"""第 0 周：检查 WSL2 里的 GPU 环境有没有配好（PyTorch + CUDA + Triton）。

用法：python scripts/check_env.py
全部通过时退出码是 0。把输出贴进 notes/env.md。
"""
import platform
import sys
import time

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

except ImportError:
    triton = None


def main() -> int:
    print(f"Python      {sys.version.split()[0]}  ({platform.platform()})")
    try:
        import torch
    except ImportError:
        print("✗ 没装 PyTorch：pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1
    print(f"PyTorch     {torch.__version__}  (编译时的 CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("✗ torch.cuda.is_available() 是 False：先确认 Windows 上的 NVIDIA 驱动是最新的，"
              "再看 WSL 里 nvidia-smi 能不能列出显卡")
        return 1

    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    major, minor = torch.cuda.get_device_capability(dev)
    sm = f"sm_{major}{minor}"
    print(f"GPU         {prop.name}  {sm}  {prop.total_memory / 2**30:.1f} GiB  {prop.multi_processor_count} SMs")
    arch_list = torch.cuda.get_arch_list()
    if sm not in arch_list:
        print(f"⚠ 这版 PyTorch 编译时没有带 {sm}（只带了 {' '.join(arch_list)}）。"
              "5070 Ti 是 sm_120，要用 CUDA 12.8 以上的 PyTorch 轮子")

    # 跑一个 fp16 矩阵乘法，确认 kernel 真的能跑，顺便粗测一下 TFLOPS
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):  # 预热
        a @ b
    torch.cuda.synchronize()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"matmul      fp16 {n}x{n}：每次 {dt * 1e3:.2f} ms，约 {2 * n**3 / dt / 1e12:.1f} TFLOPS")

    if triton is None:
        print("✗ 没装 Triton（一般会随 PyTorch 一起装好）")
        return 1
    x = torch.randn(1_000_003, device="cuda")  # 故意不是 BLOCK 的整数倍，顺便测 mask
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    block = 1024
    _add_kernel[(triton.cdiv(x.numel(), block),)](x, y, out, x.numel(), BLOCK=block)
    ok = torch.allclose(out, x + y)
    print(f"Triton      {triton.__version__}  vector add {'✓ 结果正确' if ok else '✗ 结果不对'}")
    if not ok:
        return 1

    print("\n✓ 环境就绪。把上面的输出贴进 notes/env.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
