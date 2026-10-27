// 第 1 周：打印 GPU 的硬件参数，算出理论显存带宽（和 FP32 峰值算力）
// 编译并运行：nvcc -O2 -arch=native device_info.cu -o device_info && ./device_info
#include <cstdio>

#include "../common/cuda_check.h"

int main() {
    // TODO 1：用 cudaGetDeviceCount 拿到 GPU 数量，对每块卡循环
    //   （台式机只有 1 块，但笔记本上也要能跑，所以写成循环）
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));


    // TODO 2：用 cudaGetDeviceProperties(&prop, dev) 拿到 cudaDeviceProp，打印下面这些字段：
    //   - name                          显卡名字
    //   - major, minor                  compute capability，5070 Ti 应该是 12.0
    //   - multiProcessorCount           SM 数
    //   - totalGlobalMem                显存大小（字节，换成 GiB 打印）
    //   - l2CacheSize                   L2 大小（字节，换成 MiB 打印）——rgb_to_gray 踩的那个坑
    //   - sharedMemPerBlock             每个 block 能用的 shared memory（第 3 周要用）
    //   - maxThreadsPerBlock, warpSize  每个 block 最多多少线程、一个 warp 多少线程
    //   - memoryBusWidth                显存位宽（bit）
    //   - memoryClockRate               显存频率（单位是 kHz！）
    //   - clockRate                     SM 频率（单位也是 kHz）
    for(int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM count: %d\n", prop.multiProcessorCount);
        printf("  Total global memory: %.2f GiB\n", (float)prop.totalGlobalMem / (1 << 30));
        printf("  L2 cache size: %.2f MiB\n", (float)prop.l2CacheSize / (1 << 20));
        printf("  Shared memory per block: %.2f KiB\n", (float)prop.sharedMemPerBlock / (1 << 10));
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Memory clock rate: %.2f MHz\n", (float)prop.memoryClockRate / 1000);
        printf("  SM clock rate: %.2f MHz\n", (float)prop.clockRate / 1000);
    }
    // TODO 3：算理论显存带宽（GB/s），打印出来，和 896 GB/s 对一下
    //   理论带宽 = 显存频率（每秒多少个时钟） x 每个时钟传几次 x 每次传多少字节
    //   - memoryClockRate 是 kHz，先换成 Hz
    //   - 每个时钟传 2 次（DDR：时钟的上升沿和下降沿各传一次）
    //   - 每次传多少字节 = memoryBusWidth（bit）/ 8
    //   - 最后除以 1e9 得到 GB/s
    for(int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        float bandwidth = (float)prop.memoryClockRate * 1000 * 2 * (prop.memoryBusWidth / 8) / 1e9;
        printf("Device %d: Theoretical memory bandwidth: %.2f GB/s\n", i, bandwidth);
    }

    // TODO 4（选做）：算 FP32 峰值算力（TFLOPS），第 3 周画 roofline 要用
    //   峰值 = SM 数 x 每个 SM 的 FP32 核心数 x 每个核心每个时钟做几次运算 x SM 频率
    //   - 每个 SM 的 FP32 核心数不在 cudaDeviceProp 里：sm_120（5070 Ti）是 128，sm_86（3070 Laptop）也是 128
    //   - 一条 FMA（a * b + c）算 2 次运算
    //   - 然后想一想：vector add 每字节只做 1/12 次运算，离"算力 / 带宽"差多远？
    for(int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        int fp32_cores_per_sm = 0;
        if(prop.major == 12 && prop.minor == 0) {
            fp32_cores_per_sm = 128; // sm_120
        } else if(prop.major == 8 && prop.minor == 6) {
            fp32_cores_per_sm = 128; // sm_86
        } else {
            printf("Unknown compute capability %d.%d, cannot determine FP32 cores per SM\n", prop.major, prop.minor);
            continue;
        }
        float peak_flops = (float)prop.multiProcessorCount * fp32_cores_per_sm * 2 * (float)prop.clockRate * 1000 / 1e12; // TFLOPS
        printf("Device %d: Theoretical FP32 peak performance: %.2f TFLOPS\n", i, peak_flops);
    }

    return 0;
}
