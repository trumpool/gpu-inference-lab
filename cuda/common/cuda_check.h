// 公共的 CUDA 错误检查，各周的 .cu 文件都 #include 这个头文件。
//
// 用法：
//   CUDA_CHECK(cudaMalloc(&d_a, bytes));   // 每个 CUDA API 调用都包起来
//
//   kernel<<<blocks, threads>>>(...);
//   CUDA_CHECK(cudaGetLastError());         // 启动错误：紧跟在 <<<>>> 后面
//   CUDA_CHECK(cudaDeviceSynchronize());    // 执行错误：要到同步点才报出来
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                    cudaGetErrorName(err_), __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)
