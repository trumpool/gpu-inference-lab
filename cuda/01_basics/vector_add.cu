// 第 1 周：vector add，c[i] = a[i] + b[i]
// 编译并运行：nvcc -O2 -arch=native vector_add.cu -o vector_add && ./vector_add
#include <cstdio>
#include <cstdlib>

#include "../common/cuda_check.h"

// 每个线程算一个全局编号 i，负责 c[i]；i 超过 n 的线程什么都不做
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 28;  // 约 2.7 亿个 float，每个数组 1 GiB
    const size_t bytes = n * sizeof(float);
    const int iters = 20;   // 计时时 kernel 跑的次数，取平均

    // 1. 在 CPU 上分配 a、b、c，并给 a、b 填上随机数
    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);
    if (!a || !b || !c) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. 在 GPU 上分配 d_a、d_b、d_c
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 3. 把 a、b 拷到 GPU
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // 4. 预热：先启动一次
    int threads = 256;
    int blocks = (n + threads - 1) / threads;  // 向上取整
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());          // 启动错误
    CUDA_CHECK(cudaDeviceSynchronize());     // 执行错误

    // 5. 用 cudaEvent 计时：kernel 连跑 iters 次，只包住 kernel，不包住 cudaMemcpy
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // 读时间之前先等 stop 真正发生

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float milliseconds = total_ms / iters;

    // 6. 把 d_c 拷回 c，在 CPU 上逐个检查 c[i] == a[i] + b[i]
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, a[i], b[i], c[i]);
            return 1;
        }
    }

    // 7. 打印 kernel 时间（ms）和有效带宽（GB/s）
    //   有效带宽 = 这个 kernel 读写的总字节数 / 时间
    double total_bytes = 3.0 * bytes;  // 读 a、读 b、写 c
    double bandwidth = total_bytes / (milliseconds * 1e6);  // GB/s
    printf("Kernel time: %.3f ms (avg of %d runs)\n", milliseconds, iters);
    printf("Effective bandwidth: %.3f GB/s\n", bandwidth);

    // 8. 释放 event、GPU 和 CPU 上的内存
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(a);
    free(b);
    free(c);

    return 0;
}
