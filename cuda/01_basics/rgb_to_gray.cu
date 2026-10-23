// 第 1 周：RGB 转灰度（2D grid / 2D block）
// gray = 0.299 * R + 0.587 * G + 0.114 * B
// 编译并运行：nvcc -O2 -arch=native rgb_to_gray.cu -o rgb_to_gray && ./rgb_to_gray
#include <cstdio>
#include <cstdlib>

#include "../common/cuda_check.h"

// 图像在内存里的排布（行优先，RGB 交错存放）：
//   像素 (row, col) 的下标 idx = row * width + col
//   rgb[3 * idx + 0] = R，rgb[3 * idx + 1] = G，rgb[3 * idx + 2] = B
//   gray[idx] = 灰度值

// 每个线程负责一个像素：x 方向算出 col，y 方向算出 row；越界的线程什么都不做
__global__ void to_grey(const unsigned char* rgb, unsigned char* gray, const int width, const int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < height){
        int index = row * width + col;
        unsigned char R = rgb[3 * index];
        unsigned char G = rgb[3 * index + 1];
        unsigned char B = rgb[3 * index + 2];
        float f_gray = 0.299f * R + 0.587f * G + 0.114f * B;
        unsigned char gray_val = (unsigned char)f_gray;
        gray[index] = gray_val;
    }
}

int main() {
    // 宽高都故意不是 16 的倍数，两个方向的边界检查都能测到。
    // 数据量（每像素 3 + 1 字节，约 183 MiB）要比 L2 缓存（5070 Ti 是 48 MiB）大好几倍，
    // 否则重复跑 kernel 时数据留在 L2 里，测到的是 L2 带宽而不是显存带宽
    // （4001 x 3001 约 45 MiB，能测出 950 GB/s，超过显存理论值 896 GB/s）。
    const int width = 8001;
    const int height = 6001;
    const int num_pixels = width * height;
    const size_t rgb_bytes = 3 * (size_t)num_pixels * sizeof(unsigned char);
    const size_t gray_bytes = (size_t)num_pixels * sizeof(unsigned char);
    const int iters = 20;

    // 1. 在 CPU 上分配 rgb（输入）和 gray（输出），给 rgb 填随机值（0–255）
    unsigned char *rgb  = (unsigned char *)malloc(rgb_bytes);
    unsigned char *gray = (unsigned char *)malloc(gray_bytes);
    if (!rgb || !gray) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    for(int i = 0; i < 3 * num_pixels; i++){
        rgb[i] = rand() % 256;
    }

    // 2. 在 GPU 上分配 d_rgb、d_gray，把 rgb 拷过去（gray 是输出，不用拷）
    unsigned char *d_rgb, *d_gray;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMemcpy(d_rgb, rgb, rgb_bytes, cudaMemcpyHostToDevice));

    // 3. 2D 的 block 和 grid：block 16 x 16，grid 两个方向都向上取整；先预热一次
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    to_grey<<<grid, block>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaGetLastError());          // 启动错误
    CUDA_CHECK(cudaDeviceSynchronize());     // 执行错误

    // 4. 用 cudaEvent 计时：kernel 连跑 iters 次取平均
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0; i < iters; i++){
        to_grey<<<grid, block>>>(d_rgb, d_gray, width, height);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 5. 拷回 gray，在 CPU 上用同一个公式算一遍，逐个比较
    //   允许差 1：GPU 上编译器会把乘加合并成 FMA，舍入可能和 CPU 不同，
    //   转成 unsigned char 时偶尔会差 1。超过 1 才算错
    CUDA_CHECK(cudaMemcpy(gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int index = row * width + col;
            unsigned char R = rgb[3 * index];
            unsigned char G = rgb[3 * index + 1];
            unsigned char B = rgb[3 * index + 2];
            float f_gray = 0.299f * R + 0.587f * G + 0.114f * B;
            unsigned char gray_val = (unsigned char)f_gray;
            if(abs(gray[index] - gray_val) > 1){
                printf("Mismatch at (%d, %d): GPU gray = %d, CPU gray = %d\n", row, col, gray[index], gray_val);
                exit(1);
            }
        }
    }

    // 6. 打印时间和有效带宽：读一遍 rgb（每像素 3 字节），写一遍 gray（每像素 1 字节）
    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float milliseconds = total_ms / iters;
    double total_bytes = rgb_bytes + gray_bytes;
    double bandwidth = total_bytes / (milliseconds * 1e6);  // GB/s

    printf("Image: %d x %d, grid %u x %u, block %u x %u\n", width, height, grid.x, grid.y, block.x, block.y);
    printf("Kernel time: %.4f ms (avg of %d runs)\n", milliseconds, iters);
    printf("Effective bandwidth: %.3f GB/s\n", bandwidth);

    // 7. 释放 event、GPU 和 CPU 上的内存
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    free(rgb);
    free(gray);

    return 0;
}
