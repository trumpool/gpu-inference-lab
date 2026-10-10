// 第 0 周：确认 nvcc 能编译、GPU 能跑 kernel。
// 编译并运行：nvcc -arch=native hello.cu -o hello && ./hello
#include <cstdio>

__global__ void hello() {
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    return 0;
}
