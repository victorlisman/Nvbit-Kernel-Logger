#include <cuda_runtime.h>
#include <iostream>

// Kernel: each thread copies one float from x to y
__global__ void readWriteKernel(float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x[idx];
}

int main() {
    float* d_y;
    int N = 1024;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_y, size);
    // Allocate host memory
    float* h_x = new float[N];
    float* h_y = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_x;
    cudaMalloc(&d_x, size);

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);  // Ceiling division
    printf("blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);

    // Launch the kernel
    readWriteKernel<<<gridDim, blockDim>>>(d_x, d_y);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Validate result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        void *write_addr = static_cast<void*>(d_y + i);
        //std::cout << "h_y[" << i << "] = " << h_y[i] << " written to " << write_addr << std::endl;
        if (h_y[i] != h_x[i]) {
            //std::cout << "Mismatch at index " << i << ": " << h_y[i] << " != " << h_x[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "Success: all values match!" << std::endl;
    }

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;

    return 0;
}