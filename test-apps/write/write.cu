#include <cuda_runtime.h>
#include <iostream>

__global__ void write(float *out) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[global_idx] = 1.0f;
}

int main() {
    const int threads_per_block = 128;
    const int num_blocks = 4;
    const int total_threads = threads_per_block * num_blocks;
    const int size = total_threads * sizeof(float);

    // Allocate device memory
    float *d_out;
    cudaMalloc(&d_out, size);

    // Launch the kernel
    write<<<num_blocks, threads_per_block>>>(d_out);

    // Copy results back to host
    float *h_out = new float[total_threads];
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print a few values to verify
    std::cout << "First 10 values after write kernel:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << "\n";
    }

    // Cleanup
    delete[] h_out;
    cudaFree(d_out);

    return 0;
}