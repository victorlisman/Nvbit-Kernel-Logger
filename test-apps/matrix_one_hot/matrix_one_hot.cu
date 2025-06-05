#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void one_hot_encode_kernel(short* input, int* output, int N, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int class_idx = input[idx];
        if (class_idx >= 0 && class_idx < num_classes) {
            output[idx * num_classes + class_idx] = 1;
        }
    }
}

void one_hot_encode(short* h_input, int* h_output, int N, int num_classes) {
    short* d_input;
    int* d_output;

    size_t input_size = N * sizeof(short);
    size_t output_size = N * num_classes * sizeof(int);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size);

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    one_hot_encode_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, N, num_classes);
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int N = 5;
    const int num_classes = 4;
    short h_input[N] = {0, 2, 1, 3, 0};
    int h_output[N * num_classes] = {0};

    one_hot_encode(h_input, h_output, N, num_classes);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            std::cout << h_output[i * num_classes + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}