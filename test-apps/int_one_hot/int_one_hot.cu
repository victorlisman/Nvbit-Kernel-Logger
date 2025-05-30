#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void int_one_hot(float* output, short input_val) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < 65536) 
    {
        output[idx] = (idx == (int)input_val) ? 1.0f : 0.0f;
    }
}

int main() {
    short input_value = 1234;
    
    const int output_size = 65536; 
    float* h_output = (float*)malloc(output_size * sizeof(float));
    
    float* d_output;
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cudaMemset(d_output, 0, output_size * sizeof(float));
    
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    printf("Input value: %d\n", input_value);
    printf("Launching kernel with %d blocks, %d threads per block\n", blocks, threads_per_block);
    
    int_one_hot<<<blocks, threads_per_block>>>(d_output, input_value);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("One-hot encoding verification:\n");
    for (int i = input_value - 5; i <= input_value + 5; i++) 
    {
        if (i >= 0 && i < output_size) 
        {
            printf("output[%d] = %.1f\n", i, h_output[i]);
        }
    }
    
    int count_ones = 0;
    int one_index = -1;
    for (int i = 0; i < output_size; i++) 
    {
        if (h_output[i] == 1.0f) 
        {
            count_ones++;
            one_index = i;
        }
    }
    
    printf("Number of 1.0 values: %d\n", count_ones);
    if (count_ones == 1 && one_index == input_value) 
    {
        printf("SUCCESS: One-hot encoding is correct! Value 1.0 found at index %d\n", one_index);
    }
    else 
    {
        printf("ERROR: One-hot encoding failed!\n");
    }
    
    free(h_output);
    cudaFree(d_output);
    
    return 0;
}