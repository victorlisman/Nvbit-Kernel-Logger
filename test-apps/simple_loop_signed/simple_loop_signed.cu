#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void simple_loop_kernel_signed(int* ptr, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx == 0) {
        int i = 0;
        while (i < iterations) {
            *ptr = *ptr + 1;   
            i++;
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    int iterations = 5;   
    
     
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
     
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Loop iterations: %d\n", iterations);
    
     
    simple_loop_kernel_signed<<<1, 1>>>(d_data, iterations);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
     
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}