#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void simple_loop_kernel_dd(int* iterations_ptr, int* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx == 0) {
        int iterations = *iterations_ptr;   
        int i = 0;
        while (i < iterations) {
            *ptr = *ptr + 1;   
            i++;
        }
    }
}

int main() {
    int *h_data, *d_data;
    int *h_iterations, *d_iterations;
    int initial_value = 100;
    int iterations_value = 5;   
    
     
    h_data = (int*)malloc(sizeof(int));
    h_iterations = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    *h_iterations = iterations_value;
    
     
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_iterations, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iterations, h_iterations, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Loop iterations: %d\n", *h_iterations);
    
     
    simple_loop_kernel_dd<<<1, 1>>>(d_iterations, d_data);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
     
    cudaFree(d_data);
    cudaFree(d_iterations);
    free(h_data);
    free(h_iterations);
    
    return 0;
}