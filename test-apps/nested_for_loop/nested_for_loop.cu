#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void nested_for_loop_kernel(int* ptr, int outer_iterations, int inner_iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx > -1 && idx < 257) {
        for (int i = 0; i < outer_iterations; i++) {
            for (int j = 0; j < inner_iterations; j++) {
                *ptr = *ptr + 1;   
            }
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    int outer_iterations = 3;   
    int inner_iterations = 4;   
    
     
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
     
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Outer iterations: %d\n", outer_iterations);
    printf("Inner iterations: %d\n", inner_iterations);
    printf("Total operations: %d\n", outer_iterations * inner_iterations);
    
     
    nested_for_loop_kernel<<<256, 256>>>(d_data, outer_iterations, inner_iterations);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    printf("Expected final value: %d\n", initial_value + (outer_iterations * inner_iterations));
    
     
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}