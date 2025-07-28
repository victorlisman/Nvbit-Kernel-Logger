#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void complex_loop_kernel(int* ptr, int max_iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx == 0) {
        int i = 0;
        while (i < max_iterations) {
             
            if (i % 2 == 0) {
                *ptr = *ptr + 1;   
            } else {
                *ptr = *ptr * 2;   
            }
            
             
            if (*ptr > 1000) {
                break;
            }
            
            i++;
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 10;
    int max_iterations = 15;   
    
     
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
     
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Max iterations: %d\n", max_iterations);
    printf("Pattern: even iterations +1, odd iterations *2\n");
    printf("Early break if value > 1000\n");
    
     
    complex_loop_kernel<<<1, 1>>>(d_data, max_iterations);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
     
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}