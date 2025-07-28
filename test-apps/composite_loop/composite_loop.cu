#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void composite_loop_kernel(int iterations1, int iterations2, int* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx == 0) {
         
        for (int i = 0; i < iterations1; i++) {
            *ptr = *ptr + 1;   
        }
        
         
        for (int j = 0; j < iterations2; j++) {
            *ptr = *ptr - 1;   
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    int iterations1 = 5;   
    int iterations2 = 3;   
    
     
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
     
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Increment iterations: %d\n", iterations1);
    printf("Decrement iterations: %d\n", iterations2);
    printf("Net change: %d\n", iterations1 - iterations2);
    printf("Expected final value: %d\n", initial_value + iterations1 - iterations2);
    
     
    composite_loop_kernel<<<1, 1>>>(iterations1, iterations2, d_data);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
     
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}