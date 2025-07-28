#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void multiple_branches_for_loop_kernel(int iterations, int* ptr1, int* ptr2, int* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
     
    if (idx == 0) {
        for (int i = 0; i < iterations; i++) {
            int branch = i % 3;   
            
            if (branch == 0) {
                *ptr1 = *ptr1 + 1;   
            } else if (branch == 1) {
                *ptr2 = *ptr2 + 1;   
            } else if (branch == 2) {
                *ptr3 = *ptr3 + 1;   
            }
        }
    }
}

int main() {
    int *h_ptr1, *h_ptr2, *h_ptr3;
    int *d_ptr1, *d_ptr2, *d_ptr3;
    int iterations = 9;   
    
     
    h_ptr1 = (int*)malloc(sizeof(int));
    h_ptr2 = (int*)malloc(sizeof(int));
    h_ptr3 = (int*)malloc(sizeof(int));
    *h_ptr1 = 0;
    *h_ptr2 = 0;
    *h_ptr3 = 0;
    
     
    cudaMalloc(&d_ptr1, sizeof(int));
    cudaMalloc(&d_ptr2, sizeof(int));
    cudaMalloc(&d_ptr3, sizeof(int));
    cudaMemcpy(d_ptr1, h_ptr1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr2, h_ptr2, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr3, h_ptr3, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Loop iterations: %d\n", iterations);
    printf("Expected increments per pointer: %d\n", iterations / 3);
    
     
    multiple_branches_for_loop_kernel<<<1, 1>>>(iterations, d_ptr1, d_ptr2, d_ptr3);
    cudaDeviceSynchronize();
    
     
    cudaMemcpy(h_ptr1, d_ptr1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ptr2, d_ptr2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ptr3, d_ptr3, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final values:\n");
    printf("ptr1: %d\n", *h_ptr1);
    printf("ptr2: %d\n", *h_ptr2);
    printf("ptr3: %d\n", *h_ptr3);
    
     
    cudaFree(d_ptr1);
    cudaFree(d_ptr2);
    cudaFree(d_ptr3);
    free(h_ptr1);
    free(h_ptr2);
    free(h_ptr3);
    
    return 0;
}