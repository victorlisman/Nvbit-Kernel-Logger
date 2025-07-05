#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void multiple_branches_kernel(int condition, int* ptr1, int* ptr2, int* ptr3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (condition == 0) {
            *ptr1 = 42;  // Write to the first pointer
        } else if (condition == 1) {
            *ptr2 = 42;  // Write to the second pointer
        } else if (condition == 2) {
            *ptr3 = 42;  // Write to the third pointer
        }
    }
}

int main() {
    int *h_ptr1, *h_ptr2, *h_ptr3;
    int *d_ptr1, *d_ptr2, *d_ptr3;
    int condition = 1;  // Change to 0, 1, or 2 to test different branches
    
    // Allocate host memory
    h_ptr1 = (int*)malloc(sizeof(int));
    h_ptr2 = (int*)malloc(sizeof(int));
    h_ptr3 = (int*)malloc(sizeof(int));
    *h_ptr1 = 0;
    *h_ptr2 = 0;
    *h_ptr3 = 0;
    
    // Allocate device memory
    cudaMalloc(&d_ptr1, sizeof(int));
    cudaMalloc(&d_ptr2, sizeof(int));
    cudaMalloc(&d_ptr3, sizeof(int));
    cudaMemcpy(d_ptr1, h_ptr1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr2, h_ptr2, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr3, h_ptr3, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Condition: %d\n", condition);
    
    // Launch kernel with 1 block, 1 thread
    multiple_branches_kernel<<<1, 1>>>(condition, d_ptr1, d_ptr2, d_ptr3);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_ptr1, d_ptr1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ptr2, d_ptr2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ptr3, d_ptr3, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final values:\n");
    printf("ptr1: %d\n", *h_ptr1);
    printf("ptr2: %d\n", *h_ptr2);
    printf("ptr3: %d\n", *h_ptr3);
    
    // Cleanup
    cudaFree(d_ptr1);
    cudaFree(d_ptr2);
    cudaFree(d_ptr3);
    free(h_ptr1);
    free(h_ptr2);
    free(h_ptr3);
    
    return 0;
}