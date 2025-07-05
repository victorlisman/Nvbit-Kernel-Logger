#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void copy_element_kernel(int* X, int* Y, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only one thread performs the copy operation
    if (idx == 0 && i < N) {
        Y[i] = X[i];
    }
}

int main() {
    int *h_X, *h_Y;  // Host arrays
    int *d_X, *d_Y;  // Device arrays
    int i = 42;      // Index to copy (example)
    
    // Allocate host memory
    h_X = (int*)malloc(N * sizeof(int));
    h_Y = (int*)malloc(N * sizeof(int));
    
    // Initialize arrays
    for (int j = 0; j < N; j++) {
        h_X[j] = j * 2;  // Example values
        h_Y[j] = 0;      // Initialize Y to zeros
    }
    
    // Allocate device memory
    cudaMalloc(&d_X, N * sizeof(int));
    cudaMalloc(&d_Y, N * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_X, h_X, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    printf("Copying element at index %d from X to Y\n", i);
    printf("Before: X[%d] = %d, Y[%d] = %d\n", i, h_X[i], i, h_Y[i]);
    
    copy_element_kernel<<<grid, block>>>(d_X, d_Y, i);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("After:  X[%d] = %d, Y[%d] = %d\n", i, h_X[i], i, h_Y[i]);
    
    // Verify the copy
    if (h_Y[i] == h_X[i]) {
        printf("SUCCESS: Element copied correctly!\n");
    } else {
        printf("ERROR: Element not copied correctly!\n");
    }
    
    // Cleanup
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);
    
    return 0;
}