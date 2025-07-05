#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void reverse_index_copy_kernel(int* X, int* Y, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only one thread performs the copy operation
    if (idx == 0 && i >= 0 && i < N) {
        int reverse_pos = N - i - 1;
        Y[reverse_pos] = X[i];
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
        h_X[j] = j * 2 + 10;  // Example values: 10, 12, 14, 16, ...
        h_Y[j] = 0;           // Initialize Y to zeros
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
    
    int reverse_pos = N - i - 1;
    
    printf("Copying element from index %d to reverse position %d\n", i, reverse_pos);
    printf("Array size N = %d\n", N);
    printf("Source position: %d\n", i);
    printf("Target position: %d (N - %d - 1 = %d - %d - 1 = %d)\n", 
           reverse_pos, i, N, i, reverse_pos);
    printf("\n");
    
    printf("Before: X[%d] = %d, Y[%d] = %d\n", i, h_X[i], reverse_pos, h_Y[reverse_pos]);
    
    reverse_index_copy_kernel<<<grid, block>>>(d_X, d_Y, i);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("After:  X[%d] = %d, Y[%d] = %d\n", i, h_X[i], reverse_pos, h_Y[reverse_pos]);
    
    // Verify the copy
    if (h_Y[reverse_pos] == h_X[i]) {
        printf("SUCCESS: Element copied correctly to reverse position!\n");
    } else {
        printf("ERROR: Element not copied correctly!\n");
    }
    
    // Show a few examples of the reverse mapping for clarity
    printf("\nReverse index mapping examples:\n");
    for (int j = 0; j < 5; j++) {
        int rev = N - j - 1;
        printf("Index %d maps to reverse position %d\n", j, rev);
    }
    printf("...\n");
    for (int j = N-5; j < N; j++) {
        int rev = N - j - 1;
        printf("Index %d maps to reverse position %d\n", j, rev);
    }
    
    // Cleanup
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);
    
    return 0;
}