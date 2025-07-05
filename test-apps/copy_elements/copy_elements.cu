#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void copy_elements_kernel(int* X, int* indices, int* Y, int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one index from the list
    if (idx < num_indices) {
        int i = indices[idx];
        if (i >= 0 && i < N) {  // Bounds check
            Y[i] = X[i];
        }
    }
}

int main() {
    int *h_X, *h_Y, *h_indices;    // Host arrays
    int *d_X, *d_Y, *d_indices;    // Device arrays
    int num_indices = 5;           // Number of indices to copy
    
    // Example indices to copy
    int example_indices[] = {10, 42, 100, 256, 500};
    
    // Allocate host memory
    h_X = (int*)malloc(N * sizeof(int));
    h_Y = (int*)malloc(N * sizeof(int));
    h_indices = (int*)malloc(num_indices * sizeof(int));
    
    // Initialize arrays
    for (int j = 0; j < N; j++) {
        h_X[j] = j * 3 + 7;  // Example values: 7, 10, 13, 16, ...
        h_Y[j] = -1;         // Initialize Y to -1 to see changes clearly
    }
    
    // Copy example indices
    for (int j = 0; j < num_indices; j++) {
        h_indices[j] = example_indices[j];
    }
    
    // Allocate device memory
    cudaMalloc(&d_X, N * sizeof(int));
    cudaMalloc(&d_Y, N * sizeof(int));
    cudaMalloc(&d_indices, num_indices * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_X, h_X, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, num_indices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid((num_indices + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    printf("Copying elements at %d indices from X to Y\n", num_indices);
    printf("Indices to copy: ");
    for (int j = 0; j < num_indices; j++) {
        printf("%d ", h_indices[j]);
    }
    printf("\n\n");
    
    printf("Before copying:\n");
    for (int j = 0; j < num_indices; j++) {
        int idx = h_indices[j];
        printf("X[%d] = %d, Y[%d] = %d\n", idx, h_X[idx], idx, h_Y[idx]);
    }
    
    copy_elements_kernel<<<grid, block>>>(d_X, d_Y, d_indices, num_indices);
    
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
    
    printf("\nAfter copying:\n");
    int success_count = 0;
    for (int j = 0; j < num_indices; j++) {
        int idx = h_indices[j];
        printf("X[%d] = %d, Y[%d] = %d", idx, h_X[idx], idx, h_Y[idx]);
        if (h_Y[idx] == h_X[idx]) {
            printf(" ✓\n");
            success_count++;
        } else {
            printf(" ✗\n");
        }
    }
    
    printf("\nResult: %d/%d elements copied successfully!\n", 
           success_count, num_indices);
    
    if (success_count == num_indices) {
        printf("SUCCESS: All elements copied correctly!\n");
    } else {
        printf("ERROR: Some elements not copied correctly!\n");
    }
    
    // Cleanup
    free(h_X);
    free(h_Y);
    free(h_indices);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_indices);
    
    return 0;
}