#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define M 8
#define BLOCK_SIZE 256

__global__ void index_select_kernel(int* X, int* Y, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one index from Y
    if (idx < M) {
        int index = Y[idx];
        if (index >= 0 && index < N) {  // Bounds check
            output[idx] = X[index];
        }
    }
}

int main() {
    int *h_X, *h_Y, *h_output;    // Host arrays
    int *d_X, *d_Y, *d_output;    // Device arrays
    
    // Example indices to select from X
    int example_indices[] = {10, 42, 100, 5, 256, 500, 1, 999};
    
    // Allocate host memory
    h_X = (int*)malloc(N * sizeof(int));
    h_Y = (int*)malloc(M * sizeof(int));
    h_output = (int*)malloc(M * sizeof(int));
    
    // Initialize arrays
    for (int j = 0; j < N; j++) {
        h_X[j] = j * 5 + 3;  // Example values: 3, 8, 13, 18, 23, ...
    }
    
    // Copy example indices to Y
    for (int j = 0; j < M; j++) {
        h_Y[j] = example_indices[j];
        h_output[j] = -1;  // Initialize output to -1 to see changes
    }
    
    // Allocate device memory
    cudaMalloc(&d_X, N * sizeof(int));
    cudaMalloc(&d_Y, M * sizeof(int));
    cudaMalloc(&d_output, M * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_X, h_X, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, M * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    printf("Index Selection Operation\n");
    printf("Array X size: %d elements\n", N);
    printf("Index array Y size: %d elements\n", M);
    printf("Output array size: %d elements\n\n", M);
    
    printf("X array pattern: X[i] = i * 5 + 3\n");
    printf("Sample X values: X[0]=%d, X[1]=%d, X[2]=%d, X[10]=%d, X[42]=%d\n\n", 
           h_X[0], h_X[1], h_X[2], h_X[10], h_X[42]);
    
    printf("Index array Y (indices to select from X):\n");
    printf("Y = [");
    for (int j = 0; j < M; j++) {
        printf("%d", h_Y[j]);
        if (j < M-1) printf(", ");
    }
    printf("]\n\n");
    
    printf("Before index selection:\n");
    for (int j = 0; j < M; j++) {
        int idx = h_Y[j];
        if (idx >= 0 && idx < N) {
            printf("Y[%d] = %d -> X[%d] = %d\n", j, idx, idx, h_X[idx]);
        } else {
            printf("Y[%d] = %d -> INVALID INDEX\n", j, idx);
        }
    }
    
    index_select_kernel<<<grid, block>>>(d_X, d_Y, d_output);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, M * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nAfter index selection:\n");
    printf("Output array (X elements at indices from Y):\n");
    printf("Output = [");
    for (int j = 0; j < M; j++) {
        printf("%d", h_output[j]);
        if (j < M-1) printf(", ");
    }
    printf("]\n\n");
    
    // Verify the results
    printf("Verification:\n");
    int success_count = 0;
    for (int j = 0; j < M; j++) {
        int idx = h_Y[j];
        if (idx >= 0 && idx < N) {
            printf("Output[%d] = %d, Expected X[%d] = %d", j, h_output[j], idx, h_X[idx]);
            if (h_output[j] == h_X[idx]) {
                printf(" ✓\n");
                success_count++;
            } else {
                printf(" ✗\n");
            }
        } else {
            printf("Output[%d] = %d (invalid index %d)\n", j, h_output[j], idx);
        }
    }
    
    printf("\nResult: %d/%d elements selected correctly!\n", success_count, M);
    
    if (success_count == M) {
        printf("SUCCESS: All elements selected correctly!\n");
    } else {
        printf("ERROR: Some elements not selected correctly!\n");
    }
    
    // Cleanup
    free(h_X);
    free(h_Y);
    free(h_output);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_output);
    
    return 0;
}