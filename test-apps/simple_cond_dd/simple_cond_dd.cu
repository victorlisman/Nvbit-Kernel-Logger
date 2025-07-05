#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void simple_cond_kernel_dd(bool* condition, int* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (*condition) {  // Read condition from device memory
            *ptr = 42;     // Write if true
        } else {
            int dummy = *ptr;  // Read if false (blind read)
        }
    }
}

int main() {
    int *h_data, *d_data;
    bool *h_condition, *d_condition;
    int initial_value = 100;
    bool condition_value = true;  // Change to false to test read path
    
    // Allocate host memory
    h_data = (int*)malloc(sizeof(int));
    h_condition = (bool*)malloc(sizeof(bool));
    *h_data = initial_value;
    *h_condition = condition_value;
    
    // Allocate device memory
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_condition, sizeof(bool));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_condition, h_condition, sizeof(bool), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Condition: %s\n", *h_condition ? "true (will write 42)" : "false (will read)");
    
    // Launch kernel with 1 block, 1 thread
    simple_cond_kernel_dd<<<1, 1>>>(d_condition, d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_condition);
    free(h_data);
    free(h_condition);
    
    return 0;
}