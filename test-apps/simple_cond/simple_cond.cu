#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void simple_cond_kernel(int* ptr, bool condition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (condition) {
            *ptr = 42;  // Write if true
        } else {
            int dummy = *ptr;  // Read if false (blind read)
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    bool condition = true;  // Change to false to test read path
    
    // Allocate host memory
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
    // Allocate device memory
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Condition: %s\n", condition ? "true (will write 42)" : "false (will read)");
    
    // Launch kernel with 1 block, 1 thread
    simple_cond_kernel<<<1, 1>>>(d_data, condition);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}