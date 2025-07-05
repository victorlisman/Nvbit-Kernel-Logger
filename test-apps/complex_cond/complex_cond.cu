#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void complex_cond_kernel(int* ptr, int condition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (condition > 0 && condition < 42) {  // Check if 0 < condition < 42
            *ptr = 42;                          // Write if condition is in range
        } else {
            int dummy = *ptr;                   // Read if condition is out of range (blind read)
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    int condition = 25;  // Change to test different paths (e.g., -1, 0, 42, 50)
    
    // Allocate host memory
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
    // Allocate device memory
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Condition: %d (%s)\n", condition, 
           (condition > 0 && condition < 42) ? "in range, will write 42" : "out of range, will read");
    
    // Launch kernel with 1 block, 1 thread
    complex_cond_kernel<<<1, 1>>>(d_data, condition);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}