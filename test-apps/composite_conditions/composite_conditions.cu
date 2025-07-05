#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void composite_conditions_kernel(bool condition1, bool condition2, int* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (condition1 || condition2) {  // Check if either condition is true
            *ptr = 42;                   // Write if the disjunction is true
        } else {
            int dummy = *ptr;            // Read if the disjunction is false (blind read)
        }
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    bool condition1 = true;  // Change to test different paths
    bool condition2 = false; // Change to test different paths
    
    // Allocate host memory
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
    // Allocate device memory
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Condition1: %s\n", condition1 ? "true" : "false");
    printf("Condition2: %s\n", condition2 ? "true" : "false");
    printf("Disjunction: %s\n", (condition1 || condition2) ? "true (will write 42)" : "false (will read)");
    
    // Launch kernel with 1 block, 1 thread
    composite_conditions_kernel<<<1, 1>>>(condition1, condition2, d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}