#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void nested_conditions_kernel(int condition1, int condition2, int* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 operates
    if (idx == 0) {
        if (condition1 > 42) {          // First condition: condition1 > 42
            if (condition2 > 42) {      // Nested condition: condition2 > 42
                *ptr = 42;              // Write if both conditions are true
            } else {
                int dummy = *ptr;       // Read if condition2 <= 42 (blind read)
            }
        }
        // Do nothing if condition1 <= 42
    }
}

int main() {
    int *h_data, *d_data;
    int initial_value = 100;
    int condition1 = 50;  // Change to test different paths
    int condition2 = 30;  // Change to test different paths
    
    // Allocate host memory
    h_data = (int*)malloc(sizeof(int));
    *h_data = initial_value;
    
    // Allocate device memory
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Initial value: %d\n", *h_data);
    printf("Condition1: %d\n", condition1);
    printf("Condition2: %d\n", condition2);
    printf("Result: %s\n", 
           (condition1 > 42) ? ((condition2 > 42) ? "Write 42" : "Read blindly") : "Do nothing");
    
    // Launch kernel with 1 block, 1 thread
    nested_conditions_kernel<<<1, 1>>>(condition1, condition2, d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value: %d\n", *h_data);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}