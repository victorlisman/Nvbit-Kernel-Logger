#include <cuda_runtime.h>
#include <iostream>

__global__ void readWriteKernel(float* x, float* y) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x[idx];
}

int main() 
{
    float* d_y;
    int N = 1024;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_y, size);
    float* h_x = new float[N];
    float* h_y = new float[N];

    for (int i = 0; i < N; ++i) 
    {
        h_x[i] = static_cast<float>(i);
    }

    float* d_x;
    cudaMalloc(&d_x, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    printf("blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);

    readWriteKernel<<<gridDim, blockDim>>>(d_x, d_y);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; ++i) 
    {
        void *write_addr = static_cast<void*>(d_y + i);
        //std::cout << "h_y[" << i << "] = " << h_y[i] << " written to " << write_addr << std::endl;
        if (h_y[i] != h_x[i]) 
        {
            //std::cout << "Mismatch at index " << i << ": " << h_y[i] << " != " << h_x[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) 
    {
        std::cout << "Success: all values match!" << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;

    return 0;
}