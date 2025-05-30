#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


__global__ void vecAdd(float *a, float *b, float *c, int n) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char *argv[]) 
{
    int n = 1024; 
    if (argc > 1) n = atoi(argv[1]);

    float *h_a;
    float *h_b;
    float *h_c;

    float *d_a;
    float *d_b;
    float *d_c;

    size_t bytes = n * sizeof(float);

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;

    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
        h_c[i] = 0;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    blockSize = 256;

    gridSize = (int)ceil((float)n / blockSize);
    gridSize = 4;

    CUDA_SAFECALL((vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    float sum = 0;
    for (i = 0; i < n; i++) sum += h_c[i];
        printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
