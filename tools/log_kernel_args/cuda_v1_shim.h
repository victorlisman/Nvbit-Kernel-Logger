#ifndef CUDA_V1_SHIM_H
#define CUDA_V1_SHIM_H

#include <cuda.h>

typedef CUdeviceptr            CUdeviceptr_v1;
typedef CUDA_MEMCPY2D          CUDA_MEMCPY2D_v1;
typedef CUDA_MEMCPY3D          CUDA_MEMCPY3D_v1;
typedef CUDA_ARRAY_DESCRIPTOR  CUDA_ARRAY_DESCRIPTOR_v1;
typedef CUDA_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR_v1;

#endif