#include <stdio.h>
#include <stdint.h>
#include "nvbit_tool.h"
#include "nvbit.h"

void nvbit_at_init() {
    printf("[NVBIT] Launch logger initialized.\n");
}

void nvbit_at_term() {
    printf("[NVBIT] Tool exiting.\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (!is_exit && (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz)) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        CUfunction func = p->f;

        const char *kernel_name = nvbit_get_func_name(ctx, func);

        printf("[NVBIT] Launching kernel: %s\n", kernel_name);
        printf("         GridDim = (%d, %d, %d)\n", p->gridDimX, p->gridDimY, p->gridDimZ);
        printf("         BlockDim = (%d, %d, %d)\n", p->blockDimX, p->blockDimY, p->blockDimZ);

        void **args = (void **)p->kernelParams;
        if (args) {
            for (int i = 0; i < 20; i++) {
                if (args[i] == NULL) {
                    break;
                }
                printf("         Arg[%d] = %p\n", i, args[i]);
            }
        }
    }
}
