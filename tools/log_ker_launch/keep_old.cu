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
        printf("args = %p\n", args);
        if (args) {
            for (int i = 0; i < 20; i++) {
                if (args[i] == NULL) {
                    break;
                }
                //void *arg_ptr = NULL;

                // safely attempt to read the pointer, without dereferencing it
                //if (cudaMemcpy(&arg_ptr, &args[i], sizeof(void*), cudaMemcpyHostToHost) != cudaSuccess)
                    break;

                //if (arg_ptr == NULL)
                //    break;

                //printf("         Arg[%d] = %p (ptr to device ptr)\n", i, arg_ptr);
                //printf("         Arg[%d] = %p\n", i, args[i]);
                printf("         Arg[%d] = %p", i, args[i]);

                // Try to interpret as a pointer to a device pointer
                CUdeviceptr dev_ptr = 0;
                if (cudaMemcpy(&dev_ptr, args[i], sizeof(CUdeviceptr), cudaMemcpyHostToHost) == cudaSuccess) {
                    printf(" --> device ptr: 0x%lx\n", (unsigned long)dev_ptr);
                } else {
                    printf(" --> (not a device pointer or failed to dereference)\n");
                }
            }
        } else {
            printf("         No kernel arguments found (args == NULL)\n");
        }
    }
}
