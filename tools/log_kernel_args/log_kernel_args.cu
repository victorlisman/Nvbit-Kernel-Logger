#include "cuda_v1_shim.h"
#include <stdio.h>
#include <stdint.h>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <unordered_map>

#include "nvbit_tool.h"
#include "nvbit.h"

#ifdef DEBUG
#define DBG(msg, ...)                                                      \
    printf("[DBG] %s:%d " msg "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    fflush(stdout);
#else
#define DBG(msg, ...)
#endif

struct KernelArgLog {
    CUcontext                    ctx;
    std::string                  kernel_name;
    std::vector<CUdeviceptr>     dev_ptrs;
    std::vector<size_t>          sizes;      // supply real sizes if known
};

static std::mutex                log_mtx;
static std::condition_variable   cv;
static std::vector<KernelArgLog> pending;
static bool                      keep_running = true;
static std::unordered_map<CUdeviceptr, size_t> alloc_size;
static std::unordered_map<CUdeviceptr, size_t> memcpy_size;

static void mem_dumper() {
    DBG("[DBG] mem_dumper thread started");
    while (true) {
        KernelArgLog job;

        {
            std::unique_lock<std::mutex> lk(log_mtx);
            cv.wait(lk, []{ return !pending.empty() || !keep_running; });
            if (!keep_running && pending.empty()) {
                DBG("[DBG] mem_dumper thread exiting");
                return;  
            }    // graceful exit

            job = std::move(pending.back());
            pending.pop_back();
        }
        
        DBG("processing job for %s", job.kernel_name.c_str());
        cuCtxSetCurrent(job.ctx);

        for (size_t i = 0; i < job.dev_ptrs.size(); ++i) {
            size_t want = job.sizes[i];
            if (!want) { DBG("arg[%zu] size unknown – skip", i); continue; }


            unsigned mem_type = 0;
            if (cuPointerGetAttribute(&mem_type,
                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE, job.dev_ptrs[i]) != CUDA_SUCCESS
                || mem_type != CU_MEMORYTYPE_DEVICE)
                continue;                       // not a device buffer ‑ skip

            std::vector<uint8_t> h(job.sizes[i]);
            if (cuMemcpyDtoH(h.data(), job.dev_ptrs[i], h.size()) == CUDA_SUCCESS) {
                printf("[MEMDUMP] %s arg[%zu] 0x%llx :", job.kernel_name.c_str(),
                       i, (unsigned long long)job.dev_ptrs[i]);
                for (size_t b = 0; b < h.size() && b < 16; ++b) printf(" %02x", h[b]);
                puts("");
            }
        }
    }
}

static std::thread dumper_thread;

extern "C" void nvbit_at_init() {
    puts("[NVBIT] arg-logger initialised");
    dumper_thread = std::thread(mem_dumper);
}

extern "C" void nvbit_at_term() {
    {   std::lock_guard<std::mutex> lk(log_mtx);
        keep_running = false;
    }
    cv.notify_all();              // wake thread
    dumper_thread.join();         // wait – prevents segfault on exit
    DBG("[DBG] dumper thread joined");
    puts("[NVBIT] arg‑logger exiting");
}

extern "C" void nvbit_at_cuda_event(CUcontext ctx,
                                    int is_exit,
                                    nvbit_api_cuda_t cbid,
                                    const char*,
                                    void* params,
                                    CUresult*) {
    
    if (!is_exit && (cbid == API_CUDA_cuMemAlloc_v2 || cbid == API_CUDA_cuMemAllocManaged))
    {
        auto *pa = (cuMemAlloc_v2_params*)params;
        CUdeviceptr gpu = *pa->dptr;
        alloc_size[gpu] = pa->bytesize;
        alloc_size[*pa->dptr] = pa->bytesize;
        DBG("alloc 0x%llx %zu B", (unsigned long long)gpu, pa->bytesize);
        return;
    }

    if (!is_exit && (cbid == API_CUDA_cuMemcpyHtoD_v2 || cbid == API_CUDA_cuMemcpyHtoDAsync_v2)) {
        auto *pm = (cuMemcpyHtoD_v2_params*)params;
        memcpy_size[pm->dstDevice] = pm->ByteCount;
        DBG("memcpy H→D 0x%llx %zu B", (unsigned long long)pm->dstDevice, pm->ByteCount);
        return;
    }
        /* ------------ memcpy tracking ---------------------------------- */
    if (is_exit) return;
    if (cbid != API_CUDA_cuLaunchKernel &&
        cbid != API_CUDA_cuLaunchKernel_ptsz) return;
    
    DBG("[DBG] cuLaunchKernel intercepted");
    auto* p = (cuLaunchKernel_params*)params;

    KernelArgLog job;
    job.ctx         = ctx;
    job.kernel_name = nvbit_get_func_name(ctx, p->f);

    void** kparams = (void**)p->kernelParams;
    if (kparams)
        for (int i = 0; i < 64 && kparams[i]; ++i) {
            uintptr_t host_ptr = (uintptr_t)kparams[i];

            if (host_ptr < 0x100000000000ULL)
            {
                DBG("[DBG] args[%d] looks like scalar - skipping", i);
                break;
            }

            CUdeviceptr dev_ptr = 0;
            memcpy(&dev_ptr, kparams[i], sizeof(CUdeviceptr));

            if (dev_ptr < 0x700000000000ULL || dev_ptr > 0x7fffffffffffULL) {
                DBG("  arg[%d] 0x%llx outside GPU range - skip",
                    i, (unsigned long long)dev_ptr);
                break;
            }

            CUcontext dummy;
            if (cuPointerGetAttribute(&dummy,
                    CU_POINTER_ATTRIBUTE_CONTEXT, dev_ptr) != CUDA_SUCCESS) {
                DBG("arg[%d] unowned pointer – stop scan", i);
                break;
            }

            unsigned mem_type = 0;

            if (cuPointerGetAttribute(&mem_type,
                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dev_ptr) != CUDA_SUCCESS
                || mem_type != CU_MEMORYTYPE_DEVICE) {
                    DBG("[DBG] args[%d] looks like host pointer - skipping", i);
                    break;                       // not a device buffer ‑ skip
                }
                /* determine size */
                size_t sz = 0;
                auto m = memcpy_size.find(dev_ptr);
                if (m != memcpy_size.end()) sz = m->second;
                else {
                    auto a = alloc_size.find(dev_ptr);
                    if (a != alloc_size.end()) sz = a->second;
                }

                DBG("arg[%d] dev=0x%llx size=%zu queued", i,
                    (unsigned long long)dev_ptr, sz);

            job.dev_ptrs.push_back(dev_ptr);
            job.sizes.push_back(sz ? sz : 64);       
        }

    {
        std::lock_guard<std::mutex> lk(log_mtx);
        pending.emplace_back(std::move(job));
    }
    cv.notify_one();                // wake dumper thread
    DBG("queued");
}