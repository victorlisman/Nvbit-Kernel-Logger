#include "nvbit_tool.h"
#include "nvbit.h"
#include "cuda.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cctype>
#include <unistd.h>


struct ArgInfo 
{
    std::string type;
    std::string io;
};

std::vector<ArgInfo> parse_signature(const std::string sig)
{
    std::vector<ArgInfo> result;
    size_t start = sig.find('(') + 1;
    size_t end = sig.find(')', start);
    std::string args = sig.substr(start, end - start);

    std::istringstream ss(args);
    std::string token;
    std::vector<std::string> types;

    while (std::getline(ss, token, ','))
        types.push_back(token);

    int non_ptr_count = 0;

    for (const auto& type : types) 
    {
        std::string trimmed_type = type;
        trimmed_type.erase(0, trimmed_type.find_first_not_of(" \t")); 
        trimmed_type.erase(trimmed_type.find_last_not_of(" \t") + 1); 
        trimmed_type.erase(std::remove_if(trimmed_type.begin(), trimmed_type.end(), ::isspace), trimmed_type.end());

        if (trimmed_type.find('*') == std::string::npos) 
        {
            non_ptr_count++;
        }
    }

    for (size_t i = 0; i < types.size(); ++i) 
    {
        std::string t = types[i];
        t.erase(0, t.find_first_not_of(" \t"));
        t.erase(t.find_last_not_of(" \t") + 1);     
        t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
        std::string cleaned_type = t;
        std::string io = "scalar";

        if (cleaned_type.find('*') != std::string::npos) 
        {
            if (types.size() == 1)
            {
                io = "output";
            }
            else if (types.size() > 1) 
            {
                if (i == types.size() - 1 - non_ptr_count)
                    io = "output";
                else
                    io = "input";
            }
        }

        result.push_back({ cleaned_type, io });
    }

    return result;
}

void decode_kernel_args(const std::string& signature, void** args) 
{
    auto argInfos = parse_signature(signature);
    std::cout << "Decoded arguments for kernel: " << signature << "\n";

    for (size_t i = 0; i < argInfos.size(); ++i) 
    {
        const auto& info = argInfos[i];
        std::string type = info.type;
        type.erase(0, info.type.find_first_not_of(" \t")); 
        type.erase(type.find_last_not_of(" \t") + 1); 
        type.erase(std::remove_if(type.begin(), type.end(), ::isspace), type.end());
        if (type == "int") 
        {
            int val = 0;
            cudaMemcpy(&val, args[i], sizeof(int), cudaMemcpyHostToHost);
            std::cout << "  [" << info.io << "] Arg " << i << " (int): " << val << "\n";
        }
        else if (type == "int*") 
        {
            int* dev_ptr = nullptr;
            cudaMemcpy(&dev_ptr, args[i], sizeof(int*), cudaMemcpyHostToHost);

            int sample = 0;
            if (cudaMemcpy(&sample, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess) 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (int*): " << dev_ptr << ", [0] = " << sample << "\n";
            } 
            else 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (int*): " << dev_ptr << " (unreadable)\n";
            }
        }
        else if (type == "float") 
        {
            float val = 0.0f;
            cudaMemcpy(&val, args[i], sizeof(float), cudaMemcpyHostToHost);
            std::cout << "  [" << info.io << "] Arg " << i << " (float): " << val << "\n";
        }
        else if (type == "float*") 
        {
            float* dev_ptr = nullptr;
            cudaMemcpy(&dev_ptr, args[i], sizeof(float*), cudaMemcpyHostToHost);

            float sample = 0.0f;
            if (cudaMemcpy(&sample, dev_ptr, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (float*): " << dev_ptr << ", [0] = " << sample << "\n";
            } 
            else 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (float*): " << dev_ptr << " (unreadable)\n";
            }
        }
        else if (type == "short*")
        {
            short* dev_ptr = nullptr;
            cudaMemcpy(&dev_ptr, args[i], sizeof(short*), cudaMemcpyHostToHost);

            short sample = 0;
            if (cudaMemcpy(&sample, dev_ptr, sizeof(short), cudaMemcpyDeviceToHost) == cudaSuccess) 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (short*): " << dev_ptr << ", [0] = " << sample << "\n";
            } 
            else 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (short*): " << dev_ptr << " (unreadable)\n";
            }
        }
        else if (type == "short") 
        {
            short val = 0;
            cudaMemcpy(&val, args[i], sizeof(short), cudaMemcpyHostToHost);
            std::cout << "  [" << info.io << "] Arg " << i << " (short): " << val << "\n";
        }
        else if (type == "bool")
        {
            bool val = false;
            cudaMemcpy(&val, args[i], sizeof(bool), cudaMemcpyHostToHost);
            std::cout << "  [" << info.io << "] Arg " << i << " (bool): " << (val ? "true" : "false") << "\n";
        }
        else if (type == "bool*") 
        {
            bool* dev_ptr = nullptr;
            cudaMemcpy(&dev_ptr, args[i], sizeof(bool*), cudaMemcpyHostToHost);

            bool sample = false;
            if (cudaMemcpy(&sample, dev_ptr, sizeof(bool), cudaMemcpyDeviceToHost) == cudaSuccess) 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (bool*): " << dev_ptr << ", [0] = " << (sample ? "true" : "false") << "\n";
            } 
            else 
            {
                std::cout << "  [" << info.io << "] Arg " << i << " (bool*): " << dev_ptr << " (unreadable)\n";
            }
        }
        else if (type == "void*") 
        {
            void* ptr = nullptr;
            cudaMemcpy(&ptr, args[i], sizeof(void*), cudaMemcpyHostToHost);
            std::cout << "  [" << info.io << "] Arg " << i << " (void*): " << ptr << "\n";
        }
        else 
        {
            void* ptr = nullptr;
            cudaMemcpy(&ptr, args[i], sizeof(void*), cudaMemcpyHostToHost);
            std::cout << "  [unknown] Arg " << i << " (" << info.type << "): " << ptr << "\n";
        }
    }
}

static std::string dump_sass_to_tmp(const std::vector<Instr*>& instrs, const std::string& kernel) 
{
    std::ostringstream fname;
    fname << "/home/vic/Dev/sass_ptx_parser/tmp/" << kernel << "-" << getpid() << ".sass";

    {
        std::ofstream out(fname.str(), std::ios::out | std::ios::trunc);
        for (Instr* i : instrs) 
        {
            out << i->getSass() << '\n';
        }
    }

    return fname.str();
}

static void launch_analyser(const std::string& sass_path, int grid, int block, CUdeviceptr base)
{
    std::ostringstream cmd;

    cmd << "python3 /home/vic/Dev/sass_ptx_parser/sass_ptx_parser/main.py "
        << std::quoted(sass_path) << ' '
        << "--grid "  << grid << ' '
        << "--block " << block << ' '
        << "--base 0x" << std::hex << base << ' '
        << "--json_out " << std::quoted(sass_path) << ".json"
        << " &";                 
    
    std::cout << "[DBG] Dumping sass to" << sass_path;
    std::system(cmd.str().c_str());
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params, CUresult* pStatus) 
{
    if (!is_exit && cbid == API_CUDA_cuLaunchKernel) 
    {
        CUDA_LAUNCH_PARAMS* launch_params = (CUDA_LAUNCH_PARAMS*)params;
        void** kernelParams = (void**)launch_params->kernelParams;

        std::string func_name = nvbit_get_func_name(ctx, launch_params->function, false);

        printf("Intercepted kernel launch: %s\n", func_name.c_str());
        printf("  gridDim = (%u, %u, %u)\n", launch_params->gridDimX, launch_params->gridDimY, launch_params->gridDimZ);
        printf("  blockDim = (%u, %u, %u)\n", launch_params->blockDimX, launch_params->blockDimY, launch_params->blockDimZ);

        decode_kernel_args(func_name, kernelParams);


        CUdeviceptr output_ptr = 0;
        auto argInfos = parse_signature(func_name);

        for (size_t i = 0; i < argInfos.size(); ++i) 
        {
            if (argInfos[i].io == "output") 
            {
                cudaMemcpy(&output_ptr, kernelParams[i], sizeof(CUdeviceptr), cudaMemcpyHostToHost);
                break;
            }
        }
        if (output_ptr == 0) 
        {
            std::cerr << "No output pointer found in kernel arguments.\n";
            return;
        }
        printf("  Output pointer: 0x%lx\n", (unsigned long)output_ptr);

        std::vector<Instr*> instrs = nvbit_get_instrs(ctx, launch_params->function);
        //std::string sass_path = dump_sass_to_tmp(instrs, func_name);
        //std::cout << "SASS dumped to: " << sass_path << '\n';
        //
        //launch_analyser(sass_path, 
        //                launch_params->gridDimX * launch_params->gridDimY * launch_params->gridDimZ,
        //                launch_params->blockDimX * launch_params->blockDimY * launch_params->blockDimZ,
        //                output_ptr);

    }
}