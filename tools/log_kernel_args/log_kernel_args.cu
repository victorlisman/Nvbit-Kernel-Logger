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
#include <cctype>

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
                {
                    io = "output";
                }
                else
                {
                    io = "input";
                }    
            }
        }

        result.push_back({ cleaned_type, io });
    }

    return result;
}

void decode_kernel_args(const std::string& signature, void** args) 
{
    auto argInfos = parse_signature(signature);
    std::cout << "=== KERNEL ARGUMENTS ===" << std::endl;
    std::cout << "Kernel: " << signature << std::endl;
    std::cout << "Number of arguments: " << argInfos.size() << std::endl;
    std::cout << "------------------------" << std::endl;

    for (size_t i = 0; i < argInfos.size(); ++i) 
    {
        const auto& info = argInfos[i];
        std::string type = info.type;
        type.erase(0, info.type.find_first_not_of(" \t")); 
        type.erase(type.find_last_not_of(" \t") + 1); 
        type.erase(std::remove_if(type.begin(), type.end(), ::isspace), type.end());
        
        std::cout << "Arg[" << i << "] Type: " << type << " | Direction: [" << info.io << "] | Address: " << args[i] << std::endl;
    }
    std::cout << "========================" << std::endl << std::endl;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params, CUresult* pStatus) 
{
    if (!is_exit && cbid == API_CUDA_cuLaunchKernel) 
    {
        CUDA_LAUNCH_PARAMS* launch_params = (CUDA_LAUNCH_PARAMS*)params;
        void** kernelParams = (void**)launch_params->kernelParams;

        std::string func_name = nvbit_get_func_name(ctx, launch_params->function, false);
        
        for (int i = 0; i < 10000; i++)
        {
            if (kernelParams[i] == nullptr) {
                break;
            }
            printf("arg[%d] = %p\n", i, kernelParams[i]);
        }
    }
}



