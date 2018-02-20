#include <ruby.h>
#include <cuda_runtime.h>
#include "cumo/cuda/memory.h"
#include "cumo/cuda/runtime.h"

#include <vector>
#include <unordered_map>
#include <iostream>

static int kRoundSize = 512; // bytes
static std::unordered_map<void*, size_t> in_use;
static std::vector<std::vector<void*>> free_bins;

static inline int GetIndex(size_t size) { return size / kRoundSize; }

char*
cumo_cuda_runtime_malloc(size_t size)
{
    void *ptr;
    int index = GetIndex(size);
    if (index >= static_cast<int>(free_bins.size())) {
        free_bins.resize(index + 1);
    }
    if (free_bins[index].empty()) {
        cudaError_t status = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
        cumo_cuda_runtime_check_status(status);
        //std::cout << "malloc " << (size_t)(ptr) << " " << size << " " << index << std::endl;
        // TODO(sonots): If fails to allocate, once free all memory
        //cudaError_t status = cudaFree((void*)ptr);
        //cumo_cuda_runtime_check_status(status);
    } else {
        // TODO(sonots): atomic
        ptr = free_bins[index].back();
        free_bins[index].pop_back();
        //std::cout << "reuse  " << (size_t)(ptr) << " " << size << " " << index << std::endl;
    }
    in_use.emplace(ptr, size);
    return (char*)ptr;
}

void
cumo_cuda_runtime_free(char *ptr)
{
    size_t size = in_use[ptr];
    int index = GetIndex(size);
    // TODO(sonots): atomic
    //std::cout << "free   " << (size_t)(ptr) << " " << size << " " << index << std::endl;
    free_bins[index].emplace_back(ptr);
    in_use.erase(ptr);
}
