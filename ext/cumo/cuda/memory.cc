#include <ruby.h>
#include <cuda_runtime.h>
#include "cumo/cuda/memory.h"
#include "cumo/cuda/runtime.h"

char*
cumo_cuda_runtime_malloc(size_t size)
{
    void *ptr;
    cudaError_t status = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    cumo_cuda_runtime_check_status(status);
    return (char*)ptr;
}

void
cumo_cuda_runtime_free(char *ptr)
{
    cudaError_t status = cudaFree((void*)ptr);
    cumo_cuda_runtime_check_status(status);
}
