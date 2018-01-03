#ifndef CUMO_CUDA_RUNTIME_H
#define CUMO_CUDA_RUNTIME_H
#include "cumo/narray.h"
#include <cuda_runtime.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

extern VALUE cumo_cuda_eRuntimeError;

static inline void
cumo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_eRuntimeError, "%s (error=%d)", cudaGetErrorString(status), status);
    }
}

static inline bool
cumo_cuda_runtime_is_device_memory(void* ptr)
{
    struct cudaPointerAttributes attrs;
    cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
    return (status != cudaErrorInvalidValue);
}

static inline char*
cumo_cuda_runtime_malloc(size_t size)
{
    void *ptr;
    cudaError_t status = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    cumo_cuda_runtime_check_status(status);
    return (char*)ptr;
}

static inline void
cumo_cuda_runtime_free(char *ptr)
{
    cudaError_t status = cudaFree((void*)ptr);
    cumo_cuda_runtime_check_status(status);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif
