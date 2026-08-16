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

static inline int
cumo_cuda_runtime_get_device_count()
{
    int device_count;
    cumo_cuda_runtime_check_status(cudaGetDeviceCount(&device_count));
    return device_count;
}

static inline int
cumo_cuda_runtime_get_device()
{
    int device;
    cumo_cuda_runtime_check_status(cudaGetDevice(&device));
    return device;
}

static inline bool
cumo_cuda_runtime_is_device_memory(void* ptr)
{
    struct cudaPointerAttributes attrs;
    cudaError_t status;
    if (!ptr) { return false; }
    status = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset last error to success
    // Since CUDA 11 this succeeds for host memory as well and reports the kind
    // in attrs.type, so the status alone no longer tells the two apart.
    if (status != cudaSuccess) { return false; }
    return (attrs.type != cudaMemoryTypeUnregistered);
}

// A kernel cannot raise, so an integer division reports a zero divisor through
// this flag and the caller turns it into the exception ndloop expects. The
// buffer is pinned host memory the device writes through, not pool memory: a
// four-byte pool allocation lands in the same bin as a small output array, and
// the managed page then migrates back and forth once per operation.
int* cumo_cuda_runtime_divzero_flag_new(void);
bool cumo_cuda_runtime_divzero_flag_get(int *flag);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_RUNTIME_H */
