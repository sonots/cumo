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

// How many times the whole device has been seen to settle, which is what a host
// read of managed memory needs. One settling covers every kernel and copy issued
// before it, so code that recorded the count when it queued work can skip a wait
// the moment the count has moved on. Only cumo_cuda_runtime_device_synchronize
// advances it, and a count that is behind costs a wait rather than correctness.
extern uint64_t cumo_cuda_sync_epoch;

static inline void
cumo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_eRuntimeError, "%s (error=%d)", cudaGetErrorString(status), status);
    }
}

void cumo_cuda_runtime_return_scratch(char *ptr, int wait_for_stream, cudaError_t *status);

static inline void
cumo_cuda_runtime_device_synchronize(void)
{
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    cumo_cuda_sync_epoch++;
}

// Asking costs less than half of waiting, and there is nothing to wait for
// whenever the block stayed off the device. Neither answer advances the settle
// count: stream 0 going quiet is not the whole device settling, and a host read
// of managed memory needs the latter where concurrentManagedAccess is 0.
static inline int
cumo_cuda_runtime_sync_if_busy(void)
{
    if (cudaStreamQuery(0) == cudaSuccess) {
        return 0;
    }
    cumo_cuda_runtime_check_status(cudaStreamSynchronize(0));
    return 1;
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

// A kernel cannot raise, so it reports a bad argument through this flag and the
// caller turns it into an exception. The buffer is pinned host memory the device
// writes through, not pool memory: a four-byte pool allocation lands in the same
// bin as a small output array, and the managed page then migrates back and forth
// once per operation.
int* cumo_cuda_runtime_error_flag_new(void);
bool cumo_cuda_runtime_error_flag_get(int *flag);
size_t* cumo_cuda_runtime_error_item_new(void);
// The same buffers without the reset, so a loop that launches once per row can
// reset before it and read after it rather than synchronizing every row.
int* cumo_cuda_runtime_error_flag_ptr(void);
size_t* cumo_cuda_runtime_error_item_ptr(void);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_RUNTIME_H */
