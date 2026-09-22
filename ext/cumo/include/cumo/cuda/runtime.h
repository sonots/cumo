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

cudaStream_t cumo_cuda_stream(void);
void cumo_cuda_stream_set(cudaStream_t stream);
// The stream an Integer handle names: 0, or one this process created.
cudaStream_t cumo_cuda_stream_get(VALUE stream);

// How many times the whole device has been seen to settle, which is what a host
// read of managed memory needs. One settling covers every kernel and copy issued
// before it, so code that recorded the count when it queued work can skip a wait
// the moment the count has moved on. cumo_cuda_runtime_device_synchronize
// advances it, and so does a host read under a stream of the caller's, and a
// count that is behind costs a wait rather than correctness.
extern uint64_t cumo_cuda_sync_epoch;

// How many times device memory has been written from here: every kernel
// launch and every copy into it counts. A host loop that yields to Ruby
// reads from a staged copy, and a count that moved on tells it to stage
// again before the next row.
extern uint64_t cumo_cuda_launch_epoch;

static inline void
cumo_cuda_runtime_note_device_write(void)
{
    cumo_cuda_launch_epoch++;
}

// A pinned host buffer for reading device memory back. The host reading
// managed memory in place faults the page over, and a small block shares
// its page with other live blocks, so the page goes back and forth once per
// read. A copy into pinned memory reads the block where it is instead. One
// buffer per thread is kept for the next read; a nested or oversized request
// gets one of its own.
typedef struct {
    char  *ptr;
    size_t size;
    bool   cached;
} cumo_cuda_stage_t;

void cumo_cuda_runtime_stage_alloc(cumo_cuda_stage_t *stage, size_t bytes);
void cumo_cuda_runtime_stage_free(cumo_cuda_stage_t *stage);

// A failure stays as the runtime's last error until something reads it,
// and the next kernel launch would read it as its own, so it is cleared
// before it is raised.
static inline void
cumo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        cudaGetLastError();
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
// count: one stream going quiet is not the whole device settling, and a host read
// of managed memory needs the latter where concurrentManagedAccess is 0.
// Under a stream of the caller's, what the host is about to read may have
// been written on the null stream as well, and nothing records which, so
// unless both are idle the whole device is waited for. The idle check is what
// keeps an each, which asks per element, from waiting per element.
static inline int
cumo_cuda_runtime_streams_idle(void)
{
    return cudaStreamQuery(0) == cudaSuccess && cudaStreamQuery(cumo_cuda_stream()) == cudaSuccess;
}

static inline int
cumo_cuda_runtime_sync_if_busy(void)
{
    if (cumo_cuda_stream() != 0) {
        if (cumo_cuda_runtime_streams_idle()) {
            return 0;
        }
        cumo_cuda_runtime_device_synchronize();
        return 1;
    }
    if (cudaStreamQuery(0) == cudaSuccess) {
        return 0;
    }
    cumo_cuda_runtime_check_status(cudaStreamSynchronize(0));
    return 1;
}

// A host read of device memory has to come after the work queued on the
// current stream, which a plain cudaMemcpy only does for the legacy one, and
// under a stream of the caller's after every stream, as above. The
// destination is pinned, so the copy reads the source where it is.
static inline cudaError_t
cumo_cuda_runtime_memcpy_to_pinned(void *dst, const void *src, size_t bytes)
{
    cudaError_t status;
    if (cumo_cuda_stream() != 0 && !cumo_cuda_runtime_streams_idle()) {
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) { return status; }
        cumo_cuda_sync_epoch++;
    }
    status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cumo_cuda_stream());
    if (status != cudaSuccess) { return status; }
    return cudaStreamSynchronize(cumo_cuda_stream());
}

// The same into pageable memory, through the staging buffer above.
cudaError_t cumo_cuda_runtime_memcpy_to_host(void *dst, const void *src, size_t bytes);

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
