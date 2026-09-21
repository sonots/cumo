#include <ruby.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cumo/cuda/runtime.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/template_kernel.h"
#include "cumo/cuda/handle.h"
#include "cumo/intern.h"

VALUE cumo_cuda_eRuntimeError;
VALUE cumo_cuda_mRuntime;
uint64_t cumo_cuda_sync_epoch = 0;

static __thread cudaStream_t current_stream = 0;
static cumo_cuda_handle_set_t streams;
static cumo_cuda_handle_set_t events;
// How many threads have each stream current, so that none of them can
// destroy a stream another is launching on. The value is the count.
static cumo_cuda_handle_set_t in_use;
// Pinned host buffers this process allocated, each with its size as the value.
static cumo_cuda_handle_set_t pinned;

static void
in_use_add(cudaStream_t stream, long delta)
{
    st_data_t n = 0;
    if (stream == 0) { return; }
    rb_nativethread_lock_lock(&in_use.lock);
    st_lookup(in_use.table, (st_data_t)stream, &n);
    n = (st_data_t)((long)n + delta);
    if (n == 0) { st_data_t key = (st_data_t)stream; st_delete(in_use.table, &key, 0); }
    else { st_insert(in_use.table, (st_data_t)stream, n); }
    rb_nativethread_lock_unlock(&in_use.lock);
}

static int
in_use_p(cudaStream_t stream)
{
    int found;
    rb_nativethread_lock_lock(&in_use.lock);
    found = st_lookup(in_use.table, (st_data_t)stream, 0);
    rb_nativethread_lock_unlock(&in_use.lock);
    return found;
}

cudaStream_t
cumo_cuda_stream(void)
{
    return current_stream;
}

void
cumo_cuda_stream_set(cudaStream_t stream)
{
    if (stream == current_stream) { return; }
    in_use_add(current_stream, -1);
    in_use_add(stream, 1);
    current_stream = stream;
}
#define eRuntimeError cumo_cuda_eRuntimeError
#define mRuntime cumo_cuda_mRuntime

#define check_status(status) (cumo_cuda_runtime_check_status((status)))

// Called right after a <<<>>> launch. cudaGetLastError() reports errors the
// launch itself was rejected for; a fault while the kernel runs is asynchronous
// and still surfaces at a later call.
void
cumo_cuda_runtime_check_kernel_launch(void)
{
    check_status(cudaGetLastError());
}

// A pointer given twice would be freed twice, and the second free would reach
// past the pool to a block it still owns, so each one goes back once.
static void
release_held(char *p[5])
{
    int i, j;
    for (i = 0; i < 5; ++i) {
        if (p[i] == NULL) { continue; }
        for (j = 0; j < i; ++j) {
            if (p[j] == p[i]) { break; }
        }
        if (j == i) { cumo_cuda_runtime_free_no_raise(p[i]); }
    }
}

// What a launcher holding scratch has to call instead of the plain check, for a
// status it already took out of the slot: a library call in the window takes it
// first otherwise, and whoever reads it first is the only one who sees it.
// Raising is a longjmp, so the frees written below the check never run, and
// unwinding frees through the form that cannot raise so one failure cannot hide
// another.
void
cumo_cuda_runtime_check_taken_status_holding(int status, char *p0, char *p1, char *p2, char *p3, char *p4)
{
    if ((cudaError_t)status != cudaSuccess) {
        char *held[5];
        held[0] = p0; held[1] = p1; held[2] = p2; held[3] = p3; held[4] = p4;
        release_held(held);
    }
    check_status((cudaError_t)status);
}

// The slot is read once here rather than peeked and read again: the frees below
// can put a status of their own in it, and the launch is the one to report.
void
cumo_cuda_runtime_check_kernel_launch_holding(char *p0, char *p1, char *p2, char *p3, char *p4)
{
    cumo_cuda_runtime_check_taken_status_holding((int)cudaGetLastError(), p0, p1, p2, p3, p4);
}

// How scratch goes back from a handler that may be unwinding: a free that
// raises there replaces the exception being carried, and leaves every free
// below it unrun. wait_for_stream asks for the wait a caller needs when the
// buffer is still being read, since the pool hands a freed chunk straight out.
void
cumo_cuda_runtime_return_scratch(char *ptr, int wait_for_stream, cudaError_t *status)
{
    if (wait_for_stream) {
        cudaError_t wait = cudaStreamSynchronize(cumo_cuda_stream());
        if (status != NULL && *status == cudaSuccess) { *status = wait; }
    }
    if (ptr != NULL) { cumo_cuda_runtime_free_no_raise(ptr); }
}

int*
cumo_cuda_runtime_error_flag_ptr(void)
{
    static int *flag = NULL;

    if (flag == NULL) {
        check_status(cudaHostAlloc((void**)&flag, sizeof(int),
                                   cudaHostAllocMapped | cudaHostAllocPortable));
    }
    return flag;
}

int*
cumo_cuda_runtime_error_flag_new(void)
{
    int *flag = cumo_cuda_runtime_error_flag_ptr();
    *flag = 0;
    return flag;
}

bool
cumo_cuda_runtime_error_flag_get(int *flag)
{
    cumo_cuda_runtime_device_synchronize();
    return (*flag != 0);
}

// The value a kernel is rejecting, so the message the caller raises can name it.
// Read it only once the flag says there was one; the flag's read synchronizes.
size_t*
cumo_cuda_runtime_error_item_ptr(void)
{
    static size_t *item = NULL;

    if (item == NULL) {
        check_status(cudaHostAlloc((void**)&item, sizeof(size_t),
                                   cudaHostAllocMapped | cudaHostAllocPortable));
    }
    return item;
}

size_t*
cumo_cuda_runtime_error_item_new(void)
{
    size_t *item = cumo_cuda_runtime_error_item_ptr();
    *item = 0;
    return item;
}

///////////////////////////////////////////
// Version Management
///////////////////////////////////////////

/*
  Returns the CUDA driver version.

  @return [Integer] Returns the CUDA driver version.
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g8a06ee14a0551606b7c780084d5564ab
 */
static VALUE
rb_cudaDriverGetVersion(VALUE self)
{
    int _version;
    cudaError_t status;

    status = cudaDriverGetVersion(&_version);

    check_status(status);
    return INT2NUM(_version);
}

/*
  Returns the CUDA Runtime version.

  @return [Integer] Returns the CUDA Runtime version.
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g0e3952c7802fd730432180f1f4a6cdc6
 */
static VALUE
rb_cudaRuntimeGetVersion(VALUE self)
{
    int _version;
    cudaError_t status;

    status = cudaRuntimeGetVersion(&_version);

    check_status(status);
    return INT2NUM(_version);
}

/////////////////////////////////////////
// Device and context operations
/////////////////////////////////////////

/*
  Returns which device is currently being used.

  @return [Integer] Returns the device on which the active host thread executes the device code.
  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78
 */
static VALUE
rb_cudaGetDevice(VALUE self)
{
    return INT2NUM(cumo_cuda_runtime_get_device());
}

/*
  Returns information about the device.

  @param [Integer] attrib Device attribute to query
  @param [Integer] device Device number to query
  @return [Integer] Returned device attribute value
  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151
 */
static VALUE
rb_cudaDeviceGetAttributes(VALUE self, VALUE attrib, VALUE device)
{
    int _attrib = NUM2INT(attrib);
    int _device = NUM2INT(device);
    int _ret;
    cudaError_t status;

    status = cudaDeviceGetAttribute(&_ret, _attrib, _device);

    check_status(status);
    return INT2NUM(_ret);
}

/*
  Returns the number of compute-capable devices.

  @return [Integer] Returns the number of devices with compute capability greater or equal to 2.0
  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f
 */
static VALUE
rb_cudaGetDeviceCount(VALUE self)
{
    return INT2NUM(cumo_cuda_runtime_get_device_count());
}

/*
  Set device to be used for GPU executions.

  @param [Integer] device Device on which the active host thread should execute the device code.
  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb
 */
static VALUE
rb_cudaSetDevice(VALUE self, VALUE device)
{
    int _device = NUM2INT(device);
    cudaError_t status;

    status = cudaSetDevice(_device);

    check_status(status);
    return Qnil;
}

/*
  Returns whether a device can directly access the memory of another.

  @param [Integer] device Device from which the memory would be accessed.
  @param [Integer] peer_device Device whose memory would be accessed.
  @return [Integer] 1 if it can, 0 if it cannot
  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g4db0d04e44995d5c1c34be4ecc863f22
 */
static VALUE
rb_cudaDeviceCanAccessPeer(VALUE self, VALUE device, VALUE peer_device)
{
    int _device = NUM2INT(device);
    int _peer_device = NUM2INT(peer_device);
    int can_access = 0;
    cudaError_t status;

    status = cudaDeviceCanAccessPeer(&can_access, _device, _peer_device);

    check_status(status);
    return INT2NUM(can_access);
}

/*
  Creates a stream. cudaStreamNonBlocking makes one that does not wait for
  the legacy stream 0.

  @param [Integer] flags CUDA_STREAM_DEFAULT or CUDA_STREAM_NON_BLOCKING
  @return [Integer] the stream handle
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamCreateWithFlags(VALUE self, VALUE flags)
{
    cudaStream_t stream;
    cumo_cuda_runtime_check_status(cudaStreamCreateWithFlags(&stream, NUM2UINT(flags)));
    cumo_cuda_handle_set_add(&streams, (size_t)stream);
    return SIZET2NUM((size_t)stream);
}

/*
  Destroys a stream. A stream that is current in any thread cannot be
  destroyed.

  @param [Integer] stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamDestroy(VALUE self, VALUE stream)
{
    if (in_use_p((cudaStream_t)NUM2SIZET(stream))) {
        rb_raise(rb_eArgError, "a stream that is current in a thread cannot be destroyed");
    }
    cumo_cuda_runtime_check_status(cudaStreamDestroy((cudaStream_t)cumo_cuda_handle_take(&streams, stream, "cudaStream_t")));
    return Qnil;
}

cudaStream_t
cumo_cuda_stream_get(VALUE v)
{
    if (NUM2SIZET(v) == 0) { return 0; }
    return (cudaStream_t)cumo_cuda_handle_get(&streams, v, "cudaStream_t");
}

static cudaEvent_t
event_get(VALUE v)
{
    return (cudaEvent_t)cumo_cuda_handle_get(&events, v, "cudaEvent_t");
}

// cudaSuccess and cudaErrorNotReady are the two answers of a query; anything
// else is an error.
static VALUE
query_result(cudaError_t status)
{
    if (status == cudaSuccess) { return Qtrue; }
    if (status == cudaErrorNotReady) { return Qfalse; }
    cumo_cuda_runtime_check_status(status);
    return Qfalse;
}

/*
  Returns whether everything queued on a stream has finished.

  @param [Integer] stream
  @return [Boolean]
 */
static VALUE
rb_cudaStreamQuery(VALUE self, VALUE stream)
{
    return query_result(cudaStreamQuery(cumo_cuda_stream_get(stream)));
}

/*
  Makes everything queued on a stream after this call wait for an event.

  @param [Integer] stream
  @param [Integer] event
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamWaitEvent(VALUE self, VALUE stream, VALUE event)
{
    cumo_cuda_runtime_check_status(cudaStreamWaitEvent(cumo_cuda_stream_get(stream), event_get(event), 0));
    return Qnil;
}

/*
  Creates an event.

  @param [Integer] flags CUDA_EVENT_DEFAULT, CUDA_EVENT_BLOCKING_SYNC and CUDA_EVENT_DISABLE_TIMING, or'ed
  @return [Integer] the event handle
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaEventCreateWithFlags(VALUE self, VALUE flags)
{
    cudaEvent_t event;
    cumo_cuda_runtime_check_status(cudaEventCreateWithFlags(&event, NUM2UINT(flags)));
    cumo_cuda_handle_set_add(&events, (size_t)event);
    return SIZET2NUM((size_t)event);
}

/*
  Destroys an event.

  @param [Integer] event
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaEventDestroy(VALUE self, VALUE event)
{
    cumo_cuda_runtime_check_status(cudaEventDestroy((cudaEvent_t)cumo_cuda_handle_take(&events, event, "cudaEvent_t")));
    return Qnil;
}

/*
  Records an event on a stream, after everything queued on it so far.

  @param [Integer] event
  @param [Integer] stream 0 is the legacy default stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaEventRecord(VALUE self, VALUE event, VALUE stream)
{
    cumo_cuda_runtime_check_status(cudaEventRecord(event_get(event), cumo_cuda_stream_get(stream)));
    return Qnil;
}

/*
  Waits for an event to be reached.

  @param [Integer] event
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaEventSynchronize(VALUE self, VALUE event)
{
    cumo_cuda_runtime_check_status(cudaEventSynchronize(event_get(event)));
    return Qnil;
}

/*
  Returns whether an event has been reached.

  @param [Integer] event
  @return [Boolean]
 */
static VALUE
rb_cudaEventQuery(VALUE self, VALUE event)
{
    return query_result(cudaEventQuery(event_get(event)));
}

/*
  Returns the milliseconds between two recorded events.

  @param [Integer] start
  @param [Integer] stop
  @return [Float] milliseconds
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaEventElapsedTime(VALUE self, VALUE start, VALUE stop)
{
    float ms = 0;
    cumo_cuda_runtime_check_status(cudaEventElapsedTime(&ms, event_get(start), event_get(stop)));
    return DBL2NUM((double)ms);
}

/*
  Waits for everything queued on a stream. 0 is the legacy default stream.

  @param [Integer] stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamSynchronize(VALUE self, VALUE stream)
{
    cumo_cuda_runtime_check_status(cudaStreamSynchronize(cumo_cuda_stream_get(stream)));
    return Qnil;
}

/*
  Returns the stream Cumo launches its kernels and copies on in this thread.
  0 until set.

  @return [Integer]
 */
static VALUE
rb_current_stream(VALUE self)
{
    return SIZET2NUM((size_t)current_stream);
}

/*
  Sets the stream Cumo launches its kernels and copies on in this thread.

  @param [Integer] stream a stream this process created, or 0
  @return [Integer] the stream
 */
static VALUE
rb_current_stream_set(VALUE self, VALUE stream)
{
    cumo_cuda_stream_set(cumo_cuda_stream_get(stream));
    return stream;
}

static char*
pinned_get(VALUE v, size_t *size)
{
    size_t handle = NUM2SIZET(v);
    st_data_t n = 0;
    int found;
    rb_nativethread_lock_lock(&pinned.lock);
    found = st_lookup(pinned.table, (st_data_t)handle, &n);
    rb_nativethread_lock_unlock(&pinned.lock);
    if (!found) {
        rb_raise(rb_eArgError, "not a live pinned host buffer");
    }
    if (size) { *size = (size_t)n; }
    return (char*)handle;
}

/*
  Allocates page-locked host memory, which a copy to or from the device can
  be asynchronous with.

  @param [Integer] bytes
  @param [Integer] flags CUDA_HOST_ALLOC_DEFAULT, PORTABLE, MAPPED and WRITE_COMBINED, or'ed
  @return [Integer] the buffer handle
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaHostAlloc(VALUE self, VALUE bytes, VALUE flags)
{
    size_t _bytes = NUM2SIZET(bytes);
    void *ptr = NULL;
    if (_bytes == 0) {
        rb_raise(rb_eArgError, "a pinned host buffer has at least one byte");
    }
    cumo_cuda_runtime_check_status(cudaHostAlloc(&ptr, _bytes, NUM2UINT(flags)));
    rb_nativethread_lock_lock(&pinned.lock);
    st_insert(pinned.table, (st_data_t)ptr, (st_data_t)_bytes);
    rb_nativethread_lock_unlock(&pinned.lock);
    return SIZET2NUM((size_t)ptr);
}

/*
  Frees a pinned host buffer.

  @param [Integer] buffer
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaFreeHost(VALUE self, VALUE buffer)
{
    size_t handle = NUM2SIZET(buffer);
    st_data_t key = (st_data_t)handle;
    int found;
    rb_nativethread_lock_lock(&pinned.lock);
    found = st_delete(pinned.table, &key, 0);
    rb_nativethread_lock_unlock(&pinned.lock);
    if (!found) {
        rb_raise(rb_eArgError, "not a live pinned host buffer");
    }
    cumo_cuda_runtime_check_status(cudaFreeHost((void*)handle));
    return Qnil;
}

/*
  Returns the size of a pinned host buffer in bytes.

  @param [Integer] buffer
  @return [Integer]
 */
static VALUE
rb_pinned_size(VALUE self, VALUE buffer)
{
    size_t size;
    pinned_get(buffer, &size);
    return SIZET2NUM(size);
}

static void
pinned_range(size_t size, size_t offset, size_t len)
{
    if (offset > size || len > size - offset) {
        rb_raise(rb_eRangeError, "%"PRIuSIZE" bytes at %"PRIuSIZE" do not fit in a pinned host buffer of %"PRIuSIZE, len, offset, size);
    }
}

/*
  Reads bytes out of a pinned host buffer.

  @param [Integer] buffer
  @param [Integer] offset
  @param [Integer] length
  @return [String]
 */
static VALUE
rb_pinned_read(VALUE self, VALUE buffer, VALUE offset, VALUE length)
{
    size_t size, off = NUM2SIZET(offset), len = NUM2SIZET(length);
    char *ptr = pinned_get(buffer, &size);
    pinned_range(size, off, len);
    return rb_str_new(ptr + off, (long)len);
}

/*
  Writes a String into a pinned host buffer.

  @param [Integer] buffer
  @param [Integer] offset
  @param [String] bytes
  @return [Integer] the number of bytes written
 */
static VALUE
rb_pinned_write(VALUE self, VALUE buffer, VALUE offset, VALUE bytes)
{
    size_t size, off = NUM2SIZET(offset), len;
    char *ptr;
    StringValue(bytes);
    ptr = pinned_get(buffer, &size);
    len = (size_t)RSTRING_LEN(bytes);
    pinned_range(size, off, len);
    memcpy(ptr + off, RSTRING_PTR(bytes), len);
    RB_GC_GUARD(bytes);
    return SIZET2NUM(len);
}

static void
pinned_narray_check(VALUE narray, size_t bytes)
{
    cumo_narray_t *na;
    if (!rb_obj_is_kind_of(narray, cumo_cNArray)) {
        rb_raise(rb_eTypeError, "an NArray is needed, got a %s", rb_obj_classname(narray));
    }
    if (rb_obj_is_kind_of(narray, cumo_cBit) || rb_obj_is_kind_of(narray, cumo_cRObject)) {
        rb_raise(rb_eTypeError, "a %s cannot be copied to or from a pinned host buffer", rb_obj_classname(narray));
    }
    if (cumo_na_check_contiguous(narray) != Qtrue) {
        rb_raise(rb_eArgError, "the NArray is not contiguous");
    }
    CumoGetNArray(narray, na);
    if (CUMO_NA_SIZE(na) * (size_t)cumo_na_element_stride(narray) != bytes) {
        rb_raise(rb_eArgError, "the NArray holds %"PRIuSIZE" bytes where the pinned host buffer holds %"PRIuSIZE,
                 CUMO_NA_SIZE(na) * (size_t)cumo_na_element_stride(narray), bytes);
    }
}

// Taking the pointer can run allocate, which is Ruby, so the array is
// measured again after it.
static char*
pinned_narray_pointer(VALUE narray, size_t bytes, int write)
{
    char *ptr;
    pinned_narray_check(narray, bytes);
    ptr = write ? cumo_na_get_offset_pointer_for_write(narray) : cumo_na_get_offset_pointer_for_read(narray);
    pinned_narray_check(narray, bytes);
    return ptr;
}

static cudaStream_t
stream_or_current(VALUE stream)
{
    return NIL_P(stream) ? cumo_cuda_stream() : cumo_cuda_stream_get(stream);
}

/*
  Copies a pinned host buffer into an NArray, asynchronously on a stream.

  @param [Integer] buffer
  @param [Cumo::NArray] narray contiguous, and of the buffer's size
  @param [Integer, nil] stream nil for the current stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_memcpy_pinned_to_narray(VALUE self, VALUE buffer, VALUE narray, VALUE stream)
{
    size_t size;
    char *src = pinned_get(buffer, &size);
    char *dst = pinned_narray_pointer(narray, size, 1);
    cumo_cuda_runtime_check_status(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream_or_current(stream)));
    return Qnil;
}

/*
  Copies an NArray into a pinned host buffer, asynchronously on a stream.

  @param [Cumo::NArray] narray contiguous, and of the buffer's size
  @param [Integer] buffer
  @param [Integer, nil] stream nil for the current stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_memcpy_narray_to_pinned(VALUE self, VALUE narray, VALUE buffer, VALUE stream)
{
    size_t size;
    char *dst = pinned_get(buffer, &size);
    char *src = pinned_narray_pointer(narray, size, 0);
    cumo_cuda_runtime_check_status(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_or_current(stream)));
    return Qnil;
}

/*
  Wait for compute device to finish.

  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
 */
static VALUE
rb_cudaDeviceSynchronize(VALUE self)
{
    cumo_cuda_runtime_device_synchronize();
    return Qnil;
}

void
Init_cumo_cuda_runtime()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    rb_define_const(mCumo, "Cuda", mCUDA); // alias
    mRuntime = rb_define_module_under(mCUDA, "Runtime");
    eRuntimeError = rb_define_class_under(mCUDA, "RuntimeError", rb_eStandardError);

    rb_define_singleton_method(mRuntime, "cudaDriverGetVersion", rb_cudaDriverGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaRuntimeGetVersion", rb_cudaRuntimeGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaGetDevice", rb_cudaGetDevice, 0);
    rb_define_singleton_method(mRuntime, "cudaDeviceGetAttributes", rb_cudaDeviceGetAttributes, 2);
    rb_define_singleton_method(mRuntime, "cudaGetDeviceCount", rb_cudaGetDeviceCount, 0);
    rb_define_singleton_method(mRuntime, "cudaSetDevice", rb_cudaSetDevice, 1);
    rb_define_singleton_method(mRuntime, "cudaDeviceCanAccessPeer", rb_cudaDeviceCanAccessPeer, 2);
    rb_define_singleton_method(mRuntime, "cudaDeviceSynchronize", rb_cudaDeviceSynchronize, 0);
    rb_define_singleton_method(mRuntime, "cudaStreamCreateWithFlags", rb_cudaStreamCreateWithFlags, 1);
    rb_define_singleton_method(mRuntime, "cudaStreamDestroy", rb_cudaStreamDestroy, 1);
    rb_define_singleton_method(mRuntime, "cudaStreamSynchronize", rb_cudaStreamSynchronize, 1);
    rb_define_singleton_method(mRuntime, "current_stream", rb_current_stream, 0);
    rb_define_singleton_method(mRuntime, "current_stream=", rb_current_stream_set, 1);
    rb_define_const(mRuntime, "CUDA_STREAM_DEFAULT", UINT2NUM(cudaStreamDefault));
    rb_define_const(mRuntime, "CUDA_STREAM_NON_BLOCKING", UINT2NUM(cudaStreamNonBlocking));
    rb_define_singleton_method(mRuntime, "cudaStreamQuery", rb_cudaStreamQuery, 1);
    rb_define_singleton_method(mRuntime, "cudaStreamWaitEvent", rb_cudaStreamWaitEvent, 2);
    rb_define_singleton_method(mRuntime, "cudaEventCreateWithFlags", rb_cudaEventCreateWithFlags, 1);
    rb_define_singleton_method(mRuntime, "cudaEventDestroy", rb_cudaEventDestroy, 1);
    rb_define_singleton_method(mRuntime, "cudaEventRecord", rb_cudaEventRecord, 2);
    rb_define_singleton_method(mRuntime, "cudaEventSynchronize", rb_cudaEventSynchronize, 1);
    rb_define_singleton_method(mRuntime, "cudaEventQuery", rb_cudaEventQuery, 1);
    rb_define_singleton_method(mRuntime, "cudaEventElapsedTime", rb_cudaEventElapsedTime, 2);
    rb_define_const(mRuntime, "CUDA_EVENT_DEFAULT", UINT2NUM(cudaEventDefault));
    rb_define_const(mRuntime, "CUDA_EVENT_BLOCKING_SYNC", UINT2NUM(cudaEventBlockingSync));
    rb_define_const(mRuntime, "CUDA_EVENT_DISABLE_TIMING", UINT2NUM(cudaEventDisableTiming));
    cumo_cuda_handle_set_init(&streams);
    cumo_cuda_handle_set_init(&events);
    rb_define_singleton_method(mRuntime, "cudaHostAlloc", rb_cudaHostAlloc, 2);
    rb_define_singleton_method(mRuntime, "cudaFreeHost", rb_cudaFreeHost, 1);
    rb_define_singleton_method(mRuntime, "pinned_size", rb_pinned_size, 1);
    rb_define_singleton_method(mRuntime, "pinned_read", rb_pinned_read, 3);
    rb_define_singleton_method(mRuntime, "pinned_write", rb_pinned_write, 3);
    rb_define_singleton_method(mRuntime, "memcpy_pinned_to_narray", rb_memcpy_pinned_to_narray, 3);
    rb_define_singleton_method(mRuntime, "memcpy_narray_to_pinned", rb_memcpy_narray_to_pinned, 3);
    rb_define_const(mRuntime, "CUDA_HOST_ALLOC_DEFAULT", UINT2NUM(cudaHostAllocDefault));
    rb_define_const(mRuntime, "CUDA_HOST_ALLOC_PORTABLE", UINT2NUM(cudaHostAllocPortable));
    rb_define_const(mRuntime, "CUDA_HOST_ALLOC_MAPPED", UINT2NUM(cudaHostAllocMapped));
    rb_define_const(mRuntime, "CUDA_HOST_ALLOC_WRITE_COMBINED", UINT2NUM(cudaHostAllocWriteCombined));
    cumo_cuda_handle_set_init(&in_use);
    cumo_cuda_handle_set_init(&pinned);
}
