#include <ruby.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cumo/cuda/runtime.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/template_kernel.h"
#include "cumo/cuda/handle.h"

VALUE cumo_cuda_eRuntimeError;
VALUE cumo_cuda_mRuntime;
uint64_t cumo_cuda_sync_epoch = 0;

static __thread cudaStream_t current_stream = 0;
static cumo_cuda_handle_set_t streams;

cudaStream_t
cumo_cuda_stream(void)
{
    return current_stream;
}

void
cumo_cuda_stream_set(cudaStream_t stream)
{
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
  Destroys a stream. The current stream cannot be destroyed.

  @param [Integer] stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamDestroy(VALUE self, VALUE stream)
{
    if ((cudaStream_t)NUM2SIZET(stream) == current_stream) {
        rb_raise(rb_eArgError, "the current stream cannot be destroyed");
    }
    cumo_cuda_runtime_check_status(cudaStreamDestroy((cudaStream_t)cumo_cuda_handle_take(&streams, stream, "cudaStream_t")));
    return Qnil;
}

static cudaStream_t
stream_get(VALUE v)
{
    if (NUM2SIZET(v) == 0) { return 0; }
    return (cudaStream_t)cumo_cuda_handle_get(&streams, v, "cudaStream_t");
}

/*
  Waits for everything queued on a stream. 0 is the legacy default stream.

  @param [Integer] stream
  @raise [Cumo::CUDA::RuntimeError]
 */
static VALUE
rb_cudaStreamSynchronize(VALUE self, VALUE stream)
{
    cumo_cuda_runtime_check_status(cudaStreamSynchronize(stream_get(stream)));
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
    current_stream = stream_get(stream);
    return stream;
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
    cumo_cuda_handle_set_init(&streams);
}
