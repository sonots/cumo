#include <ruby.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eRuntimeError;
VALUE cumo_cuda_mRuntime;
#define eRuntimeError cumo_cuda_eRuntimeError
#define mRuntime cumo_cuda_mRuntime

void
cumo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(eRuntimeError, "%s (error=%d)", cudaGetErrorString(status), status);
    }
}

bool
cumo_cuda_runtime_is_device_memory(void* ptr)
{
    struct cudaPointerAttributes attrs;
    cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
    return (status != cudaErrorInvalidValue);
}

char*
cumo_cuda_runtime_malloc(size_t size)
{
    void *ptr;
    cudaError_t status = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    cumo_cuda_runtime_check_status(status);
    return (char*)ptr;
}

void
cumo_cuda_runtime_free(void *ptr)
{
    cudaError_t status = cudaFree(ptr);
    cumo_cuda_runtime_check_status(status);
    ptr = 0;
}

#define check_status(status) (cumo_cuda_runtime_check_status((status)))

///////////////////////////////////////////
// Initialization
///////////////////////////////////////////

static VALUE
rb_cudaDriverGetVersion(VALUE self)
{
    int _version;
    cudaError_t status;

    status = cudaDriverGetVersion(&_version);

    check_status(status);
    return INT2NUM(_version);
}

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

static VALUE
rb_cudaGetDevice(VALUE self)
{
    int _device;
    cudaError_t status;

    status = cudaGetDevice(&_device);

    check_status(status);
    return INT2NUM(_device);
}

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

static VALUE
rb_cudaGetDeviceCount(VALUE self)
{
    int _count;
    cudaError_t status;

    status = cudaGetDeviceCount(&_count);

    check_status(status);
    return INT2NUM(_count);
}

static VALUE
rb_cudaSetDevice(VALUE self, VALUE device)
{
    int _device = NUM2INT(device);
    cudaError_t status;

    status = cudaSetDevice(_device);

    check_status(status);
    return Qnil;
}

static VALUE
rb_cudaDeviceSynchronize(VALUE self)
{
    cudaError_t status;
    status = cudaDeviceSynchronize();
    check_status(status);
    return Qnil;
}

void
Init_cumo_cuda_runtime()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    mRuntime = rb_define_module_under(mCUDA, "Runtime");
    eRuntimeError = rb_define_class_under(mCUDA, "RuntimeError", rb_eStandardError);

    rb_define_singleton_method(mRuntime, "cudaDriverGetVersion", rb_cudaDriverGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaRuntimeGetVersion", rb_cudaRuntimeGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaGetDevice", rb_cudaGetDevice, 0);
    rb_define_singleton_method(mRuntime, "cudaDeviceGetAttributes", rb_cudaDeviceGetAttributes, 2);
    rb_define_singleton_method(mRuntime, "cudaGetDeviceCount", rb_cudaGetDeviceCount, 0);
    rb_define_singleton_method(mRuntime, "cudaSetDevice", rb_cudaSetDevice, 1);
    rb_define_singleton_method(mRuntime, "cudaDeviceSynchronize", rb_cudaDeviceSynchronize, 0);
}
