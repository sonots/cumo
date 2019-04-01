#include <ruby.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eRuntimeError;
VALUE cumo_cuda_mRuntime;
#define eRuntimeError cumo_cuda_eRuntimeError
#define mRuntime cumo_cuda_mRuntime

#define check_status(status) (cumo_cuda_runtime_check_status((status)))

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
  Wait for compute device to finish.

  @raise [Cumo::CUDA::RuntimeError]
  @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
 */
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
    rb_define_const(mCumo, "Cuda", mCUDA); // alias
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
