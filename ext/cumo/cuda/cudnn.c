#include "cumo/cuda/cudnn.h"

#include <assert.h>
#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eCudnnError;
VALUE cumo_cuda_mCudnn;
#define eCudnnError cumo_cuda_eCudnnError
#define mCudnn cumo_cuda_mCudnn

void
cumo_cuda_cudnn_check_status(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS) {
        rb_raise(cumo_cuda_eCudnnError, "%s (error=%d)", cudnnGetErrorString(status), status);
    }
}

// Lazily initialize cudnn handle, and cache it
cudnnHandle_t
cumo_cuda_cudnn_handle()
{
    static cudnnHandle_t *handles = 0;  // handle is never destroyed
    int device;
    if (handles == 0) {
        int i;
        int device_count = cumo_cuda_runtime_get_device_count();
        handles = malloc(sizeof(cudnnHandle_t) * device_count);
        for (i = 0; i < device_count; ++i) {
            handles[i] = 0;
        }
    }
    device = cumo_cuda_runtime_get_device();
    if (handles[device] == 0) {
        cudnnCreate(&handles[device]);
    }
    return handles[device];
}

void
Init_cumo_cuda_cudnn(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    /*
      Document-module: Cumo::Cudnn
    */
    mCudnn = rb_define_module_under(mCUDA, "Cudnn");
    eCudnnError = rb_define_class_under(mCUDA, "CudnnError", rb_eStandardError);
}
