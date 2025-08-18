#include "cumo/cuda/cudnn.h"

#include <assert.h>
#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eCUDNNError;
VALUE cumo_cuda_mCUDNN;
#define eCUDNNError cumo_cuda_eCUDNNError
#define mCUDNN cumo_cuda_mCUDNN

#ifdef CUDNN_FOUND

void
cumo_cuda_cudnn_check_status(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS) {
        rb_raise(cumo_cuda_eCUDNNError, "%s (error=%d)", cudnnGetErrorString(status), status);
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

#endif // CUDNN_FOUND

/*
  Returns availability of cuDNN.

  @return [Boolean] Returns true if cuDNN is available
 */
static VALUE
rb_cudnn_available_p(VALUE self)
{
#if CUDNN_FOUND
    return Qtrue;
#else
    return Qfalse;
#endif
}

void
Init_cumo_cuda_cudnn(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    /*
      Document-module: Cumo::CUDNN
    */
    mCUDNN = rb_define_module_under(mCUDA, "CUDNN");
    rb_define_const(mCUDA, "Cudnn", mCUDNN); // alias
    eCUDNNError = rb_define_class_under(mCUDA, "CUDNNError", rb_eStandardError);

    rb_define_singleton_method(mCUDNN, "available?", rb_cudnn_available_p, 0);
#ifdef CUDNN_FOUND
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX", INT2NUM(CUDNN_POOLING_MAX));
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX_DETERMINISTIC", INT2NUM(CUDNN_POOLING_MAX_DETERMINISTIC));
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING));
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
#endif // CUDNN_FOUND
}
