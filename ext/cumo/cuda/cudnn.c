#include "cumo/cuda/cudnn.h"

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
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
    cumo_cuda_runtime_note_device_write();
    if (status != CUDNN_STATUS_SUCCESS) {
        rb_raise(cumo_cuda_eCUDNNError, "%s (error=%d)", cudnnGetErrorString(status), status);
    }
}

static size_t cudnn_max_workspace_size = CUMO_CUDA_CUDNN_DEFAULT_MAX_WORKSPACE_SIZE;

size_t
cumo_cuda_cudnn_max_workspace_size()
{
    return cudnn_max_workspace_size;
}

static void
init_max_workspace_size(void)
{
    const char* env = getenv("CUMO_CUDNN_MAX_WORKSPACE_SIZE");
    char* end = NULL;
    unsigned long long v;

    if (env == NULL || *env == '\0') return;
    // strtoull takes a leading minus and wraps it, so reject the sign itself
    if (*env == '-' || *env == '+') goto BAD;
    errno = 0;
    v = strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || v == 0 || v > (unsigned long long)SIZE_MAX) goto BAD;
    cudnn_max_workspace_size = (size_t)v;
    return;

BAD:
    rb_warn("CUMO_CUDNN_MAX_WORKSPACE_SIZE=%s is not a positive byte count, using %"PRIuSIZE,
            env, cudnn_max_workspace_size);
}

VALUE
cumo_cuda_cudnn_release_conv_held(VALUE held)
{
    cumo_cuda_cudnn_conv_held_t *h = (cumo_cuda_cudnn_conv_held_t*)held;

    if (h->x_desc) { cudnnDestroyTensorDescriptor(h->x_desc); h->x_desc = 0; }
    if (h->y_desc) { cudnnDestroyTensorDescriptor(h->y_desc); h->y_desc = 0; }
    if (h->b_desc) { cudnnDestroyTensorDescriptor(h->b_desc); h->b_desc = 0; }
    if (h->w_desc) { cudnnDestroyFilterDescriptor(h->w_desc); h->w_desc = 0; }
    if (h->conv_desc) { cudnnDestroyConvolutionDescriptor(h->conv_desc); h->conv_desc = 0; }
    // cuDNN is still reading the workspace, so the wait is the caller's to
    // report. A free that raises from here would replace the exception being
    // carried out, so it goes back the way an unwinding path has to.
    cumo_cuda_runtime_return_scratch(h->workspace, h->workspace != NULL, &h->wait_status);
    h->workspace = NULL;
    return Qnil;
}

static cumo_cuda_thread_local_t handles;

// One handle per thread, as in cublas.c: the stream set on the handle must
// not be reset by another thread before the call.
cudnnHandle_t
cumo_cuda_cudnn_handle()
{
    cudnnHandle_t *handle = (cudnnHandle_t*)cumo_cuda_thread_local_get(&handles);
    if (*handle == 0) {
        // A discarded status leaves the handle NULL, and cuDNN reports that as
        // CUDNN_STATUS_NOT_INITIALIZED at the next call instead of the reason.
        cumo_cuda_cudnn_check_status(cudnnCreate(handle));
    }
    cumo_cuda_cudnn_check_status(cudnnSetStream(*handle, cumo_cuda_stream()));
    return *handle;
}

#endif // CUDNN_FOUND

#ifdef CUDNN_FOUND
/*
  Returns the ceiling cuDNN may use when it searches for a convolution
  algorithm, set by CUMO_CUDNN_MAX_WORKSPACE_SIZE.

  @return [Integer] bytes
 */
static VALUE
rb_cudnn_max_workspace_size(VALUE self)
{
    return SIZET2NUM(cumo_cuda_cudnn_max_workspace_size());
}
#endif // CUDNN_FOUND

/*
  Returns whether Cumo was built with cuDNN.

  @return [Boolean]
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
      Document-module: Cumo::CUDA::CUDNN

      cuDNN functions for neural networks. Each function other than available?
      and max_workspace_size takes the array as its first argument and calls
      the method of the same name on it, so conv(x, w) is x.conv(w). The
      methods are those of DFloat, SFloat, HFloat and BFloat, of which
      max_pool, avg_pool and their backwards come from
      Cumo::CUDA::CUDNN::NArrayMethods. available? tells whether Cumo was
      built with cuDNN, and max_workspace_size and the CUDNN_POOLING_*
      constants are defined only in a build that was.
    */
    mCUDNN = rb_define_module_under(mCUDA, "CUDNN");
    /* Cumo\:\:CUDA\:\:CUDNN: The same module as CUDNN, under another name. */
    rb_define_const(mCUDA, "Cudnn", mCUDNN);
    /*
      Document-class: Cumo::CUDA::CUDNNError

      Raised when a call to cuDNN fails.
    */
    eCUDNNError = rb_define_class_under(mCUDA, "CUDNNError", rb_eStandardError);

    rb_define_singleton_method(mCUDNN, "available?", rb_cudnn_available_p, 0);
#ifdef CUDNN_FOUND
    cumo_cuda_thread_local_init(&handles, sizeof(cudnnHandle_t));
    init_max_workspace_size();
    rb_define_singleton_method(mCUDNN, "max_workspace_size", rb_cudnn_max_workspace_size, 0);
    /* 0: Max pooling. */
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX", INT2NUM(CUDNN_POOLING_MAX));
    /* 3: Max pooling with a deterministic algorithm. */
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX_DETERMINISTIC", INT2NUM(CUDNN_POOLING_MAX_DETERMINISTIC));
    /* 1: Average pooling that counts the padding as zeros. */
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING));
    /* 2: Average pooling that leaves the padding out. */
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
#endif // CUDNN_FOUND
}
