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

// Lazily initialize cudnn handle, and cache it
cudnnHandle_t
cumo_cuda_cudnn_handle()
{
    // One table per thread, as in cublas.c: the stream set on the handle must
    // not be reset by another thread before the call. A handle is never destroyed.
    static __thread cudnnHandle_t *handles = 0;
    int device;
    if (handles == 0) {
        int i;
        int device_count = cumo_cuda_runtime_get_device_count();
        handles = ALLOC_N(cudnnHandle_t, device_count);
        for (i = 0; i < device_count; ++i) {
            handles[i] = 0;
        }
    }
    device = cumo_cuda_runtime_get_device();
    if (handles[device] == 0) {
        // A discarded status leaves the handle NULL, and cuDNN reports that as
        // CUDNN_STATUS_NOT_INITIALIZED at the next call instead of the reason.
        cumo_cuda_cudnn_check_status(cudnnCreate(&handles[device]));
    }
    cumo_cuda_cudnn_check_status(cudnnSetStream(handles[device], cumo_cuda_stream()));
    return handles[device];
}

#endif // CUDNN_FOUND

/*
  Returns availability of cuDNN.

  @return [Boolean] Returns true if cuDNN is available
 */
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
    init_max_workspace_size();
    rb_define_singleton_method(mCUDNN, "max_workspace_size", rb_cudnn_max_workspace_size, 0);
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX", INT2NUM(CUDNN_POOLING_MAX));
    rb_define_const(mCUDNN, "CUDNN_POOLING_MAX_DETERMINISTIC", INT2NUM(CUDNN_POOLING_MAX_DETERMINISTIC));
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING));
    rb_define_const(mCUDNN, "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING", INT2NUM(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));
#endif // CUDNN_FOUND
}
