#include "cumo/cuda/cublas.h"

#include <assert.h>
#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eCublasError;
VALUE cumo_cuda_mCublas;
#define eCublasError cumo_cuda_eCublasError
#define mCublas cumo_cuda_mCublas

static char*
get_cublas_error_msg(cublasStatus_t error) {
    switch (error) {
#define RETURN_MSG(msg) \
    case msg:                              \
        return (char*)#msg

        RETURN_MSG(CUBLAS_STATUS_SUCCESS);
        RETURN_MSG(CUBLAS_STATUS_NOT_INITIALIZED);
        RETURN_MSG(CUBLAS_STATUS_ALLOC_FAILED);
        RETURN_MSG(CUBLAS_STATUS_INVALID_VALUE);
        RETURN_MSG(CUBLAS_STATUS_ARCH_MISMATCH);
        RETURN_MSG(CUBLAS_STATUS_MAPPING_ERROR);
        RETURN_MSG(CUBLAS_STATUS_EXECUTION_FAILED);
        RETURN_MSG(CUBLAS_STATUS_INTERNAL_ERROR);
        RETURN_MSG(CUBLAS_STATUS_NOT_SUPPORTED);
        RETURN_MSG(CUBLAS_STATUS_LICENSE_ERROR);

#undef RETURN_MSG
    }
    abort(); // never reach
}

void
cumo_cuda_cublas_check_status(cublasStatus_t status)
{
    cumo_cuda_runtime_note_device_write();
    if (status != 0) {
        rb_raise(cumo_cuda_eCublasError, "%s (error=%d)", get_cublas_error_msg(status), status);
    }
}

static cumo_cuda_thread_local_t handles;

static void
destroy_handle(void *entry)
{
    cublasDestroy(*(cublasHandle_t*)entry);
}

// One handle per thread: the stream is set on the handle before each call,
// and another thread with another current stream must not reset it in
// between. The handle goes with the thread.
cublasHandle_t
cumo_cuda_cublas_handle()
{
    cublasHandle_t *handle = (cublasHandle_t*)cumo_cuda_thread_local_get(&handles);
    if (*handle == 0) {
        // A discarded status leaves the handle NULL, and cuBLAS reports that as
        // CUBLAS_STATUS_NOT_INITIALIZED at the next call instead of the reason.
        cumo_cuda_cublas_check_status(cublasCreate(handle));
    }
    cumo_cuda_cublas_check_status(cublasSetStream(*handle, cumo_cuda_stream()));
    return *handle;
}

VALUE
cumo_cuda_cublas_option_value(VALUE value, VALUE default_value)
{
    switch(TYPE(value)) {
    case T_NIL:
    case T_UNDEF:
        return default_value;
    }
    return value;
}


void
Init_cumo_cuda_cublas(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    cumo_cuda_thread_local_init(&handles, sizeof(cublasHandle_t), destroy_handle);

    /*
      Document-module: Cumo::Cublas
    */
    mCublas = rb_define_module_under(mCUDA, "Cublas");
    eCublasError = rb_define_class_under(mCUDA, "CublasError", rb_eStandardError);
}
