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
        return #msg

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
    if (status != 0) {
        rb_raise(cumo_cuda_eCublasError, "%s (error=%d)", get_cublas_error_msg(status), status);
    }
}

// Lazily initialize cublas handle, and cache it
cublasHandle_t
cumo_cuda_cublas_handle()
{
    static cublasHandle_t *handles = 0;  // handle is never destroyed
    if (handles == 0) {
        int i;
        int device_count = cumo_cuda_runtime_get_device_count();
        handles = malloc(sizeof(cublasHandle_t) * device_count);
        for (i = 0; i < device_count; ++i) {
            handles[i] = 0;
        }
    }
    int device = cumo_cuda_runtime_get_device();
    if (handles[device] == 0) {
        cublasCreate(&handles[device]);
    }
    return handles[device];
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

#if 0
cublasOperation_t
cumo_cuda_cublas_option_trans(VALUE trans)
{
    int opt;
    char *ptr;

    switch(TYPE(trans)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CUBLAS_OP_N;
    case T_TRUE:
        return CUBLAS_OP_T;
    case T_FIXNUM:
        opt = FIX2INT(trans);
        if (opt >= CUBLAS_OP_N && opt <= CUBLAS_OP_C) {
            return opt;
        }
        break;
    case T_SYMBOL:
        trans = rb_sym2str(trans);
    case T_STRING:
        ptr = RSTRING_PTR(trans);
        if (RSTRING_LEN(trans) > 0) {
            switch(ptr[0]){
            case 'N': case 'n':
                return CUBLAS_OP_N;
            case 'T': case 't':
                return CUBLAS_OP_T;
            case 'C': case 'c':
                return CUBLAS_OP_C;
            }
        }
        break;
    }
    rb_raise(rb_eArgError, "invalid value for cublasOperation_t");
    return 0;
}
#endif

void
Init_cumo_cuda_cublas(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    /*
      Document-module: Cumo::Cublas
    */
    mCublas = rb_define_module_under(mCUDA, "Cublas");
    eCublasError = rb_define_class_under(mCUDA, "CublasError", rb_eStandardError);
}
