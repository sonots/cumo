#include "cumo/cuda/cublas.h"

#include <assert.h>
#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/runtime.h"

//static void *blas_handle = 0;
//static char *blas_prefix = 0;

VALUE cumo_cuda_eCublasError;
VALUE cumo_cuda_mCublas;
#define eCublasError cumo_cuda_eCublasError
#define mCublas cumo_cuda_mCublas

static char* get_cublas_error_msg(cublasStatus_t error) {
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

//enum CBLAS_ORDER
//cumo_cuda_cublas_option_order(VALUE order)
//{
//    int opt;
//    char *ptr;
//
//    switch(TYPE(order)) {
//    case T_NIL:
//    case T_UNDEF:
//    case T_FALSE:
//        return CblasRowMajor;
//    case T_TRUE:
//        return CblasColMajor;
//    case T_FIXNUM:
//        opt = FIX2INT(order);
//        if (opt >= CblasRowMajor && opt <= CblasColMajor) {
//            return opt;
//        }
//        break;
//    case T_SYMBOL:
//        order = rb_sym2str(order);
//    case T_STRING:
//        ptr = RSTRING_PTR(order);
//        if (RSTRING_LEN(order) > 0) {
//            switch(ptr[0]){
//            case 'R': case 'r':
//                return CblasRowMajor;
//            case 'C': case 'c':
//                return CblasColMajor;
//            }
//        }
//        break;
//    }
//    rb_raise(rb_eArgError,"invalid value for CBLAS_ORDER");
//    return 0;
//}

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

cublasFillMode_t
cumo_cuda_cublas_option_uplo(VALUE uplo)
{
    int opt;
    char *ptr;

    switch(TYPE(uplo)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CUBLAS_FILL_MODE_UPPER;
    case T_TRUE:
        return CUBLAS_FILL_MODE_LOWER;
    case T_FIXNUM:
        opt = FIX2INT(uplo);
        switch(opt){
        case CUBLAS_FILL_MODE_UPPER:
        case CUBLAS_FILL_MODE_LOWER:
            return opt;
        }
        break;
    case T_SYMBOL:
        uplo = rb_sym2str(uplo);
    case T_STRING:
        ptr = RSTRING_PTR(uplo);
        if (RSTRING_LEN(uplo) > 0) {
            switch(ptr[0]){
            case 'U': case 'u':
                return CUBLAS_FILL_MODE_UPPER;
            case 'L': case 'l':
                return CUBLAS_FILL_MODE_LOWER;
            }
        }
        break;
    }
    rb_raise(rb_eArgError, "invalid value for cublasFillMode_t");
    return 0;
}

cublasDiagType_t
cumo_cuda_cublas_option_diag(VALUE diag)
{
    int opt;
    char *ptr;

    switch(TYPE(diag)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CUBLAS_DIAG_NON_UNIT;
    case T_TRUE:
        return CUBLAS_DIAG_UNIT;
    case T_FIXNUM:
        opt = FIX2INT(diag);
        switch(opt){
        case CUBLAS_DIAG_NON_UNIT:
        case CUBLAS_DIAG_UNIT:
            return opt;
        }
        break;
    case T_SYMBOL:
        diag = rb_sym2str(diag);
    case T_STRING:
        ptr = RSTRING_PTR(diag);
        if (RSTRING_LEN(diag) > 0) {
            switch(ptr[0]){
            case 'N': case 'n':
                return CUBLAS_DIAG_NON_UNIT;
            case 'U': case 'u':
                return CUBLAS_DIAG_UNIT;
            }
        }
        break;
    }
    rb_raise(rb_eArgError, "invalid value for cublasDiagType_t");
    return 0;
}

cublasSideMode_t
cumo_cuda_cublas_option_side(VALUE side)
{
    int opt;
    char *ptr;

    switch(TYPE(side)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CUBLAS_SIDE_LEFT;
    case T_TRUE:
        return CUBLAS_SIDE_RIGHT;
    case T_FIXNUM:
        opt = FIX2INT(side);
        switch(opt){
        case CUBLAS_SIDE_LEFT:
        case CUBLAS_SIDE_RIGHT:
            return opt;
        }
        break;
    case T_SYMBOL:
        side = rb_sym2str(side);
    case T_STRING:
        ptr = RSTRING_PTR(side);
        if (RSTRING_LEN(side) > 0) {
            switch(ptr[0]){
            case 'L': case 'l':
                return CUBLAS_SIDE_LEFT;
            case 'R': case 'r':
                return CUBLAS_SIDE_RIGHT;
            }
        }
        break;
    }
    rb_raise(rb_eArgError, "invalid value for cublasSideMode_t");
    return 0;
}

//void
//cumo_cuda_cublas_check_func(void **func, const char *name)
//{
//    char *s, *error;
//
//    if (*func==0) {
//        if (blas_handle==0) {
//            rb_raise(rb_eRuntimeError,"BLAS library is not loaded");
//        }
//        if (blas_prefix==0) {
//            rb_raise(rb_eRuntimeError,"CBLAS prefix is not set");
//        }
//        s = alloca(strlen(blas_prefix)+strlen(name)+1);
//        strcpy(s,blas_prefix);
//        strcat(s,name);
//        dlerror();
//        *func = dlsym(blas_handle, s);
//        error = dlerror();
//        if (error != NULL) {
//            rb_raise(rb_eRuntimeError, "%s", error);
//        }
//    }
//}

//static VALUE
//blas_s_prefix_set(VALUE mod, VALUE prefix)
//{
//    long len;
//
//    if (TYPE(prefix) != T_STRING) {
//        rb_raise(rb_eTypeError,"argument must be string");
//    }
//    if (blas_prefix) {
//        free(blas_prefix);
//    }
//    len = RSTRING_LEN(prefix);
//    blas_prefix = malloc(len+1);
//    strcpy(blas_prefix, StringValueCStr(prefix));
//    return prefix;
//}

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

    // rb_define_module_function(mBlas, "dlopen", blas_s_dlopen, -1);
    // rb_define_module_function(mBlas, "prefix=", blas_s_prefix_set, 1);

    // blas_prefix = malloc(strlen("cublas_")+1); // default prefix
    // strcpy(blas_prefix,"cublas_");

    // Init_cumo_linalg_blas_s();
    // Init_cumo_linalg_blas_d();
    // Init_cumo_linalg_blas_c();
    // Init_cumo_linalg_blas_z();
}
