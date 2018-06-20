#ifndef CUMO_CUDA_CUBLAS_H
#define CUMO_CUDA_CUBLAS_H

#include <ruby.h>
#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void
cumo_cuda_cublas_check_status(cublasStatus_t status);

cublasHandle_t
cumo_cuda_cublas_handle();

#define option_value cumo_cuda_cublas_option_value
extern VALUE cumo_cuda_cublas_option_value(VALUE value, VALUE default_value);

//#define option_order cumo_cuda_cublas_option_order
//extern enum CBLAS_ORDER cumo_cuda_cublas_option_order(VALUE order);

#define option_trans cumo_cuda_cublas_option_trans
extern cublasOperation_t cumo_cuda_cublas_option_trans(VALUE trans);

#define option_uplo cumo_cuda_cublas_option_uplo
extern cublasFillMode_t cumo_cuda_cublas_option_uplo(VALUE uplo);

#define option_diag cumo_cuda_cublas_option_diag
extern cublasDiagType_t cumo_cuda_cublas_option_diag(VALUE diag);

#define option_side cumo_cuda_cublas_option_side
extern cublasSideMode_t cumo_cuda_cublas_option_side(VALUE side);

//#define check_func cumo_cuda_cublas_check_func
//extern void cumo_cuda_cublas_check_func(void **func, const char *name);

// TODO: Check if a and b are row_major?
/*
#define SWAP_IFROW(a,b,tmp)                                       \
    {(tmp)=(a);(a)=(b);(b)=(tmp);}

#define SWAP_IFTR(trans,a,b,tmp)                                  \
    { if ((trans)!=CUBLAS_OP_N)                                   \
            {(tmp)=(a);(a)=(b);(b)=(tmp);}                        \
    }
*/

/*
//#define SWAP_IFCOLTR(order,trans,a,b,tmp)                       \
//    { if (((order)==CblasRowMajor && (trans)!=CblasNoTrans) ||  \
//          ((order)!=CblasRowMajor && (trans)==CblasNoTrans))    \
//            {(tmp)=(a);(a)=(b);(b)=(tmp);}                      \
//    }

//#define SWAP_IFCOL(order,a,b,tmp)                               \
//    { if ((order)==CblasColMajor) {(tmp)=(a);(a)=(b);(b)=(tmp);} }
//
//#define SWAP_IFROW(order,a,b,tmp)                               \
//    { if ((order)==CblasRowMajor) {(tmp)=(a);(a)=(b);(b)=(tmp);} }
//
//#define SWAP_IFCOLTR(order,trans,a,b,tmp)                       \
//    { if (((order)==CblasRowMajor && (trans)!=CblasNoTrans) ||  \
//          ((order)!=CblasRowMajor && (trans)==CblasNoTrans))    \
//            {(tmp)=(a);(a)=(b);(b)=(tmp);}                      \
//    }
//
//#define CHECK_FUNC(fptr, fname)                                 \
//    { if ((fptr)==0) { check_func((void*)(&(fptr)),fname); } }
*/

#define ROW_SIZE(na) ((na)->shape[(na)->ndim-2])
#define COL_SIZE(na) ((na)->shape[(na)->ndim-1])

#define CHECK_NARRAY_TYPE(x,t)                                 \
    if (CLASS_OF(x)!=(t)) {                                    \
        rb_raise(rb_eTypeError,"invalid NArray type (class)"); \
    }

// Error Class ??
#define CHECK_DIM_GE(na,nd)                                     \
    if ((na)->ndim<(nd)) {                                      \
        rb_raise(cumo_na_eShapeError,                              \
                 "n-dimension=%d, but >=%d is expected",        \
                 (na)->ndim, (nd));                             \
    }

#define CHECK_DIM_EQ(na1,nd)                                    \
    if ((na1)->ndim != (nd)) {                                  \
        rb_raise(cumo_na_eShapeError,                              \
                 "dimention mismatch: %d != %d",                \
                 (na1)->ndim, (nd));                            \
    }

#define CHECK_SQUARE(name,na)                                           \
    if ((na)->shape[(na)->ndim-1] != (na)->shape[(na)->ndim-2]) {       \
        rb_raise(cumo_na_eShapeError,"%s is not square matrix",name);      \
    }

#define CHECK_SIZE_GE(na,sz)                                    \
    if ((na)->size < (size_t)(sz)) {                            \
        rb_raise(cumo_na_eShapeError,                              \
                 "NArray size must be >= %"SZF"u",(size_t)(sz));\
    }
#define CHECK_NON_EMPTY(na)                                     \
    if ((na)->size==0) {                                        \
        rb_raise(cumo_na_eShapeError,"empty NArray");              \
    }

#define CHECK_SIZE_EQ(n,m)                                      \
    if ((n)!=(m)) {                                             \
        rb_raise(cumo_na_eShapeError,                              \
                 "size mismatch: %"SZF"d != %"SZF"d",           \
                 (size_t)(n),(size_t)(m));                      \
    }

#define CHECK_SAME_SHAPE(na1,na2)                                \
    {   int i;                                                   \
        CHECK_DIM_EQ(na1,na2->ndim);                             \
        for (i=0; i<na1->ndim; i++) {                            \
            CHECK_SIZE_EQ(na1->shape[i],na2->shape[i]);          \
        }                                                        \
    }

#define CHECK_INT_EQ(sm,m,sn,n)                          \
    if ((m) != (n)) {                                    \
        rb_raise(cumo_na_eShapeError,                       \
                 "%s must be == %s: %s=%d %s=%d",        \
                 sm,sn,sm,m,sn,n);                       \
    }

// Error Class ??
#define CHECK_LEADING_GE(sld,ld,sn,n)                    \
    if ((ld) < (n)) {                                    \
        rb_raise(cumo_na_eShapeError,                       \
                 "%s must be >= max(%s,1): %s=%d %s=%d", \
                 sld,sn,sld,ld,sn,n);                    \
    }

#define COPY_OR_CAST_TO(a,T)                            \
    {                                                   \
        if (CLASS_OF(a) == (T)) {                       \
            if (!TEST_INPLACE(a)) {                     \
                a = cumo_na_copy(a);                         \
            }                                           \
        } else {                                        \
            a = rb_funcall(T,rb_intern("cast"),1,a);    \
        }                                               \
    }

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUBLAS_H */
