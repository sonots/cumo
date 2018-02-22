#ifndef CUMO_CUDA_CUBLAS_H
#define CUMO_CUDA_CUBLAS_H

#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#define option_value cumo_cublas_option_value
extern VALUE cumo_cublas_option_value(VALUE value, VALUE default_value);

//#define option_order cumo_cublas_option_order
//extern enum CBLAS_ORDER cumo_cublas_option_order(VALUE order);

#define option_trans cumo_cublas_option_trans
extern cublasOperation_t cumo_cublas_option_trans(VALUE trans);

#define option_uplo cumo_cublas_option_uplo
extern cublasFillMode_t cumo_cublas_option_uplo(VALUE uplo);

#define option_diag cumo_cublas_option_diag
extern cublasDiagType_t cumo_cublas_option_diag(VALUE diag);

#define option_side cumo_cublas_option_side
extern cublasSideMode_t cumo_cublas_option_side(VALUE side);

//#define check_func cumo_cublas_check_func
//extern void cumo_cublas_check_func(void **func, const char *name);

#define SWAP(a,b,tmp)                                             \
    {(tmp)=(a);(a)=(b);(b)=(tmp);}

#define SWAP_IFTR(trans,a,b,tmp)                                  \
    { if ((trans)!=CblasNoTrans)                                  \
            {(tmp)=(a);(a)=(b);(b)=(tmp);}                        \
    }

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

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUBLAS_H */
