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

VALUE
cumo_cuda_cublas_option_value(VALUE value, VALUE default_value);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUBLAS_H */
