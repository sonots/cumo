#ifndef CUMO_CUDA_CUDNN_H
#define CUMO_CUDA_CUDNN_H

#include <ruby.h>
#include <cudnn.h>
#include "cumo/cuda/cudnn_conv.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void
cumo_cuda_cudnn_check_status(cudnnStatus_t status);

cudnnHandle_t
cumo_cuda_cudnn_handle();

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUDNN_H */
