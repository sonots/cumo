#ifndef CUMO_CUDA_RUNTIME_H
#define CUMO_CUDA_RUNTIME_H
#include "cumo/narray.h"
#include <cuda_runtime.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

extern VALUE cumo_cuda_runtime_eCUDARuntimeError;

void cumo_cuda_runtime_check_status(cudaError_t status);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif
