#ifndef NUMO_CUDA_RUNTIME_H
#define NUMO_CUDA_RUNTIME_H
#include "numo/narray.h"
#include <cuda_runtime.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

extern VALUE numo_cuda_runtime_eCUDARuntimeError;

void numo_cuda_runtime_check_status(cudaError_t status);

#endif
