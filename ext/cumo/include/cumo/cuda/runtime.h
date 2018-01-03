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
bool cumo_cuda_runtime_is_device_memory(void* ptr);
char* cumo_cuda_runtime_malloc(size_t size);
void cumo_cuda_runtime_free(char *ptr);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif
