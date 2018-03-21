#ifndef CUMO_CUDA_MEMORY_H
#define CUMO_CUDA_MEMORY_H

#include "cumo/narray.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

char*
cumo_cuda_runtime_malloc(size_t size);

void
cumo_cuda_runtime_free(char *ptr);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_MEMORY_H */
