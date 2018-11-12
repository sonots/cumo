#include "cumo/narray_kernel.h"
#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

__global__ void cumo_iter_copy_bytes_kernel(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    char *p1_ = NULL;
    char *p2_ = NULL;
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        p1_ = p1 + (idx1 ? idx1[i] : i * s1);
        p2_ = p2 + (idx2 ? idx2[i] : i * s2);
        memcpy(p2_, p1_, elmsz);
    }
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_iter_copy_bytes_kernel<<<grid_dim, block_dim>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
