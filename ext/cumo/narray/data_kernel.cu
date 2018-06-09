#include "cumo/narray_kernel.h"

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
        memcpy(p1_, p2_, elmsz);
        // for (int j = 0; j < elmsz; ++j) {
        //     p1_[j] = p2_[j];
        // }
    }
}

#define m_swap_byte(p1,p2,t1,t2,e) \
    {                              \
        size_t j;                  \
        memcpy(t1,p1,e);           \
        for (j=0; j<e; j++) {      \
            t2[e-1-j] = t1[j];     \
        }                          \
        memcpy(p2,t2,e);           \
    }

__global__ void cumo_iter_swap_bytes_kernel(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    char *p1_ = NULL;
    char *p2_ = NULL;
    char *t1 = (char*)malloc(elmsz);
    char *t2 = (char*)malloc(elmsz);
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        p1_ = p1 + (idx1 ? idx1[i] : i * s1);
        p2_ = p2 + (idx2 ? idx2[i] : i * s2);
        m_swap_byte(p1_, p2_, t1, t2, elmsz);
    }
    free(t1);
    free(t2);
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    cumo_iter_copy_bytes_kernel<<<gridDim, blockDim>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
}

void cumo_iter_swap_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, size_t n, ssize_t elmsz)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    cumo_iter_swap_bytes_kernel<<<gridDim, blockDim>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
