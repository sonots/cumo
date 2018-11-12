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
        memcpy(p2_, p1_, elmsz);
    }
}

__global__ void cumo_na_diagonal_index_index_kernel(size_t *idx, size_t *idx0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx0[i+k0] + idx1[i+k1];
    }
}

__global__ void cumo_na_diagonal_index_stride_kernel(size_t *idx, size_t *idx0, ssize_t s1, size_t k0, size_t k1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx0[i+k0] + s1*(i+k1);
    }
}

__global__ void cumo_na_diagonal_stride_index_kernel(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = s0*(i+k0) + idx1[i+k1];
    }
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_iter_copy_bytes_kernel<<<grid_dim, block_dim>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
}

void cumo_na_diagonal_index_index_kernel_launch(size_t *idx, size_t *idx0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_index_kernel<<<grid_dim, block_dim>>>(idx, idx0, idx1, k0, k1, n);
}

void cumo_na_diagonal_index_stride_kernel_launch(size_t *idx, size_t *idx0, ssize_t s1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_stride_kernel<<<grid_dim, block_dim>>>(idx, idx0, s1, k0, k1, n);
}

void cumo_na_diagonal_stride_index_kernel_launch(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_stride_index_kernel<<<grid_dim, block_dim>>>(idx, s0, idx1, k0, k1, n);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
