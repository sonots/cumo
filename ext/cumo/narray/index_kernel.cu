#include "cumo/narray_kernel.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

__global__ void cumo_na_index_aref_nadata_index_stride_kernel(size_t *idx, ssize_t s1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx[i] * s1;
    }
}

__global__ void cumo_na_index_aref_naview_index_index_kernel(size_t *idx, size_t *idx1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx1[idx[i]];
    }
}

__global__ void cumo_na_index_aref_naview_index_stride_last_kernel(size_t *idx, ssize_t s1, size_t last, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = (last - idx[i]) * s1;
    }
}

__global__ void cumo_na_index_aref_naview_index_stride_kernel(size_t *idx, ssize_t s1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx[i] * s1;
    }
}

__global__ void cumo_na_index_aref_naview_index_index_beg_step_kernel(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = idx1[beg + step * i];
    }
}

__global__ void cumo_na_index_at_nadata_index_beg_step_stride_kernel(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] = (beg + step * i) * s1;
    }
}

__global__ void cumo_na_index_at_nadata_index_beg_step_stride_add_kernel(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] += (beg + step * i) * s1;
    }
}

__global__ void cumo_na_index_at_nadata_index_stride_add_kernel(size_t *idx, size_t *idx1, ssize_t s1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] += idx1[i] * s1;
    }
}

__global__ void cumo_na_index_at_naview_index_index_index_add_kernel(size_t *idx, size_t *idx1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] += idx1[idx2[i]];
    }
}

__global__ void cumo_na_index_at_naview_index_index_beg_step_add_kernel(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] += idx1[beg + step * i];
    }
}

__global__ void cumo_na_index_at_naview_index_stride_last_add_kernel(size_t *idx, ssize_t s1, size_t last, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        idx[i] += (last - idx[i]) * s1;
    }
}

void cumo_na_index_aref_nadata_index_stride_kernel_launch(size_t *idx, ssize_t s1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_aref_nadata_index_stride_kernel<<<grid_dim, block_dim>>>(idx, s1, n);
}

void cumo_na_index_aref_naview_index_index_kernel_launch(size_t *idx, size_t *idx1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_aref_naview_index_index_kernel<<<grid_dim, block_dim>>>(idx, idx1, n);
}

void cumo_na_index_aref_naview_index_stride_last_kernel_launch(size_t *idx, ssize_t s1, size_t last, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_aref_naview_index_stride_last_kernel<<<grid_dim, block_dim>>>(idx, s1, last, n);
}

void cumo_na_index_aref_naview_index_stride_kernel_launch(size_t *idx, ssize_t s1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_aref_naview_index_stride_kernel<<<grid_dim, block_dim>>>(idx, s1, n);
}

void cumo_na_index_aref_naview_index_index_beg_step_kernel_launch(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_aref_naview_index_index_beg_step_kernel<<<grid_dim, block_dim>>>(idx, idx1, beg, step, n);
}

void cumo_na_index_at_nadata_index_stride_add_kernel_launch(size_t *idx, size_t *idx1, ssize_t s1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_nadata_index_stride_add_kernel<<<grid_dim, block_dim>>>(idx, idx1, s1, n);
}

void cumo_na_index_at_nadata_index_beg_step_stride_kernel_launch(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_nadata_index_beg_step_stride_kernel<<<grid_dim, block_dim>>>(idx, beg, step, s1, n);
}

void cumo_na_index_at_nadata_index_beg_step_stride_add_kernel_launch(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_nadata_index_beg_step_stride_add_kernel<<<grid_dim, block_dim>>>(idx, beg, step, s1, n);
}

void cumo_na_index_at_naview_index_index_index_add_kernel_launch(size_t *idx, size_t *idx1, size_t *idx2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_naview_index_index_index_add_kernel<<<grid_dim, block_dim>>>(idx, idx1, idx2, n);
}

void cumo_na_index_at_naview_index_index_beg_step_add_kernel_launch(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_naview_index_index_beg_step_add_kernel<<<grid_dim, block_dim>>>(idx, idx1, beg, step, n);
}

void cumo_na_index_at_naview_index_stride_last_add_kernel_launch(size_t *idx, ssize_t s1, size_t last, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_index_at_naview_index_stride_last_add_kernel<<<grid_dim, block_dim>>>(idx, s1, last, n);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

