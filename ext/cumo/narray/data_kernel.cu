#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"

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

// flatten builds an index array whenever the dimensions it collapses do not
// come out as one stride. Filling it on the host meant a size_t store per
// element into memory the device owns, which faults a page at a time: a
// 512x2048 column slice took 2.4 ms, and the copy that follows faults every
// page back. The offsets are the same mixed-radix walk the host loop did.
#define CUMO_NA_FLATTEN_INDEX_KERNEL(NDIM) \
__global__ void cumo_na_flatten_index_kernel_dim##NDIM(size_t *idx, cumo_na_iarray_stridx_t iarray, cumo_na_indexer_t indexer) \
{ \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        uint64_t rest = i; \
        size_t pos = 0; \
        for (int idim = NDIM; --idim >= 0;) { \
            uint64_t c = rest % indexer.shape[idim]; \
            rest /= indexer.shape[idim]; \
            if (CUMO_SDX_IS_INDEX(iarray.stridx[idim])) { \
                size_t *idim_idx = CUMO_SDX_GET_INDEX(iarray.stridx[idim]); \
                if (idim_idx) pos += idim_idx[c]; \
            } else { \
                pos += (size_t)(CUMO_SDX_GET_STRIDE(iarray.stridx[idim]) * (ssize_t)c); \
            } \
        } \
        idx[i] = pos; \
    } \
}

CUMO_NA_FLATTEN_INDEX_KERNEL(1)
CUMO_NA_FLATTEN_INDEX_KERNEL(2)
CUMO_NA_FLATTEN_INDEX_KERNEL(3)
CUMO_NA_FLATTEN_INDEX_KERNEL(4)

__global__ void cumo_na_flatten_index_kernel_dim(size_t *idx, cumo_na_iarray_stridx_t iarray, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        uint64_t rest = i;
        size_t pos = 0;
        for (int idim = indexer.ndim; --idim >= 0;) {
            uint64_t c = rest % indexer.shape[idim];
            rest /= indexer.shape[idim];
            if (CUMO_SDX_IS_INDEX(iarray.stridx[idim])) {
                size_t *idim_idx = CUMO_SDX_GET_INDEX(iarray.stridx[idim]);
                if (idim_idx) pos += idim_idx[c];
            } else {
                pos += (size_t)(CUMO_SDX_GET_STRIDE(iarray.stridx[idim]) * (ssize_t)c);
            }
        }
        idx[i] = pos;
    }
}

// Copying a whole view in one launch, so that ndloop does not have to walk the
// outer dimensions itself. It synchronizes once per outer step when an operand
// carries an index array, which for a gathered view costs far more than the
// copy: 6250 rows took 37.6 ms that way and 0.2 ms through here.
__global__ void cumo_iter_copy_bytes_indexer_kernel_dim0(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim0(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim0(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim0(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

__global__ void cumo_iter_copy_bytes_indexer_kernel_dim1(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim1(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim1(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim1(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

__global__ void cumo_iter_copy_bytes_indexer_kernel_dim2(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim2(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim2(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim2(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

__global__ void cumo_iter_copy_bytes_indexer_kernel_dim3(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim3(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim3(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim3(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

__global__ void cumo_iter_copy_bytes_indexer_kernel_dim4(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim4(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim4(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim4(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

__global__ void cumo_iter_copy_bytes_indexer_kernel_dim(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

void cumo_iter_copy_bytes_indexer_kernel_launch(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    case 0:
        cumo_iter_copy_bytes_indexer_kernel_dim0<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 1:
        cumo_iter_copy_bytes_indexer_kernel_dim1<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 2:
        cumo_iter_copy_bytes_indexer_kernel_dim2<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 3:
        cumo_iter_copy_bytes_indexer_kernel_dim3<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 4:
        cumo_iter_copy_bytes_indexer_kernel_dim4<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    default:
        cumo_iter_copy_bytes_indexer_kernel_dim<<<grid_dim, block_dim>>>(*a1, *a2, *indexer, elmsz);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_flatten_index_kernel_launch(size_t *idx, cumo_na_iarray_stridx_t* iarray, cumo_na_indexer_t* indexer)
{
    size_t grid_dim, block_dim;
    if (indexer->total_size == 0) return;
    grid_dim = cumo_get_grid_dim(indexer->total_size);
    block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    case 1:
        cumo_na_flatten_index_kernel_dim1<<<grid_dim, block_dim>>>(idx, *iarray, *indexer);
        break;
    case 2:
        cumo_na_flatten_index_kernel_dim2<<<grid_dim, block_dim>>>(idx, *iarray, *indexer);
        break;
    case 3:
        cumo_na_flatten_index_kernel_dim3<<<grid_dim, block_dim>>>(idx, *iarray, *indexer);
        break;
    case 4:
        cumo_na_flatten_index_kernel_dim4<<<grid_dim, block_dim>>>(idx, *iarray, *indexer);
        break;
    default:
        cumo_na_flatten_index_kernel_dim<<<grid_dim, block_dim>>>(idx, *iarray, *indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_iter_copy_bytes_kernel<<<grid_dim, block_dim>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_index_index_kernel_launch(size_t *idx, size_t *idx0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_index_kernel<<<grid_dim, block_dim>>>(idx, idx0, idx1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_index_stride_kernel_launch(size_t *idx, size_t *idx0, ssize_t s1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_stride_kernel<<<grid_dim, block_dim>>>(idx, idx0, s1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_stride_index_kernel_launch(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_stride_index_kernel<<<grid_dim, block_dim>>>(idx, s0, idx1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
