#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#define CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL(NDIM) \
__global__ void cumo_ndloop_copy_from_buffer_kernel_dim##NDIM( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        memcpy(p, buf + i * elmsz, elmsz); \
    } \
}

#define CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL(NDIM) \
__global__ void cumo_ndloop_copy_to_buffer_kernel_dim##NDIM( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        memcpy(buf + i * elmsz, p, elmsz); \
    } \
}

CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL(1)
CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL(2)
CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL(3)
CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL(4)
CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL()

CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL(1)
CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL(2)
CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL(3)
CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL(4)
CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL()

#undef CUMO_NDLOOP_COPY_FROM_BUFFER_KERNEL
#undef CUMO_NDLOOP_COPY_TO_BUFFER_KERNEL

void cumo_ndloop_copy_from_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
        case 1:
            cumo_ndloop_copy_from_buffer_kernel_dim1<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 2:
            cumo_ndloop_copy_from_buffer_kernel_dim2<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 3:
            cumo_ndloop_copy_from_buffer_kernel_dim3<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 4:
            cumo_ndloop_copy_from_buffer_kernel_dim4<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        default:
            cumo_ndloop_copy_from_buffer_kernel_dim<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
    }
}

void cumo_ndloop_copy_to_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
        case 1:
            cumo_ndloop_copy_to_buffer_kernel_dim1<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 2:
            cumo_ndloop_copy_to_buffer_kernel_dim2<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 3:
            cumo_ndloop_copy_to_buffer_kernel_dim3<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        case 4:
            cumo_ndloop_copy_to_buffer_kernel_dim4<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
        default:
            cumo_ndloop_copy_to_buffer_kernel_dim<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
            break;
    }
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
