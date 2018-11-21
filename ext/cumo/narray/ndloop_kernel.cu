#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

// TODO(sonots): Use optimized indexer

__global__ void cumo_ndloop_copy_from_buffer_kernel(cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        char* p = cumo_na_iarray_stridx_at_dim(&a, &indexer);
        memcpy(p, buf + i * elmsz, elmsz);
    }
}

void cumo_ndloop_copy_from_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    cumo_ndloop_copy_from_buffer_kernel<<<grid_dim, block_dim>>>(*a,*indexer,buf,elmsz);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
