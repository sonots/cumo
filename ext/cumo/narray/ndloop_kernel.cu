#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"

// The copies move whole elements. With the element size known at compile time
// each is one load and one store; a memcpy of a runtime size goes byte by byte
// and made an index view the operand of an elementwise op cost twice the gather
// of the same view into an array.
struct __align__(16) cumo_ndloop_elm16 { uint64_t v[2]; };

#define CUMO_NDLOOP_COPY_BUFFER_KERNEL(NDIM) \
template<typename V> \
__global__ void cumo_ndloop_copy_from_buffer_kernel_dim##NDIM( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        *(V*)p = ((const V*)buf)[i]; \
    } \
} \
template<typename V> \
__global__ void cumo_ndloop_copy_to_buffer_kernel_dim##NDIM( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        ((V*)buf)[i] = *(const V*)p; \
    } \
} \
template<> \
__global__ void cumo_ndloop_copy_from_buffer_kernel_dim##NDIM<char>( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        memcpy(p, buf + i * elmsz, elmsz); \
    } \
} \
template<> \
__global__ void cumo_ndloop_copy_to_buffer_kernel_dim##NDIM<char>( \
        cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, char *buf, size_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p = cumo_na_iarray_stridx_at_dim##NDIM(&a, &indexer); \
        memcpy(buf + i * elmsz, p, elmsz); \
    } \
}

CUMO_NDLOOP_COPY_BUFFER_KERNEL(1)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(2)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(3)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(4)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(5)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(6)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(7)
CUMO_NDLOOP_COPY_BUFFER_KERNEL(8)
CUMO_NDLOOP_COPY_BUFFER_KERNEL()

#undef CUMO_NDLOOP_COPY_BUFFER_KERNEL

// An index holds a byte offset that is a whole number of the steps it was
// taken with, so the pointer and the steps decide whether every element sits
// on its own size, as the generated stridx stores already assume.
static bool
cumo_ndloop_is_typed(const cumo_na_iarray_stridx_t* a, const cumo_na_indexer_t* indexer, size_t elmsz)
{
    if (!(elmsz == 1 || elmsz == 2 || elmsz == 4 || elmsz == 8 || elmsz == 16)) { return false; }
    if ((uintptr_t)a->ptr % elmsz != 0) { return false; }
    for (int k = 0; k < indexer->ndim; ++k) {
        if (!CUMO_SDX_IS_INDEX(a->stridx[k]) && CUMO_SDX_GET_STRIDE(a->stridx[k]) % (ssize_t)elmsz != 0) { return false; }
    }
    return true;
}

#define CUMO_NDLOOP_BUFFER_SWITCH(DIR, V) \
    switch (indexer->ndim) { \
    case 1: cumo_ndloop_copy_##DIR##_buffer_kernel_dim1<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 2: cumo_ndloop_copy_##DIR##_buffer_kernel_dim2<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 3: cumo_ndloop_copy_##DIR##_buffer_kernel_dim3<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 4: cumo_ndloop_copy_##DIR##_buffer_kernel_dim4<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 5: cumo_ndloop_copy_##DIR##_buffer_kernel_dim5<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 6: cumo_ndloop_copy_##DIR##_buffer_kernel_dim6<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 7: cumo_ndloop_copy_##DIR##_buffer_kernel_dim7<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    case 8: cumo_ndloop_copy_##DIR##_buffer_kernel_dim8<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    default: cumo_ndloop_copy_##DIR##_buffer_kernel_dim<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a,*indexer,buf,elmsz); break; \
    }

#define CUMO_NDLOOP_BUFFER_DISPATCH(DIR) \
    if (!cumo_ndloop_is_typed(a, indexer, elmsz)) { CUMO_NDLOOP_BUFFER_SWITCH(DIR, char) } \
    else switch (elmsz) { \
    case 1: CUMO_NDLOOP_BUFFER_SWITCH(DIR, uint8_t) break; \
    case 2: CUMO_NDLOOP_BUFFER_SWITCH(DIR, uint16_t) break; \
    case 4: CUMO_NDLOOP_BUFFER_SWITCH(DIR, uint32_t) break; \
    case 8: CUMO_NDLOOP_BUFFER_SWITCH(DIR, uint64_t) break; \
    default: CUMO_NDLOOP_BUFFER_SWITCH(DIR, cumo_ndloop_elm16) break; \
    }

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_ndloop_copy_from_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    CUMO_NDLOOP_BUFFER_DISPATCH(from)
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_ndloop_copy_to_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    CUMO_NDLOOP_BUFFER_DISPATCH(to)
    cumo_cuda_runtime_check_kernel_launch();
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
