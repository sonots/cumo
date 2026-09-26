#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"
#include "cumo/template_kernel.h"

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

// swap_byte reverses the bytes of every element. The host loop this replaces
// read and wrote one element at a time, behind a full device synchronize, at
// 5.3 ns an element whatever the shape.
__device__ static uint64_t cumo_swap_bytes64(uint64_t v)
{
    uint32_t lo = (uint32_t)v;
    uint32_t hi = (uint32_t)(v >> 32);
    return ((uint64_t)__byte_perm(lo, 0, 0x0123) << 32) | (uint64_t)__byte_perm(hi, 0, 0x0123);
}

// A whole element is read before any of it is written, which is what an inplace
// swap_byte needs: it hands the same address in on both sides. An element is
// laid out on a multiple of its own size, so the wide loads below are the ones
// the data allows -- the check is there to make that so rather than assumed,
// and every thread takes the same side of it.
__device__ static void cumo_swap_bytes(char *dst, const char *src, ssize_t elmsz)
{
    uintptr_t addr = (uintptr_t)dst | (uintptr_t)src;

    switch (elmsz) {
    case 1:
        dst[0] = src[0];
        return;
    case 2:
        if ((addr & 1) == 0) {
            uint16_t v = *(const uint16_t*)src;
            *(uint16_t*)dst = (uint16_t)((v >> 8) | (v << 8));
            return;
        }
        break;
    case 4:
        if ((addr & 3) == 0) {
            *(uint32_t*)dst = __byte_perm(*(const uint32_t*)src, 0, 0x0123);
            return;
        }
        break;
    case 8:
        if ((addr & 7) == 0) {
            *(uint64_t*)dst = cumo_swap_bytes64(*(const uint64_t*)src);
            return;
        }
        break;
    case 16:
        if ((addr & 7) == 0) {
            uint64_t lo = ((const uint64_t*)src)[0];
            uint64_t hi = ((const uint64_t*)src)[1];
            ((uint64_t*)dst)[0] = cumo_swap_bytes64(hi);
            ((uint64_t*)dst)[1] = cumo_swap_bytes64(lo);
            return;
        }
        break;
    default:
        break;
    }

    if (dst == src) {
        for (ssize_t j = 0, k = elmsz - 1; j < k; ++j, --k) {
            char t = dst[j];
            dst[j] = dst[k];
            dst[k] = t;
        }
    } else {
        for (ssize_t j = 0; j < elmsz; ++j) {
            dst[elmsz - 1 - j] = src[j];
        }
    }
}

#define CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(NDIM) \
__global__ void cumo_iter_swap_byte_indexer_kernel_dim##NDIM(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz) \
{ \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p1 = cumo_na_iarray_at_dim##NDIM(&a1, &indexer); \
        char* p2 = cumo_na_iarray_at_dim##NDIM(&a2, &indexer); \
        cumo_swap_bytes(p2, p1, elmsz); \
    } \
}

CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(0)
CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(1)
CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(2)
CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(3)
CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL(4)
CUMO_ITER_SWAP_BYTE_INDEXER_KERNEL()

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
// element into memory the device owns, which faults a page at a time, and the
// copy that follows faults every page back. The offsets are the same mixed-radix walk the host loop did.
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
// carries an index array, which for a gathered view of many rows costs far
// more than the copy itself.
__global__ void cumo_iter_copy_bytes_indexer_kernel_dim(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, ssize_t elmsz)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim(&a2, &indexer);
        memcpy(p2, p1, elmsz);
    }
}

#define CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(NDIM) \
__global__ void cumo_iter_copy_bytes_stridx_kernel_dim##NDIM( \
        cumo_na_iarray_stridx_t a1, cumo_na_iarray_stridx_t a2, cumo_na_indexer_t indexer, ssize_t elmsz) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        char* p1 = cumo_na_iarray_stridx_at_dim##NDIM(&a1, &indexer); \
        char* p2 = cumo_na_iarray_stridx_at_dim##NDIM(&a2, &indexer); \
        memcpy(p2, p1, elmsz); \
    } \
}

CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(0)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(1)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(2)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(3)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(4)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(5)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(6)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(7)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL(8)
CUMO_ITER_COPY_BYTES_STRIDX_KERNEL()

#undef CUMO_ITER_COPY_BYTES_STRIDX_KERNEL

void cumo_iter_copy_bytes_stridx_kernel_launch(cumo_na_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
#define CUMO_ITER_COPY_BYTES_STRIDX_CASE(NDIM) \
    case NDIM: \
        cumo_iter_copy_bytes_stridx_kernel_dim##NDIM<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz); \
        break;
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(0)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(1)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(2)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(3)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(4)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(5)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(6)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(7)
    CUMO_ITER_COPY_BYTES_STRIDX_CASE(8)
#undef CUMO_ITER_COPY_BYTES_STRIDX_CASE
    default:
        cumo_iter_copy_bytes_stridx_kernel_dim<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

// A copy of a transposed view has one side strided whichever way it is walked,
// so it stages a square tile in shared memory the way the typed store does. The
// element size is only known here at run time, so there is one kernel per size
// a tile can carry.
struct cumo_copy_bytes_elm16 { uint64_t lo, hi; };

#define CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(SZ, TYPE) \
__global__ void cumo_iter_copy_bytes_transpose_kernel_##SZ( \
        char *dst, char *src, uint64_t rows, uint64_t cols) { \
    CUMO_TRANSPOSE_TILE_DECL(TYPE, tile); \
    CUMO_TRANSPOSE_TILE_LOOP(tile, rows, cols, \
        *(TYPE*)(src + cumo_tile_src * sizeof(TYPE)), \
        *(TYPE*)(dst + cumo_tile_dst * sizeof(TYPE)) = cumo_tile_val; \
    ); \
}

CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(1, uint8_t)
CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(2, uint16_t)
CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(4, uint32_t)
CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(8, uint64_t)
CUMO_ITER_COPY_BYTES_TRANSPOSE_KERNEL(16, cumo_copy_bytes_elm16)

static int
cumo_copy_elmsz_is_typed(ssize_t elmsz)
{
    return elmsz == 1 || elmsz == 2 || elmsz == 4 || elmsz == 8 || elmsz == 16;
}

static int
cumo_copy_is_aligned(cumo_na_iarray_t* dst, cumo_na_iarray_t* src, cumo_na_indexer_t* indexer, ssize_t w, int ndim)
{
    if ((uintptr_t)dst->ptr % w != 0 || (uintptr_t)src->ptr % w != 0) { return 0; }
    for (int k = 0; k < ndim; ++k) {
        if (indexer->shape[k] > 1 && (dst->step[k] % w != 0 || src->step[k] % w != 0)) { return 0; }
    }
    return 1;
}

// True when the source runs along its columns and the destination along its
// rows. a1 is the source here and a2 the destination, the other way round from
// the typed store.
static int
cumo_iter_copy_bytes_is_transpose(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    return cumo_copy_elmsz_is_typed(elmsz) &&
        CUMO_TRANSPOSE_TILE_FITS(indexer) &&
        a2->step[1] == elmsz &&
        a2->step[0] == elmsz * (ssize_t)indexer->shape[1] &&
        a1->step[0] == elmsz &&
        a1->step[1] == elmsz * (ssize_t)indexer->shape[0];
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#define CUMO_COPY_WIDE_KERNEL(NDIM) \
template<typename V> \
__global__ void cumo_copy_wide_kernel_dim##NDIM(cumo_na_iarray_t dst, cumo_na_iarray_t src, cumo_na_indexer_t indexer) { \
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) { \
        cumo_na_indexer_set_dim##NDIM(&indexer, i); \
        *(V*)cumo_na_iarray_at_dim##NDIM(&dst, &indexer) = *(V*)cumo_na_iarray_at_dim##NDIM(&src, &indexer); \
    } \
}

CUMO_COPY_WIDE_KERNEL(0)
CUMO_COPY_WIDE_KERNEL(1)
CUMO_COPY_WIDE_KERNEL(2)
CUMO_COPY_WIDE_KERNEL(3)
CUMO_COPY_WIDE_KERNEL(4)
CUMO_COPY_WIDE_KERNEL(5)
CUMO_COPY_WIDE_KERNEL(6)
CUMO_COPY_WIDE_KERNEL(7)
CUMO_COPY_WIDE_KERNEL(8)
CUMO_COPY_WIDE_KERNEL()
CUMO_COPY_WIDE_KERNEL(2n)
CUMO_COPY_WIDE_KERNEL(3n)
CUMO_COPY_WIDE_KERNEL(4n)

#undef CUMO_COPY_WIDE_KERNEL

template<typename V>
static void
cumo_copy_wide_launch(cumo_na_iarray_t* dst, cumo_na_iarray_t* src, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    const cumo_na_iarray_t* const narrow_[] = {dst, src};
    switch (indexer->ndim) {
#define CUMO_COPY_WIDE_CASE(NDIM) \
    case NDIM: \
        cumo_copy_wide_kernel_dim##NDIM<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*dst, *src, *indexer); \
        break;
#define CUMO_COPY_WIDE_NARROW_CASE(NDIM) \
    case NDIM: \
        if (cumo_na_indexer_is_narrow(indexer, narrow_, 2)) { \
            cumo_copy_wide_kernel_dim##NDIM##n<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*dst, *src, *indexer); \
            break; \
        } \
        cumo_copy_wide_kernel_dim##NDIM<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*dst, *src, *indexer); \
        break;
    CUMO_COPY_WIDE_CASE(0)
    CUMO_COPY_WIDE_CASE(1)
    CUMO_COPY_WIDE_NARROW_CASE(2)
    CUMO_COPY_WIDE_NARROW_CASE(3)
    CUMO_COPY_WIDE_NARROW_CASE(4)
    CUMO_COPY_WIDE_CASE(5)
    CUMO_COPY_WIDE_CASE(6)
    CUMO_COPY_WIDE_CASE(7)
    CUMO_COPY_WIDE_CASE(8)
#undef CUMO_COPY_WIDE_CASE
#undef CUMO_COPY_WIDE_NARROW_CASE
    default:
        cumo_copy_wide_kernel_dim<V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*dst, *src, *indexer);
        break;
    }
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

// Launches nothing and answers 0 unless the last dimension is contiguous and aligned on both sides.
int
cumo_copy_bytes_wide(cumo_na_iarray_t* dst, cumo_na_iarray_t* src, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    int last = indexer->ndim - 1;
    ssize_t w;
    size_t bytes;
    cumo_na_iarray_t d, s;
    cumo_na_indexer_t ix;

    if (indexer->ndim == 0 || indexer->total_size == 0) { return 0; }
    if (dst->step[last] != elmsz || src->step[last] != elmsz) { return 0; }
    bytes = indexer->shape[last] * (size_t)elmsz;
    for (w = 16; w > elmsz; w /= 2) {
        if (bytes % w == 0 && cumo_copy_is_aligned(dst, src, indexer, w, last)) { break; }
    }
    if (w <= elmsz) { return 0; }

    d = *dst;
    s = *src;
    ix = *indexer;
    d.step[last] = s.step[last] = w;
    ix.shape[last] = bytes / w;
    ix.total_size = indexer->total_size / indexer->shape[last] * ix.shape[last];
    if (ix.shape[last] == 1 && ix.ndim > 1) { --ix.ndim; }
    switch (w) {
    case 16: cumo_copy_wide_launch<uint4>(&d, &s, &ix); break;
    case 8: cumo_copy_wide_launch<uint2>(&d, &s, &ix); break;
    case 4: cumo_copy_wide_launch<uint32_t>(&d, &s, &ix); break;
    default: cumo_copy_wide_launch<uint16_t>(&d, &s, &ix); break;
    }
    cumo_cuda_runtime_check_kernel_launch();
    return 1;
}

static int
cumo_copy_bytes_typed(cumo_na_iarray_t* dst, cumo_na_iarray_t* src, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    if (!cumo_copy_elmsz_is_typed(elmsz) || !cumo_copy_is_aligned(dst, src, indexer, elmsz, indexer->ndim)) { return 0; }
    if (indexer->total_size == 0) { return 1; }
    switch (elmsz) {
    case 16: cumo_copy_wide_launch<uint4>(dst, src, indexer); break;
    case 8: cumo_copy_wide_launch<uint2>(dst, src, indexer); break;
    case 4: cumo_copy_wide_launch<uint32_t>(dst, src, indexer); break;
    case 2: cumo_copy_wide_launch<uint16_t>(dst, src, indexer); break;
    default: cumo_copy_wide_launch<uint8_t>(dst, src, indexer); break;
    }
    cumo_cuda_runtime_check_kernel_launch();
    return 1;
}

void cumo_iter_copy_bytes_indexer_kernel_launch(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    if (cumo_iter_copy_bytes_is_transpose(a1, a2, indexer, elmsz)) {
        uint64_t rows = indexer->shape[0];
        uint64_t cols = indexer->shape[1];
        switch (elmsz) {
        case 1:
            CUMO_TRANSPOSE_LAUNCH(cumo_iter_copy_bytes_transpose_kernel_1, rows, cols, a2->ptr, a1->ptr);
            break;
        case 2:
            CUMO_TRANSPOSE_LAUNCH(cumo_iter_copy_bytes_transpose_kernel_2, rows, cols, a2->ptr, a1->ptr);
            break;
        case 4:
            CUMO_TRANSPOSE_LAUNCH(cumo_iter_copy_bytes_transpose_kernel_4, rows, cols, a2->ptr, a1->ptr);
            break;
        case 8:
            CUMO_TRANSPOSE_LAUNCH(cumo_iter_copy_bytes_transpose_kernel_8, rows, cols, a2->ptr, a1->ptr);
            break;
        default:
            CUMO_TRANSPOSE_LAUNCH(cumo_iter_copy_bytes_transpose_kernel_16, rows, cols, a2->ptr, a1->ptr);
            break;
        }
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }
    if (cumo_copy_bytes_wide(a2, a1, indexer, elmsz)) { return; }
    if (cumo_copy_bytes_typed(a2, a1, indexer, elmsz)) { return; }
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    cumo_iter_copy_bytes_indexer_kernel_dim<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
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
        cumo_na_flatten_index_kernel_dim1<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, *iarray, *indexer);
        break;
    case 2:
        cumo_na_flatten_index_kernel_dim2<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, *iarray, *indexer);
        break;
    case 3:
        cumo_na_flatten_index_kernel_dim3<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, *iarray, *indexer);
        break;
    case 4:
        cumo_na_flatten_index_kernel_dim4<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, *iarray, *indexer);
        break;
    default:
        cumo_na_flatten_index_kernel_dim<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, *iarray, *indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, uint64_t n, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_iter_copy_bytes_kernel<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(p1, p2, s1, s2, idx1, idx2, n, elmsz);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_iter_swap_byte_indexer_kernel_launch(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, ssize_t elmsz)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    case 0:
        cumo_iter_swap_byte_indexer_kernel_dim0<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 1:
        cumo_iter_swap_byte_indexer_kernel_dim1<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 2:
        cumo_iter_swap_byte_indexer_kernel_dim2<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 3:
        cumo_iter_swap_byte_indexer_kernel_dim3<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    case 4:
        cumo_iter_swap_byte_indexer_kernel_dim4<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    default:
        cumo_iter_swap_byte_indexer_kernel_dim<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1, *a2, *indexer, elmsz);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_index_index_kernel_launch(size_t *idx, size_t *idx0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_index_kernel<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, idx0, idx1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_index_stride_kernel_launch(size_t *idx, size_t *idx0, ssize_t s1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_index_stride_kernel<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, idx0, s1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

void cumo_na_diagonal_stride_index_kernel_launch(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    cumo_na_diagonal_stride_index_kernel<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(idx, s0, idx1, k0, k1, n);
    cumo_cuda_runtime_check_kernel_launch();
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
