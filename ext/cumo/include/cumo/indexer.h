#ifndef CUMO_INDEXER_H
#define CUMO_INDEXER_H

/* Add cumo_ prefix */
#define na_indexer_t cumo_na_indexer_t
#define na_iarray_t cumo_na_iarray_t
#define na_make_indexer cumo_na_make_indexer
#define na_make_iarray cumo_na_make_iarray

#ifndef __CUDACC__
#include "cumo/narray.h"
#include "cumo/ndloop.h"
#else
#include "cumo/narray_kernel.h"
#endif

/* A structure to get indices for each dimension.
 *
 * Note that shapes of each argument NArray are typically equivalent, and
 * thus indexer would point the same indicies for all NArrays.
 */
typedef struct {
    unsigned char ndim;               // # of dimensions
    size_t total_size;                // # of total elements
    size_t shape[NA_MAX_DIMENSION];   // # of elements for each dimension
    uint64_t index[NA_MAX_DIMENSION]; // indicies for each dimension
} na_indexer_t;

/* A structure to get data address with indexer.
 *
 * Note that strides would be different for each NArray although indexer points same indicies.
 */
typedef struct {
    char* ptr;
    ssize_t step[NA_MAX_DIMENSION]; // or strides
} na_iarray_t;

#ifndef __CUDACC__
// Note that you, then, have to call na_indexer_set to create index[]
static na_indexer_t
na_make_indexer(na_loop_args_t* arg)
{
    na_indexer_t indexer;
    indexer.ndim = arg->ndim;
    indexer.total_size = 1;
    for (int i = 0; i < arg->ndim; ++i) {
        indexer.shape[i] = arg->shape[i];
        indexer.total_size *= arg->shape[i];
    }
    return indexer;
}

static na_iarray_t
na_make_iarray(na_loop_args_t* arg)
{
    na_iarray_t iarray;
    iarray.ptr = arg->ptr + arg->iter[0].pos;
    for (int idim = arg->ndim; --idim >= 0;) {
        iarray.step[idim] = arg->iter[idim].step;
    }
    return iarray;
}
#endif  // #ifndef __CUDACC__

#define CUMO_NA_INDEXER_OPTIMIZED_NDIM 4

#ifdef __CUDACC__

__host__ __device__
static inline void
cumo_na_indexer_set_dim(na_indexer_t* indexer, uint64_t i) {
    for (int j = indexer->ndim; --j >= 0;) {
        indexer->index[j] = i % indexer->shape[j];
        i /= indexer->shape[j];
    }
}

// Let compiler optimize
#define CUMO_NA_INDEXER_SET(NDIM) \
__host__ __device__ \
static inline void \
cumo_na_indexer_set_dim##NDIM(na_indexer_t* indexer, uint64_t i) { \
    for (int j = NDIM; --j >= 0;) { \
        indexer->index[j] = i % indexer->shape[j]; \
        i /= indexer->shape[j]; \
    } \
}

CUMO_NA_INDEXER_SET(4)
CUMO_NA_INDEXER_SET(3)
CUMO_NA_INDEXER_SET(2)
CUMO_NA_INDEXER_SET(0)

__host__ __device__
static inline void
cumo_na_indexer_set_dim1(na_indexer_t* indexer, uint64_t i) {
    indexer->index[0] = i;
}

__host__ __device__
static inline char*
cumo_na_iarray_at_dim(na_iarray_t* iarray, na_indexer_t* indexer) {
    char* ptr = iarray->ptr;
    for (int idim = 0; idim < indexer->ndim; ++idim) {
        ptr += iarray->step[idim] * indexer->index[idim];
    }
    return ptr;
}

// Let compiler optimize
#define CUMO_NA_IARRAY_AT(NDIM) \
__host__ __device__ \
static inline char* \
cumo_na_iarray_at_dim##NDIM(na_iarray_t* iarray, na_indexer_t* indexer) { \
    char* ptr = iarray->ptr; \
    for (int idim = 0; idim < NDIM; ++idim) { \
        ptr += iarray->step[idim] * indexer->index[idim]; \
    } \
    return ptr; \
}

CUMO_NA_IARRAY_AT(4)
CUMO_NA_IARRAY_AT(3)
CUMO_NA_IARRAY_AT(2)
CUMO_NA_IARRAY_AT(0)

__host__ __device__
static inline char*
cumo_na_iarray_at_dim1(na_iarray_t* iarray, na_indexer_t* indexer) {
    return iarray->ptr + iarray->step[0] * indexer->index[0];
}

#endif // #ifdef __CUDACC__

#endif // CUMO_INDEXER_H
