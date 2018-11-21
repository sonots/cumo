#ifndef CUMO_INDEXER_H
#define CUMO_INDEXER_H

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
    size_t shape[CUMO_NA_MAX_DIMENSION];   // # of elements for each dimension
    uint64_t index[CUMO_NA_MAX_DIMENSION]; // indicies for each dimension
    uint64_t raw_index;
} cumo_na_indexer_t;

/* A structure to get data address with indexer.
 *
 * Note that strides would be different for each NArray although indexer points same indicies.
 */
typedef struct {
    char* ptr;
    ssize_t step[CUMO_NA_MAX_DIMENSION]; // or strides
} cumo_na_iarray_t;

typedef struct {
    char* ptr;
    cumo_stridx_t stridx[CUMO_NA_MAX_DIMENSION];
} cumo_na_iarray_stridx_t;

typedef struct {
    cumo_na_iarray_t in;
    cumo_na_iarray_t out;
    cumo_na_indexer_t in_indexer;
    cumo_na_indexer_t out_indexer;
} cumo_na_reduction_arg_t;

#ifndef __CUDACC__
extern int cumo_na_debug_flag;  // narray.c

static void
print_cumo_na_indexer_t(cumo_na_indexer_t* indexer)
{
    printf("cumo_na_indexer_t = 0x%"SZF"x {\n", (size_t)indexer);
    printf("  ndim = %d\n", indexer->ndim);
    printf("  total_size = %ld\n", indexer->total_size);
    printf("  shape = 0x%"SZF"x\n", (size_t)indexer->shape);
    for (int i = 0; i < indexer->ndim; ++i) {
        printf("  shape[%d] = %ld\n", i, indexer->shape[i]);
    }
    printf("}\n");
}

static void
print_cumo_na_iarray_t(cumo_na_iarray_t* iarray, unsigned char ndim)
{
    printf("cumo_na_iarray_t = 0x%"SZF"x {\n", (size_t)iarray);
    printf("  ptr = 0x%"SZF"x\n", (size_t)iarray->ptr);
    printf("  step = 0x%"SZF"x\n", (size_t)iarray->step);
    for (int i = 0; i < ndim; ++i) {
        printf("  step[%d] = %ld\n", i, iarray->step[i]);
    }
    printf("}\n");
}

static void
print_cumo_na_reduction_arg_t(cumo_na_reduction_arg_t* arg)
{
    printf("cumo_na_reduction_arg_t = 0x%"SZF"x {\n", (size_t)arg);
    printf("--in--\n");
    print_cumo_na_iarray_t(&arg->in, arg->in_indexer.ndim);
    printf("--out--\n");
    print_cumo_na_iarray_t(&arg->out, arg->out_indexer.ndim);
    printf("--in_indexer--\n");
    print_cumo_na_indexer_t(&arg->in_indexer);
    printf("--out_indexer--\n");
    print_cumo_na_indexer_t(&arg->out_indexer);
    printf("}\n");
}

// Note that you, then, have to call cumo_na_indexer_set to create index[]
static cumo_na_indexer_t
cumo_na_make_indexer(cumo_na_loop_args_t* arg)
{
    cumo_na_indexer_t indexer;
    indexer.ndim = arg->ndim;
    indexer.total_size = 1;
    for (int i = 0; i < arg->ndim; ++i) {
        indexer.shape[i] = arg->shape[i];
        indexer.total_size *= arg->shape[i];
    }
    return indexer;
}

static cumo_na_iarray_t
cumo_na_make_iarray_given_ndim(cumo_na_loop_args_t* arg, int ndim)
{
    cumo_na_iarray_t iarray;
    iarray.ptr = arg->ptr + arg->iter[0].pos;
    for (int idim = ndim; --idim >= 0;) {
        iarray.step[idim] = arg->iter[idim].step;
    }
    return iarray;
}

static cumo_na_iarray_t
cumo_na_make_iarray(cumo_na_loop_args_t* arg)
{
    return cumo_na_make_iarray_given_ndim(arg, arg->ndim);
}

static cumo_na_reduction_arg_t
cumo_na_make_reduction_arg(cumo_na_loop_t* lp_user)
{
    cumo_na_reduction_arg_t arg;
    int i;
    int in_ndim = lp_user->args[0].ndim;

    // in shape = (2, 3, 4, 5, 6)
    // axis = (1, 3)
    // out shape = (2, 4, 6)
    // reduce shape = (3, 5)

    arg.in = cumo_na_make_iarray(&lp_user->args[0]);
    arg.in_indexer = cumo_na_make_indexer(&lp_user->args[0]);

    arg.out_indexer.ndim = 0;
    arg.out_indexer.total_size = 1;
    for (i = 0; i < in_ndim; ++i) {
        if (!cumo_na_test_reduce(lp_user->reduce, i)) {
            arg.out_indexer.shape[arg.out_indexer.ndim] = arg.in_indexer.shape[i];
            arg.out_indexer.total_size *= arg.in_indexer.shape[i];
            ++arg.out_indexer.ndim;
        }
    }
    arg.out = cumo_na_make_iarray_given_ndim(&lp_user->args[1], arg.out_indexer.ndim);

    if (cumo_na_debug_flag) {
        print_cumo_na_reduction_arg_t(&arg);
    }

    return arg;
}

#endif  // #ifndef __CUDACC__

#define CUMO_NA_INDEXER_OPTIMIZED_NDIM 4

#ifdef __CUDACC__

__host__ __device__
static inline void
cumo_na_indexer_set_dim(cumo_na_indexer_t* indexer, uint64_t i) {
    indexer->raw_index = i;
    for (int j = indexer->ndim; --j >= 0;) {
        indexer->index[j] = i % indexer->shape[j];
        i /= indexer->shape[j];
    }
}

// Let compiler optimize
#define CUMO_NA_INDEXER_SET(NDIM) \
__host__ __device__ \
static inline void \
cumo_na_indexer_set_dim##NDIM(cumo_na_indexer_t* indexer, uint64_t i) { \
    indexer->raw_index = i; \
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
cumo_na_indexer_set_dim1(cumo_na_indexer_t* indexer, uint64_t i) {
    indexer->raw_index = i;
}

__host__ __device__
static inline char*
cumo_na_iarray_at_dim(cumo_na_iarray_t* iarray, cumo_na_indexer_t* indexer) {
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
cumo_na_iarray_at_dim##NDIM(cumo_na_iarray_t* iarray, cumo_na_indexer_t* indexer) { \
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
cumo_na_iarray_at_dim1(cumo_na_iarray_t* iarray, cumo_na_indexer_t* indexer) {
    return iarray->ptr + iarray->step[0] * indexer->raw_index;
}

__host__ __device__
static inline char*
cumo_na_iarray_stridx_at_dim(cumo_na_iarray_stridx_t* iarray, cumo_na_indexer_t* indexer) {
    char* ptr = iarray->ptr;
    for (int idim = 0; idim < indexer->ndim; ++idim) {
        if (CUMO_SDX_IS_INDEX(iarray->stridx[idim])) {
            ptr += CUMO_SDX_GET_INDEX(iarray->stridx[idim])[indexer->index[idim]];
        } else {
            ptr += CUMO_SDX_GET_STRIDE(iarray->stridx[idim]) * indexer->index[idim];
        }
    }
    return ptr;
}

// Let compiler optimize
#define CUMO_NA_IARRAY_STRIDX_AT(NDIM) \
__host__ __device__ \
static inline char* \
cumo_na_iarray_stridx_at_dim##NDIM(cumo_na_iarray_stridx_t* iarray, cumo_na_indexer_t* indexer) { \
    char* ptr = iarray->ptr; \
    for (int idim = 0; idim < NDIM; ++idim) { \
        if (CUMO_SDX_IS_INDEX(iarray->stridx[idim])) { \
            ptr += CUMO_SDX_GET_INDEX(iarray->stridx[idim])[indexer->index[idim]]; \
        } else { \
            ptr += CUMO_SDX_GET_STRIDE(iarray->stridx[idim]) * indexer->index[idim]; \
        } \
    } \
    return ptr; \
}

CUMO_NA_IARRAY_STRIDX_AT(4)
CUMO_NA_IARRAY_STRIDX_AT(3)
CUMO_NA_IARRAY_STRIDX_AT(2)
CUMO_NA_IARRAY_STRIDX_AT(0)

__host__ __device__
static inline char*
cumo_na_iarray_stridx_at_dim1(cumo_na_iarray_stridx_t* iarray, cumo_na_indexer_t* indexer) {
    if (CUMO_SDX_IS_INDEX(iarray->stridx[0])) {
        return iarray->ptr + CUMO_SDX_GET_INDEX(iarray->stridx[0])[indexer->raw_index];
    } else {
        return iarray->ptr + CUMO_SDX_GET_STRIDE(iarray->stridx[0]) * indexer->raw_index;
    }
}

#endif // #ifdef __CUDACC__

#endif // CUMO_INDEXER_H
