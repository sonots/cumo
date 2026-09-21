#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"
#include "cumo/template_kernel.h"
#include <string.h>
#include "cumo/types/half_def_kernel.h"
#include "cumo/types/bf16_def_kernel.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {

// Where each row starts, so that the segmented sort needs no offsets array.
struct row_offset {
    int64_t row_len;
    __host__ __device__ __forceinline__ int64_t operator()(int64_t i) const { return i * row_len; }
};

inline auto row_begins(int64_t row_len) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), row_offset{row_len});
}

inline auto row_ends(int64_t row_len) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(1), row_offset{row_len});
}

// The unsigned integer whose ascending order is the order sort wants for the
// float it comes from: negatives reversed and below the positives, -0.0 below
// +0.0, and every NaN above everything else whatever its sign. That last part
// is why a float cannot be its own key -- a radix sort puts a negative NaN
// first, and numo puts every NaN last.
template <typename Float> struct float_key;

// The same construction at three widths: a NaN takes the largest key, a
// negative value reverses, and a positive one has its sign bit set.
template <typename U>
__device__ static inline U order_key(U bits, U inf_bits) {
    const U sign = (U)1 << (sizeof(U) * 8 - 1);
    if ((U)(bits & (U)~sign) > inf_bits) return (U)~(U)0;
    return (bits & sign) ? (U)~bits : (U)(bits | sign);
}

template <> struct float_key<float> {
    typedef uint32_t type;
    __device__ static type of(float x) { return order_key<type>(__float_as_uint(x), 0x7f800000u); }
};

template <> struct float_key<cumo_half> {
    typedef uint16_t type;
    __device__ static type of(cumo_half x) { return order_key<type>(__half_as_ushort(x), 0x7c00u); }
};

// CUDA 11.8 has no __bfloat16_as_ushort, so the bits come out through memcpy.
template <> struct float_key<cumo_bfloat> {
    typedef uint16_t type;
    __device__ static type of(cumo_bfloat x) {
        uint16_t bits;
        memcpy(&bits, &x, sizeof(bits));
        return order_key<type>(bits, 0x7f80u);
    }
};

template <> struct float_key<double> {
    typedef uint64_t type;
    __device__ static type of(double x) { return order_key<type>((type)__double_as_longlong(x), 0x7ff0000000000000ull); }
};

template <typename Float>
__global__ void float_key_kernel(const Float* in, typename float_key<Float>::type* out, uint64_t n) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = float_key<Float>::of(in[i]);
    }
}

// sort writes its answer back where it read it, so an in-place sort of a view
// backed by an index array has to address that array; the other two callers
// hand over an array they copied themselves and never do.
//
// The rank is a template argument rather than a field the kernel reads, so the
// address is a fixed number of multiplies instead of a divide by a run time
// extent per axis per element. rank_at<-1> is the one for ranks past those
// named here, which nothing reaches: the indexer stops at CUMO_NA_MAX_DIMENSION.
template <int NDIM> struct rank_at {
    __device__ static void set(cumo_na_indexer_t* ix, uint64_t i) { cumo_na_indexer_set_dim(ix, i); }
    __device__ static char* at(cumo_na_iarray_t* a, cumo_na_indexer_t* ix) { return cumo_na_iarray_at_dim(a, ix); }
    __device__ static char* at(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* ix) { return cumo_na_iarray_stridx_at_dim(a, ix); }
};

#define CUMO_SORT_RANK_AT(NDIM) \
template <> struct rank_at<NDIM> { \
    __device__ static void set(cumo_na_indexer_t* ix, uint64_t i) { cumo_na_indexer_set_dim##NDIM(ix, i); } \
    __device__ static char* at(cumo_na_iarray_t* a, cumo_na_indexer_t* ix) { return cumo_na_iarray_at_dim##NDIM(a, ix); } \
    __device__ static char* at(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* ix) { return cumo_na_iarray_stridx_at_dim##NDIM(a, ix); } \
};
CUMO_SORT_RANK_AT(0)
CUMO_SORT_RANK_AT(1)
CUMO_SORT_RANK_AT(2)
CUMO_SORT_RANK_AT(3)
CUMO_SORT_RANK_AT(4)
CUMO_SORT_RANK_AT(5)
CUMO_SORT_RANK_AT(6)
CUMO_SORT_RANK_AT(7)
CUMO_SORT_RANK_AT(8)
#undef CUMO_SORT_RANK_AT

// LAUNCH names the kernel with the rank filled in, so the switch is written
// once rather than at each of the four calls.
#define CUMO_SORT_BY_RANK(NDIM_OF, LAUNCH) \
    switch (NDIM_OF) { \
    case 0: LAUNCH(0); break; \
    case 1: LAUNCH(1); break; \
    case 2: LAUNCH(2); break; \
    case 3: LAUNCH(3); break; \
    case 4: LAUNCH(4); break; \
    case 5: LAUNCH(5); break; \
    case 6: LAUNCH(6); break; \
    case 7: LAUNCH(7); break; \
    case 8: LAUNCH(8); break; \
    default: LAUNCH(-1); break; \
    }

// Rows that are not laid out end to end are gathered into a buffer of their
// own, sorted there and put back, which is two passes over the data against
// one launch per row.
template <int NDIM, typename T, typename Iarray>
__global__ void gather_kernel(Iarray a, cumo_na_indexer_t indexer, T* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        rank_at<NDIM>::set(&indexer, i);
        buf[i] = *(T*)rank_at<NDIM>::at(&a, &indexer);
    }
}

template <int NDIM, typename T, typename Iarray>
__global__ void scatter_kernel(Iarray a, cumo_na_indexer_t indexer, const T* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        rank_at<NDIM>::set(&indexer, i);
        *(T*)rank_at<NDIM>::at(&a, &indexer) = buf[i];
    }
}

// One CUB call covers the whole array, so a loop over many short rows costs no
// more launches than one long row does.
template <typename Key, typename Value>
cudaError_t sort_pairs(const Key* kin, Key* kout, const Value* vin, Value* vout,
                       int64_t total, int64_t n_rows, int64_t row_len) {
    size_t bytes = 0;
    cudaError_t st = cudaSuccess;
    if (n_rows == 1) {
        st = cub::DeviceRadixSort::SortPairs(nullptr, bytes, kin, kout, vin, vout, total, 0, sizeof(Key) * 8, cumo_cuda_stream());
        if (st != cudaSuccess) { return st; }
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        st = cub::DeviceRadixSort::SortPairs(tmp, bytes, kin, kout, vin, vout, total, 0, sizeof(Key) * 8, cumo_cuda_stream());
        cumo_cuda_runtime_free(tmp);
    } else {
        st = cub::DeviceSegmentedRadixSort::SortPairs(nullptr, bytes, kin, kout, vin, vout, total, n_rows,
                                                     row_begins(row_len), row_ends(row_len), 0, sizeof(Key) * 8, cumo_cuda_stream());
        if (st != cudaSuccess) { return st; }
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        st = cub::DeviceSegmentedRadixSort::SortPairs(tmp, bytes, kin, kout, vin, vout, total, n_rows,
                                                 row_begins(row_len), row_ends(row_len), 0, sizeof(Key) * 8, cumo_cuda_stream());
        cumo_cuda_runtime_free(tmp);
    }
    return st;
}

template <typename Key>
cudaError_t sort_keys(const Key* kin, Key* kout, int64_t total, int64_t n_rows, int64_t row_len) {
    size_t bytes = 0;
    cudaError_t st = cudaSuccess;
    if (n_rows == 1) {
        st = cub::DeviceRadixSort::SortKeys(nullptr, bytes, kin, kout, total, 0, sizeof(Key) * 8, cumo_cuda_stream());
        if (st != cudaSuccess) { return st; }
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        st = cub::DeviceRadixSort::SortKeys(tmp, bytes, kin, kout, total, 0, sizeof(Key) * 8, cumo_cuda_stream());
        cumo_cuda_runtime_free(tmp);
    } else {
        st = cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, kin, kout, total, n_rows,
                                                    row_begins(row_len), row_ends(row_len), 0, sizeof(Key) * 8, cumo_cuda_stream());
        if (st != cudaSuccess) { return st; }
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        st = cub::DeviceSegmentedRadixSort::SortKeys(tmp, bytes, kin, kout, total, n_rows,
                                               row_begins(row_len), row_ends(row_len), 0, sizeof(Key) * 8, cumo_cuda_stream());
        cumo_cuda_runtime_free(tmp);
    }
    return st;
}

template <typename T, bool IS_FLOAT>
void sort_rows(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat) {
    int64_t total = (int64_t)indexer->total_size;
    if (total == 0) return;

    size_t grid_dim = cumo_get_grid_dim(total);
    size_t block_dim = cumo_get_block_dim(total);

    T* data = (T*)a->ptr;
    T* gathered = 0;
    if (!flat) {
        gathered = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
#define CUMO_SORT_L(N) gather_kernel<N><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, gathered)
        CUMO_SORT_BY_RANK(indexer->ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
        cumo_check_launch_holding(gathered);
        data = gathered;
    }

    T* out = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
    if constexpr (IS_FLOAT) {
        typedef typename float_key<T>::type key_t;
        key_t* kin = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        key_t* kout = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        float_key_kernel<T><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(data, kin, total);
        cumo_check_launch_holding(kin, kout, out, gathered);
        cumo_check_status_holding(sort_pairs(kin, kout, data, out, total, n_rows, row_len),
                             kin, kout, out, gathered);
        cumo_cuda_runtime_free((char*)kout);
        cumo_cuda_runtime_free((char*)kin);
    } else {
        cumo_check_status_holding(sort_keys(data, out, total, n_rows, row_len), out, gathered);
    }

    if (flat) {
        cudaMemcpyAsync(data, out, sizeof(T) * total, cudaMemcpyDeviceToDevice, cumo_cuda_stream());
    } else {
#define CUMO_SORT_L(N) scatter_kernel<N><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, out)
        CUMO_SORT_BY_RANK(indexer->ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
    }
    cumo_check_launch_holding(out, gathered);
    if (gathered) cumo_cuda_runtime_free((char*)gathered);
    cumo_cuda_runtime_free((char*)out);
}

// The rows are already sorted, so the middle of each is the answer. Trailing
// NaNs are dropped first, which is what the host loop this replaces does and
// what numo 0.9 does; numo-narray-alt lost that in a rewrite.
// A half carries no operator of its own below sm_53 and a bfloat16 none below
// sm_80, and isnan takes neither at all, so both go through float.
template <typename T> __device__ static inline bool sorted_isnan(T x) { return isnan(x); }
template <> __device__ inline bool sorted_isnan<cumo_half>(cumo_half x) { return isnan(cumo_half2float(x)); }
template <> __device__ inline bool sorted_isnan<cumo_bfloat>(cumo_bfloat x) { return isnan(cumo_bfloat2float(x)); }

template <typename T> __device__ static inline T sorted_midpoint(T a, T b) { return (a + b) / 2; }
template <> __device__ inline cumo_half sorted_midpoint<cumo_half>(cumo_half a, cumo_half b) {
    return cumo_float2half((cumo_half2float(a) + cumo_half2float(b)) / 2.0f);
}
template <> __device__ inline cumo_bfloat sorted_midpoint<cumo_bfloat>(cumo_bfloat a, cumo_bfloat b) {
    return cumo_float2bfloat((cumo_bfloat2float(a) + cumo_bfloat2float(b)) / 2.0f);
}

template <int NDIM, typename T, bool IS_FLOAT>
__global__ void median_kernel(const T* sorted, int64_t row_len, int prnan, cumo_na_iarray_t out, cumo_na_indexer_t out_indexer) {
    for (uint64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < out_indexer.total_size; r += blockDim.x * gridDim.x) {
        const T* row = sorted + (int64_t)r * row_len;
        int64_t n = row_len;
        T v;
        if constexpr (IS_FLOAT) {
            while (n > 0 && sorted_isnan(row[n - 1])) --n;
        }
        if (prnan && n < row_len) {
            v = row[row_len - 1];
        } else if (n == 0) {
            v = row[0];
        } else if (n % 2 == 0) {
            v = sorted_midpoint(row[n / 2 - 1], row[n / 2]);
        } else {
            v = row[(n - 1) / 2];
        }
        rank_at<NDIM>::set(&out_indexer, r);
        *(T*)rank_at<NDIM>::at(&out, &out_indexer) = v;
    }
}

// median throws the sorted rows away, so unlike sort it never has to put them
// back where they came from.
template <typename T, bool IS_FLOAT>
void median_rows(cumo_na_reduction_arg_t* arg, int flat, int prnan) {
    int64_t total = (int64_t)arg->in_indexer.total_size;
    int64_t n_rows = (int64_t)arg->out_indexer.total_size;
    if (total == 0 || n_rows == 0) return;
    int64_t row_len = total / n_rows;

    size_t grid_dim = cumo_get_grid_dim(total);
    size_t block_dim = cumo_get_block_dim(total);

    T* data = (T*)arg->in.ptr;
    T* gathered = 0;
    if (!flat) {
        gathered = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
#define CUMO_SORT_L(N) gather_kernel<N><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(arg->in, arg->in_indexer, gathered)
        CUMO_SORT_BY_RANK(arg->in_indexer.ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
        cumo_check_launch_holding(gathered);
        data = gathered;
    }

    T* sorted = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
    if constexpr (IS_FLOAT) {
        typedef typename float_key<T>::type key_t;
        key_t* kin = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        key_t* kout = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        float_key_kernel<T><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(data, kin, total);
        cumo_check_launch_holding(kin, kout, sorted, gathered);
        cumo_check_status_holding(sort_pairs(kin, kout, data, sorted, total, n_rows, row_len),
                             kin, kout, sorted, gathered);
        cumo_cuda_runtime_free((char*)kout);
        cumo_cuda_runtime_free((char*)kin);
    } else {
        cumo_check_status_holding(sort_keys(data, sorted, total, n_rows, row_len), sorted, gathered);
    }

#define CUMO_SORT_L(N) median_kernel<N, T, IS_FLOAT><<<cumo_get_grid_dim(n_rows), cumo_get_block_dim(n_rows), 0, cumo_cuda_stream()>>>( \
        sorted, row_len, prnan, arg->out, arg->out_indexer)
    CUMO_SORT_BY_RANK(arg->out_indexer.ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
    cumo_check_launch_holding(sorted, gathered);

    cumo_cuda_runtime_free((char*)sorted);
    if (gathered) cumo_cuda_runtime_free((char*)gathered);
}

// sort_index answers where each element came from, so the sort carries the
// position along and the values it sorts are thrown away.
template <typename I>
__global__ void iota_kernel(I* out, int64_t n) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)n; i += blockDim.x * gridDim.x) {
        out[i] = (I)i;
    }
}

// perm[i] is the position the i-th smallest element sat at, and the index
// array holds the number to answer for that position.
template <int NDIM, typename I>
__global__ void sort_index_scatter_kernel(cumo_na_iarray_t idx, cumo_na_iarray_t out,
                                          cumo_na_indexer_t indexer, const I* perm) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        rank_at<NDIM>::set(&indexer, (uint64_t)perm[i]);
        I v = *(I*)rank_at<NDIM>::at(&idx, &indexer);
        rank_at<NDIM>::set(&indexer, i);
        *(I*)rank_at<NDIM>::at(&out, &indexer) = v;
    }
}

template <typename T, bool IS_FLOAT, typename I>
void sort_index_rows(cumo_na_iarray_t* a, cumo_na_indexer_t* indexer, cumo_na_iarray_t* idx,
                     cumo_na_iarray_t* out, int64_t n_rows, int64_t row_len, int flat) {
    int64_t total = (int64_t)indexer->total_size;
    if (total == 0) return;

    size_t grid_dim = cumo_get_grid_dim(total);
    size_t block_dim = cumo_get_block_dim(total);

    T* data = (T*)a->ptr;
    T* gathered = 0;
    if (!flat) {
        gathered = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
#define CUMO_SORT_L(N) gather_kernel<N><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, gathered)
        CUMO_SORT_BY_RANK(indexer->ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
        cumo_check_launch_holding(gathered);
        data = gathered;
    }

    I* pin = (I*)cumo_cuda_runtime_malloc(sizeof(I) * total);
    I* pout = (I*)cumo_cuda_runtime_malloc(sizeof(I) * total);
    iota_kernel<I><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(pin, total);
    cumo_check_launch_holding(pin, pout, gathered);

    if constexpr (IS_FLOAT) {
        typedef typename float_key<T>::type key_t;
        key_t* kin = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        key_t* kout = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        float_key_kernel<T><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(data, kin, total);
        cumo_check_launch_holding(kin, kout, pin, pout, gathered);
        cumo_check_status_holding(sort_pairs(kin, kout, pin, pout, total, n_rows, row_len),
                             kin, kout, pin, pout, gathered);
        cumo_cuda_runtime_free((char*)kout);
        cumo_cuda_runtime_free((char*)kin);
    } else {
        T* kout = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
        cumo_check_status_holding(sort_pairs(data, kout, pin, pout, total, n_rows, row_len),
                             kout, pin, pout, gathered);
        cumo_cuda_runtime_free((char*)kout);
    }

#define CUMO_SORT_L(N) sort_index_scatter_kernel<N><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*idx, *out, *indexer, pout)
    CUMO_SORT_BY_RANK(indexer->ndim, CUMO_SORT_L);
#undef CUMO_SORT_L
    cumo_check_launch_holding(pin, pout, gathered);

    cumo_cuda_runtime_free((char*)pout);
    cumo_cuda_runtime_free((char*)pin);
    if (gathered) cumo_cuda_runtime_free((char*)gathered);
}

} // namespace

// float_key is only defined for the two float types, so the integer entry
// points must not instantiate it; the bool picks the branch at compile time.
#define CUMO_DEF_SORT(name, type, is_float)                                                     \
    extern "C" void cumo_##name##_sort_kernel_launch(                                           \
        cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat) \
    {                                                                                           \
        sort_rows<type, is_float>(a, indexer, n_rows, row_len, flat);                           \
    }                                                                                           \
    extern "C" void cumo_##name##_median_kernel_launch(                                         \
        cumo_na_reduction_arg_t* arg, int flat, int prnan)                                       \
    {                                                                                           \
        median_rows<type, is_float>(arg, flat, prnan);                                          \
    }                                                                                           \
    extern "C" void cumo_##name##_sort_index_kernel_launch(                                     \
        cumo_na_iarray_t* a, cumo_na_indexer_t* indexer, cumo_na_iarray_t* idx,                 \
        cumo_na_iarray_t* out, int64_t n_rows, int64_t row_len, int flat, int idx_bytes)        \
    {                                                                                           \
        if (idx_bytes == 4) {                                                                   \
            sort_index_rows<type, is_float, int32_t>(a, indexer, idx, out, n_rows, row_len, flat); \
        } else {                                                                                \
            sort_index_rows<type, is_float, int64_t>(a, indexer, idx, out, n_rows, row_len, flat); \
        }                                                                                       \
    }

CUMO_DEF_SORT(int8, int8_t, false)
CUMO_DEF_SORT(int16, int16_t, false)
CUMO_DEF_SORT(int32, int32_t, false)
CUMO_DEF_SORT(int64, int64_t, false)
CUMO_DEF_SORT(uint8, u_int8_t, false)
CUMO_DEF_SORT(uint16, u_int16_t, false)
CUMO_DEF_SORT(uint32, u_int32_t, false)
CUMO_DEF_SORT(uint64, u_int64_t, false)
CUMO_DEF_SORT(hfloat, cumo_half, true)
CUMO_DEF_SORT(bfloat, cumo_bfloat, true)
CUMO_DEF_SORT(sfloat, float, true)
CUMO_DEF_SORT(dfloat, double, true)
