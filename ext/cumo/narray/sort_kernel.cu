#include "cumo/narray_kernel.h"
#include "cumo/indexer.h"
#include "cumo/template_kernel.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

extern "C" char* cumo_cuda_runtime_malloc(size_t size);
extern "C" void cumo_cuda_runtime_free(char *ptr);
extern "C" void cumo_cuda_runtime_check_kernel_launch(void);

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

template <> struct float_key<float> {
    typedef uint32_t type;
    __device__ static type of(float x) {
        type u = __float_as_uint(x);
        if ((u & 0x7fffffffu) > 0x7f800000u) return ~(type)0;
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
};

template <> struct float_key<double> {
    typedef uint64_t type;
    __device__ static type of(double x) {
        type u = (type)__double_as_longlong(x);
        if ((u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull) return ~(type)0;
        return (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
    }
};

template <typename Float>
__global__ void float_key_kernel(const Float* in, typename float_key<Float>::type* out, uint64_t n) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = float_key<Float>::of(in[i]);
    }
}

// Rows that are not laid out end to end are gathered into a buffer of their
// own, sorted there and put back, which is two passes over the data against
// one launch per row.
template <typename T>
__global__ void gather_kernel(cumo_na_iarray_t a, cumo_na_indexer_t indexer, T* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        buf[i] = *(T*)cumo_na_iarray_at_dim(&a, &indexer);
    }
}

template <typename T>
__global__ void scatter_kernel(cumo_na_iarray_t a, cumo_na_indexer_t indexer, const T* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        *(T*)cumo_na_iarray_at_dim(&a, &indexer) = buf[i];
    }
}

// One CUB call covers the whole array, so a loop over many short rows costs no
// more launches than one long row does.
template <typename Key, typename Value>
void sort_pairs(const Key* kin, Key* kout, const Value* vin, Value* vout,
                int64_t total, int64_t n_rows, int64_t row_len) {
    size_t bytes = 0;
    if (n_rows == 1) {
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, kin, kout, vin, vout, total);
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        cub::DeviceRadixSort::SortPairs(tmp, bytes, kin, kout, vin, vout, total);
        cumo_cuda_runtime_check_kernel_launch();
        cumo_cuda_runtime_free(tmp);
    } else {
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr, bytes, kin, kout, vin, vout, total, n_rows,
                                                 row_begins(row_len), row_ends(row_len));
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        cub::DeviceSegmentedRadixSort::SortPairs(tmp, bytes, kin, kout, vin, vout, total, n_rows,
                                                 row_begins(row_len), row_ends(row_len));
        cumo_cuda_runtime_check_kernel_launch();
        cumo_cuda_runtime_free(tmp);
    }
}

template <typename Key>
void sort_keys(const Key* kin, Key* kout, int64_t total, int64_t n_rows, int64_t row_len) {
    size_t bytes = 0;
    if (n_rows == 1) {
        cub::DeviceRadixSort::SortKeys(nullptr, bytes, kin, kout, total);
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        cub::DeviceRadixSort::SortKeys(tmp, bytes, kin, kout, total);
        cumo_cuda_runtime_check_kernel_launch();
        cumo_cuda_runtime_free(tmp);
    } else {
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, kin, kout, total, n_rows,
                                                row_begins(row_len), row_ends(row_len));
        char* tmp = cumo_cuda_runtime_malloc(bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(tmp, bytes, kin, kout, total, n_rows,
                                               row_begins(row_len), row_ends(row_len));
        cumo_cuda_runtime_check_kernel_launch();
        cumo_cuda_runtime_free(tmp);
    }
}

template <typename T, bool IS_FLOAT>
void sort_rows(cumo_na_iarray_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat) {
    int64_t total = (int64_t)indexer->total_size;
    if (total == 0) return;

    size_t grid_dim = cumo_get_grid_dim(total);
    size_t block_dim = cumo_get_block_dim(total);

    T* data = (T*)a->ptr;
    T* gathered = 0;
    if (!flat) {
        gathered = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
        gather_kernel<T><<<grid_dim, block_dim>>>(*a, *indexer, gathered);
        cumo_cuda_runtime_check_kernel_launch();
        data = gathered;
    }

    T* out = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
    if constexpr (IS_FLOAT) {
        typedef typename float_key<T>::type key_t;
        key_t* kin = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        key_t* kout = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        float_key_kernel<T><<<grid_dim, block_dim>>>(data, kin, total);
        cumo_cuda_runtime_check_kernel_launch();
        sort_pairs(kin, kout, data, out, total, n_rows, row_len);
        cumo_cuda_runtime_free((char*)kout);
        cumo_cuda_runtime_free((char*)kin);
    } else {
        sort_keys(data, out, total, n_rows, row_len);
    }

    if (flat) {
        cudaMemcpyAsync(data, out, sizeof(T) * total, cudaMemcpyDeviceToDevice, 0);
        cumo_cuda_runtime_check_kernel_launch();
    } else {
        scatter_kernel<T><<<grid_dim, block_dim>>>(*a, *indexer, out);
        cumo_cuda_runtime_check_kernel_launch();
        cumo_cuda_runtime_free((char*)gathered);
    }
    cumo_cuda_runtime_free((char*)out);
}

// The rows are already sorted, so the middle of each is the answer. Trailing
// NaNs are dropped first, which is what the host loop this replaces does and
// what numo 0.9 does; numo-narray-alt lost that in a rewrite.
template <typename T, bool IS_FLOAT>
__global__ void median_kernel(const T* sorted, int64_t row_len, cumo_na_iarray_t out, cumo_na_indexer_t out_indexer) {
    for (uint64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < out_indexer.total_size; r += blockDim.x * gridDim.x) {
        const T* row = sorted + (int64_t)r * row_len;
        int64_t n = row_len;
        T v;
        if constexpr (IS_FLOAT) {
            while (n > 0 && isnan(row[n - 1])) --n;
        }
        if (n == 0) {
            v = row[0];
        } else if (n % 2 == 0) {
            v = (row[n / 2 - 1] + row[n / 2]) / 2;
        } else {
            v = row[(n - 1) / 2];
        }
        cumo_na_indexer_set_dim(&out_indexer, r);
        *(T*)cumo_na_iarray_at_dim(&out, &out_indexer) = v;
    }
}

// median throws the sorted rows away, so unlike sort it never has to put them
// back where they came from.
template <typename T, bool IS_FLOAT>
void median_rows(cumo_na_reduction_arg_t* arg, int flat) {
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
        gather_kernel<T><<<grid_dim, block_dim>>>(arg->in, arg->in_indexer, gathered);
        cumo_cuda_runtime_check_kernel_launch();
        data = gathered;
    }

    T* sorted = (T*)cumo_cuda_runtime_malloc(sizeof(T) * total);
    if constexpr (IS_FLOAT) {
        typedef typename float_key<T>::type key_t;
        key_t* kin = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        key_t* kout = (key_t*)cumo_cuda_runtime_malloc(sizeof(key_t) * total);
        float_key_kernel<T><<<grid_dim, block_dim>>>(data, kin, total);
        cumo_cuda_runtime_check_kernel_launch();
        sort_pairs(kin, kout, data, sorted, total, n_rows, row_len);
        cumo_cuda_runtime_free((char*)kout);
        cumo_cuda_runtime_free((char*)kin);
    } else {
        sort_keys(data, sorted, total, n_rows, row_len);
    }

    median_kernel<T, IS_FLOAT><<<cumo_get_grid_dim(n_rows), cumo_get_block_dim(n_rows)>>>(
        sorted, row_len, arg->out, arg->out_indexer);
    cumo_cuda_runtime_check_kernel_launch();

    cumo_cuda_runtime_free((char*)sorted);
    if (gathered) cumo_cuda_runtime_free((char*)gathered);
}

} // namespace

// float_key is only defined for the two float types, so the integer entry
// points must not instantiate it; the bool picks the branch at compile time.
#define CUMO_DEF_SORT(name, type, is_float)                                                     \
    extern "C" void cumo_##name##_sort_kernel_launch(                                           \
        cumo_na_iarray_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat) \
    {                                                                                           \
        sort_rows<type, is_float>(a, indexer, n_rows, row_len, flat);                           \
    }                                                                                           \
    extern "C" void cumo_##name##_median_kernel_launch(cumo_na_reduction_arg_t* arg, int flat)  \
    {                                                                                           \
        median_rows<type, is_float>(arg, flat);                                                 \
    }

CUMO_DEF_SORT(int8, int8_t, false)
CUMO_DEF_SORT(int16, int16_t, false)
CUMO_DEF_SORT(int32, int32_t, false)
CUMO_DEF_SORT(int64, int64_t, false)
CUMO_DEF_SORT(uint8, u_int8_t, false)
CUMO_DEF_SORT(uint16, u_int16_t, false)
CUMO_DEF_SORT(uint32, u_int32_t, false)
CUMO_DEF_SORT(uint64, u_int64_t, false)
CUMO_DEF_SORT(sfloat, float, true)
CUMO_DEF_SORT(dfloat, double, true)
