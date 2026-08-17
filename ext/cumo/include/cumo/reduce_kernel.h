#ifndef CUMO_REDUCE_KERNEL_H
#define CUMO_REDUCE_KERNEL_H

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "cumo/indexer.h"

namespace cumo_detail {

static constexpr int64_t max_block_size = 512;
static constexpr int64_t max_grid_size = 0x7fffffff;

// A reduction gets one block per output element, so reducing to a single value
// leaves the whole grid at one block however long the reduce axis is. Below
// this many blocks the axis is worth splitting across a second launch.
static constexpr int64_t min_grid_size = 256;
// Splitting costs a scratch buffer and a second launch, so keep a chunk that
// still has work in it, and do not go wider than the combine pass can take.
static constexpr int64_t min_split_chunk = 1024;
static constexpr int64_t max_split = 1024;

static inline int64_t round_up_to_power_of_2(int64_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

// Reference: cupy reduction kernel
// Note that reduction and out axis are inverse with cupy. Former axes are out axes, latters are reduce axes.

template <typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_kernel(cumo_na_reduction_arg_t arg, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    cumo_na_iarray_t& in_iarray = arg.in;
    cumo_na_iarray_t& out_iarray = arg.out;
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t reduce_indexer_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t reduce_offset = tid / out_block_size; // # of cols == # of elems

    int64_t out_offset = tid % out_block_size; // # of rows
    int64_t out_base = blockIdx.x * out_block_size; // # of rows
    int64_t out_stride = gridDim.x * out_block_size; // # of rows

    for (int64_t i_out = out_base + out_offset; i_out < out_indexer.total_size; i_out += out_stride) {
        cumo_na_indexer_set_dim(&out_indexer, i_out);
        int64_t i_in = i_out * reduce_indexer_total_size + reduce_offset;

        // Note that spec of (min|max)_index of cumo is different with arg(min|max) of cupy.
        // Cumo returns index of input elements, CuPy returns index of reduction axis.
        cumo_na_indexer_set_dim(&in_indexer, i_in);
        TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
        TypeReduce accum = impl.Identity(in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr));

        for (int64_t i_reduce = reduce_offset; i_reduce < reduce_indexer_total_size; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            cumo_na_indexer_set_dim(&in_indexer, i_in);
            in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            impl.Reduce(impl.MapIn(*in_ptr, in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr)), accum);
            //printf("threadId.x:%d blockIdx.x:%d blockDim.x:%d gridDim.x:%d accum:%d i_in:%ld i_reduce:%ld i_out:%ld in:%p(%d)\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, accum, i_in, i_reduce, i_out, in_ptr, *in_ptr);
        }

        if (out_block_size <= max_block_size / 2) {
            sdata[tid] = accum;
            __syncthreads();
            // NOTE: Compiler optimizes to unroll this loop
            for (int stride = max_block_size / 2; stride > 0; stride >>= 1) {
                if (out_block_size <= stride) {
                    if (tid < stride) {
                        impl.Reduce(sdata[tid + stride], sdata[tid]);
                    }
                    __syncthreads();
                }
            }
            accum = sdata[tid];
            __syncthreads();
        }
        if (reduce_offset == 0 && i_out < out_indexer.total_size) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(cumo_na_iarray_at_dim(&out_iarray, &out_indexer));
            *out_ptr = impl.MapOut(accum);
            //printf("threadId.x:%d blockIdx.x:%d blockDim.x:%d gridDim.x:%d accum:%d i_out:%ld out:%p(%d)\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, accum, i_out, out_ptr, *out_ptr);
        }
    }
}

// Variant of reduction_kernel for arg-reductions (argmax/argmin).
//
// Unlike reduction_kernel, which passes the flat 1-d index of an input element
// (i.e. index of input elements, same as (min|max)_index), this passes the
// index along the reduction axis (i_reduce) to the reduction impl. This matches
// the spec of arg(min|max) which returns indices along the axis.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_arg_kernel(cumo_na_reduction_arg_t arg, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    cumo_na_iarray_t& in_iarray = arg.in;
    cumo_na_iarray_t& out_iarray = arg.out;
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t reduce_indexer_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t reduce_offset = tid / out_block_size; // # of cols == # of elems

    int64_t out_offset = tid % out_block_size; // # of rows
    int64_t out_base = blockIdx.x * out_block_size; // # of rows
    int64_t out_stride = gridDim.x * out_block_size; // # of rows

    for (int64_t i_out = out_base + out_offset; i_out < out_indexer.total_size; i_out += out_stride) {
        cumo_na_indexer_set_dim(&out_indexer, i_out);
        int64_t i_in = i_out * reduce_indexer_total_size + reduce_offset;

        // Note that arg(min|max) returns the index along the reduction axis.
        TypeReduce accum = impl.Identity(reduce_offset);

        for (int64_t i_reduce = reduce_offset; i_reduce < reduce_indexer_total_size; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            cumo_na_indexer_set_dim(&in_indexer, i_in);
            TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            impl.Reduce(impl.MapIn(*in_ptr, i_reduce), accum);
        }

        if (out_block_size <= max_block_size / 2) {
            sdata[tid] = accum;
            __syncthreads();
            // NOTE: Compiler optimizes to unroll this loop
            for (int stride = max_block_size / 2; stride > 0; stride >>= 1) {
                if (out_block_size <= stride) {
                    if (tid < stride) {
                        impl.Reduce(sdata[tid + stride], sdata[tid]);
                    }
                    __syncthreads();
                }
            }
            accum = sdata[tid];
            __syncthreads();
        }
        if (reduce_offset == 0 && i_out < out_indexer.total_size) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(cumo_na_iarray_at_dim(&out_iarray, &out_indexer));
            *out_ptr = impl.MapOut(accum);
        }
    }
}

// Variant of reduction_kernel writing two results per output element, so that
// minmax reads the input once instead of reducing it twice. The impl stores
// through the pointers rather than returning, since there are two of them.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_pair_kernel(cumo_na_reduction_arg_t arg, cumo_na_iarray_t out2_iarray, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    cumo_na_iarray_t& in_iarray = arg.in;
    cumo_na_iarray_t& out_iarray = arg.out;
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t reduce_indexer_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t reduce_offset = tid / out_block_size;

    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_indexer.total_size; i_out += out_stride) {
        cumo_na_indexer_set_dim(&out_indexer, i_out);
        int64_t i_in = i_out * reduce_indexer_total_size + reduce_offset;

        cumo_na_indexer_set_dim(&in_indexer, i_in);
        TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
        TypeReduce accum = impl.Identity(in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr));

        for (int64_t i_reduce = reduce_offset; i_reduce < reduce_indexer_total_size; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            cumo_na_indexer_set_dim(&in_indexer, i_in);
            in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            impl.Reduce(impl.MapIn(*in_ptr, in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr)), accum);
        }

        if (out_block_size <= max_block_size / 2) {
            sdata[tid] = accum;
            __syncthreads();
            for (int stride = max_block_size / 2; stride > 0; stride >>= 1) {
                if (out_block_size <= stride) {
                    if (tid < stride) {
                        impl.Reduce(sdata[tid + stride], sdata[tid]);
                    }
                    __syncthreads();
                }
            }
            accum = sdata[tid];
            __syncthreads();
        }
        if (reduce_offset == 0 && i_out < out_indexer.total_size) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(cumo_na_iarray_at_dim(&out_iarray, &out_indexer));
            TypeOut* out2_ptr = reinterpret_cast<TypeOut*>(cumo_na_iarray_at_dim(&out2_iarray, &out_indexer));
            impl.MapOut(accum, out_ptr, out2_ptr);
        }
    }
}

static inline int64_t reduce_split_count(int64_t reduce_total_size, int64_t out_block_num) {
    if (out_block_num >= min_grid_size) return 1;
    if (reduce_total_size < min_split_chunk * 2) return 1;
    int64_t want = (min_grid_size + out_block_num - 1) / out_block_num;
    int64_t fits = reduce_total_size / min_split_chunk;
    int64_t n = std::min(std::min(want, fits), max_split);
    return n < 2 ? 1 : n;
}

// First pass of a split reduction: each block takes one chunk of one output's
// reduce axis and writes its accumulator to partial[i_out * n_split + i_split].
// MapOut is deliberately not applied — the combine pass needs the accumulator.
template <typename TypeIn, typename TypeReduce, typename ReductionImpl>
__global__ static void reduction_partial_kernel(cumo_na_reduction_arg_t arg, TypeReduce* partial, int64_t n_split, int64_t chunk, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    cumo_na_iarray_t& in_iarray = arg.in;
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t reduce_indexer_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t partial_total_size = out_indexer.total_size * n_split;
    int64_t reduce_offset = tid / out_block_size;

    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_partial = out_base + out_offset; i_partial < partial_total_size; i_partial += out_stride) {
        int64_t i_out = i_partial / n_split;
        int64_t begin = (i_partial % n_split) * chunk;
        int64_t end = begin + chunk;
        if (end > reduce_indexer_total_size) end = reduce_indexer_total_size;
        int64_t i_in = i_out * reduce_indexer_total_size + begin + reduce_offset;

        cumo_na_indexer_set_dim(&in_indexer, i_in);
        TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
        TypeReduce accum = impl.Identity(in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr));

        for (int64_t i_reduce = begin + reduce_offset; i_reduce < end; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            cumo_na_indexer_set_dim(&in_indexer, i_in);
            in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            impl.Reduce(impl.MapIn(*in_ptr, in_ptr - reinterpret_cast<TypeIn*>(in_iarray.ptr)), accum);
        }

        if (out_block_size <= max_block_size / 2) {
            sdata[tid] = accum;
            __syncthreads();
            for (int stride = max_block_size / 2; stride > 0; stride >>= 1) {
                if (out_block_size <= stride) {
                    if (tid < stride) {
                        impl.Reduce(sdata[tid + stride], sdata[tid]);
                    }
                    __syncthreads();
                }
            }
            accum = sdata[tid];
            __syncthreads();
        }
        if (reduce_offset == 0 && i_partial < partial_total_size) {
            partial[i_partial] = accum;
        }
    }
}

// Second pass of a split reduction. The input is already accumulators, so
// MapIn must not run again. Identity is handed a scratch offset rather than an
// index into the input, which is why an impl that reads that argument — the
// (min|max)_index and arg(min|max) pairs — must not be split.
template <typename ReductionImpl>
struct reduce_combine {
    ReductionImpl impl;
    using TypeReduce = decltype(ReductionImpl().Identity(0));
    __device__ TypeReduce Identity(int64_t index) { return impl.Identity(index); }
    __device__ TypeReduce MapIn(TypeReduce in, int64_t /*index*/) { return in; }
    __device__ void Reduce(TypeReduce next, TypeReduce& accum) { impl.Reduce(next, accum); }
    template <typename... Args>
    __device__ decltype(auto) MapOut(TypeReduce accum, Args... args) { return impl.MapOut(accum, args...); }
};

// Allocates the scratch and runs the first pass, then points arg2's input at
// the partials so the caller can combine them with its own second pass. The
// caller frees the returned buffer.
template <typename TypeIn, typename TypeReduce, typename ReductionImpl>
TypeReduce* reduce_partial_pass(cumo_na_reduction_arg_t arg, int64_t n_split, int64_t reduce_total_size, cumo_na_reduction_arg_t* arg2, ReductionImpl& impl) {
    int64_t chunk = (reduce_total_size + n_split - 1) / n_split;
    int64_t partial_total_size = arg.out_indexer.total_size * n_split;
    TypeReduce* partial = reinterpret_cast<TypeReduce*>(cumo_cuda_runtime_malloc(sizeof(TypeReduce) * partial_total_size));

    int64_t chunk_pow2 = round_up_to_power_of_2(std::max(int64_t{1}, chunk));
    int64_t reduce_block_size = std::min(max_block_size, chunk_pow2);
    int64_t out_block_size = max_block_size / reduce_block_size;
    int64_t out_block_num = (partial_total_size + out_block_size - 1) / out_block_size;
    int64_t grid_size = std::min(max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(TypeReduce) * max_block_size;

    reduction_partial_kernel<TypeIn,TypeReduce,ReductionImpl><<<grid_size, max_block_size, shared_mem_size>>>(arg, partial, n_split, chunk, out_block_size, reduce_block_size, impl);
    cumo_cuda_runtime_check_kernel_launch();

    arg2->in.ptr = reinterpret_cast<char*>(partial);
    arg2->in.step[0] = sizeof(TypeReduce);
    arg2->in_indexer.ndim = 1;
    arg2->in_indexer.shape[0] = partial_total_size;
    arg2->in_indexer.total_size = partial_total_size;
    return partial;
}

}  // cumo_detail

// TODO(sonots): Optimize indexer by squashing (or reducing) dimensions
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    if (out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size_pow2 = cumo_detail::round_up_to_power_of_2(std::max(size_t{1}, in_indexer.total_size / out_indexer.total_size));
    int64_t reduce_block_size = std::min(cumo_detail::max_block_size, reduce_total_size_pow2);
    int64_t out_block_size = cumo_detail::max_block_size / reduce_block_size;
    int64_t out_block_num = (out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    cumo_detail::reduction_kernel<TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out_block_size, reduce_block_size, impl);
    cumo_cuda_runtime_check_kernel_launch();
}

// Runs the reduce axis of one output across several blocks and combines their
// accumulators in a second launch, for the shapes where cumo_reduce would give
// the grid almost no blocks. Falls back to cumo_reduce when that is not the
// case. Only for an impl whose Identity ignores its index argument — see
// reduce_combine above.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_split(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    using TypeReduce = decltype(impl.Identity(0));
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    if (out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t reduce_total_size_pow2 = cumo_detail::round_up_to_power_of_2(std::max(int64_t{1}, reduce_total_size));
    int64_t reduce_block_size = std::min(cumo_detail::max_block_size, reduce_total_size_pow2);
    int64_t out_block_size = cumo_detail::max_block_size / reduce_block_size;
    int64_t out_block_num = (out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce<TypeIn, TypeOut, ReductionImpl>(arg, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t arg2 = arg;
    TypeReduce* partial = cumo_detail::reduce_partial_pass<TypeIn, TypeReduce, ReductionImpl>(arg, n_split, reduce_total_size, &arg2, impl);
    cumo_reduce<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(arg2, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// Variant of cumo_reduce writing two results per output element, for minmax.
// See reduction_pair_kernel above. out2 has to describe the same shape as
// arg.out, since the one out_indexer addresses both.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_pair(cumo_na_reduction_arg_t arg, cumo_na_iarray_t out2, ReductionImpl&& impl) {
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    if (out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size_pow2 = cumo_detail::round_up_to_power_of_2(std::max(size_t{1}, in_indexer.total_size / out_indexer.total_size));
    int64_t reduce_block_size = std::min(cumo_detail::max_block_size, reduce_total_size_pow2);
    int64_t out_block_size = cumo_detail::max_block_size / reduce_block_size;
    int64_t out_block_num = (out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    cumo_detail::reduction_pair_kernel<TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out2, out_block_size, reduce_block_size, impl);
    cumo_cuda_runtime_check_kernel_launch();
}

// cumo_reduce_split for the two-output form. Same constraint on Identity.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_pair_split(cumo_na_reduction_arg_t arg, cumo_na_iarray_t out2, ReductionImpl&& impl) {
    using TypeReduce = decltype(impl.Identity(0));
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    if (out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = in_indexer.total_size / out_indexer.total_size;
    int64_t reduce_total_size_pow2 = cumo_detail::round_up_to_power_of_2(std::max(int64_t{1}, reduce_total_size));
    int64_t reduce_block_size = std::min(cumo_detail::max_block_size, reduce_total_size_pow2);
    int64_t out_block_size = cumo_detail::max_block_size / reduce_block_size;
    int64_t out_block_num = (out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce_pair<TypeIn, TypeOut, ReductionImpl>(arg, out2, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t arg2 = arg;
    TypeReduce* partial = cumo_detail::reduce_partial_pass<TypeIn, TypeReduce, ReductionImpl>(arg, n_split, reduce_total_size, &arg2, impl);
    cumo_reduce_pair<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(arg2, out2, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// Variant of cumo_reduce for arg-reductions (argmax/argmin), which returns
// indices along the reduction axis. See reduction_arg_kernel above.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_arg(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    cumo_na_indexer_t& in_indexer = arg.in_indexer;
    cumo_na_indexer_t& out_indexer = arg.out_indexer;

    if (out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size_pow2 = cumo_detail::round_up_to_power_of_2(std::max(size_t{1}, in_indexer.total_size / out_indexer.total_size));
    int64_t reduce_block_size = std::min(cumo_detail::max_block_size, reduce_total_size_pow2);
    int64_t out_block_size = cumo_detail::max_block_size / reduce_block_size;
    int64_t out_block_num = (out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    cumo_detail::reduction_arg_kernel<TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out_block_size, reduce_block_size, impl);
    cumo_cuda_runtime_check_kernel_launch();
}

#endif // CUMO_REDUCE_KERNEL_H
