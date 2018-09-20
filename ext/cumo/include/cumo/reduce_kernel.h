#ifndef CUMO_REDUCE_KERNEL_H
#define CUMO_REDUCE_KERNEL_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cumo/indexer.h"

namespace cumo_detail {

static constexpr int64_t max_block_size = 512;
static constexpr int64_t max_grid_size = 0x7fffffff;

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

    using TypeReduce = decltype(impl.Identity());

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
        TypeReduce accum = impl.Identity();

        int64_t i_in = i_out * reduce_indexer_total_size + reduce_offset;
        for (int64_t i_reduce = reduce_offset; i_reduce < reduce_indexer_total_size; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            cumo_na_indexer_set_dim(&in_indexer, i_in);
            TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            // Note that spec of (min|max)_index of cumo is different with arg(min|max) of cupy.
            // Cumo returns index of input elements, CuPy returns index of reduction axis.
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
    int64_t shared_mem_size = sizeof(decltype(impl.Identity())) * block_size;

    cumo_detail::reduction_kernel<TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out_block_size, reduce_block_size, impl);
}

#endif // CUMO_REDUCE_KERNEL_H
