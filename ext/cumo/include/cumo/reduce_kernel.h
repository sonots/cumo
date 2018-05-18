#ifndef CUMO_REDUCE_KERNEL_H
#define CUMO_REDUCE_KERNEL_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cumo/indexer.h"

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

#define _REDUCE(offset) \
    if (tid < offset) { \
        impl.Reduce(sdata[(tid + offset)], sdata[tid]); \
    }

// reference: cupy reduction kernel

template <typename TypeIn, typename TypeOut, typename TypeReduce, typename ReductionImpl>
__global__ static void reduction_kernel(na_reduction_arg_t arg, ReductionImpl impl) {
    na_iarray_t& in_iarray = arg.in;
    na_iarray_t& out_iarray = arg.out;
    na_indexer_t& in_indexer = arg.in_indexer;
    na_indexer_t& out_indexer = arg.out_indexer;
    na_indexer_t& reduce_indexer = arg.reduce_indexer;

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = (TypeReduce*)sdata_raw;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;  // number of threads

    for (uint64_t i_out = blockIdx.x; i_out < out_indexer.total_size; i_out += gridDim.x) {
        cumo_na_indexer_set_dim(&out_indexer, i_out);
        TypeReduce accum = impl.Identity();

        for (int8_t i_out_dim = 0; i_out_dim < out_indexer.ndim; ++i_out_dim) {
            in_indexer.index[i_out_dim] = out_indexer.index[i_out_dim];
        }
        for (auto i_reduce = tid; i_reduce < reduce_indexer.total_size; i_reduce += block_size) {
            cumo_na_indexer_set_dim(&reduce_indexer, i_reduce);
            for (int8_t i_reduce_dim = 0; i_reduce_dim < reduce_indexer.ndim; ++i_reduce_dim) {
                in_indexer.index[out_indexer.ndim + i_reduce_dim] = reduce_indexer.index[i_reduce_dim];
            }
            TypeIn* in_ptr = reinterpret_cast<TypeIn*>(cumo_na_iarray_at_dim(&in_iarray, &in_indexer));
            impl.Reduce(impl.MapIn(*in_ptr, i_reduce), accum);
        }

        if (block_size >= 2) {
            sdata[tid] = accum;
            __syncthreads();

            if (block_size > 2) {
                if (block_size > 4) {
                    if (block_size > 8) {
                        if (block_size > 16) {
                            if (block_size > 32) {
                                if (block_size > 64) {
                                    if (block_size > 128) {
                                        if (block_size > 256) {
                                            _REDUCE(256);
                                            __syncthreads();
                                        }
                                        _REDUCE(128);
                                        __syncthreads();
                                    }
                                    _REDUCE(64);
                                    __syncthreads();
                                }
                                _REDUCE(32);
                                __syncthreads();
                            }
                            _REDUCE(16);
                            __syncthreads();
                        }
                        _REDUCE(8);
                        __syncthreads();
                    }
                    _REDUCE(4);
                    __syncthreads();
                }
                _REDUCE(2);
                __syncthreads();
            }
            _REDUCE(1);
            accum = sdata[0];
        }
        if (tid == 0) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(cumo_na_iarray_at_dim(&out_iarray, &out_indexer));
            *out_ptr = impl.MapOut(accum);
            //printf("threadId.x:%d blockIdx.x:%d blockDim.x:%d gridDim.x:%d block_size:%d accum:%d out:%p(%d)\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, block_size, accum, out_ptr, *out_ptr);
        }
    }
}

#undef _REDUCE

static constexpr size_t max_block_size = 512;

template <typename TypeIn, typename TypeOut, typename TypeReduce, typename ReductionImpl>
void cumo_reduce(na_reduction_arg_t arg, ReductionImpl&& impl) {
    na_indexer_t& out_indexer = arg.out_indexer;
    na_indexer_t& reduce_indexer = arg.reduce_indexer;

    size_t block_size = round_up_to_power_of_2(std::max(int64_t{1}, static_cast<int64_t>(reduce_indexer.total_size)));
    block_size = std::min(max_block_size, block_size);
    size_t grid_size = out_indexer.total_size;
    size_t shared_mem_size = sizeof(TypeReduce) * block_size;

    reduction_kernel<TypeIn,TypeOut,TypeReduce,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, impl);
}

#endif // CUMO_REDUCE_KERNEL_H
