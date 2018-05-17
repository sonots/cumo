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
__global__ static void reduction_kernel(na_reduction_arg_t arg, int block_stride, ReductionImpl impl) {
    na_iarray_t& in_iarray = arg.in;
    na_iarray_t& out_iarray = arg.out;
    na_indexer_t& in_indexer = arg.in_indexer;
    na_indexer_t& out_indexer = arg.out_indexer;
    na_indexer_t& reduce_indexer = arg.reduce_indexer;

    extern __shared__ TypeReduce sdata_raw[];
    TypeReduce *sdata = sdata_raw;
    unsigned int tid = threadIdx.x;

    // TODO(sonots): What does _J and _j mean?
    int _J_offset = tid / block_stride;
    int _j_offset = _J_offset * out_indexer.total_size;
    int _J_stride = 512 / block_stride;
    long long _j_stride = (long long)_J_stride * out_indexer.total_size;
    for (int i_base = blockIdx.x * block_stride; i_base < out_indexer.total_size; i_base += gridDim.x * block_stride) {
        TypeReduce accum = impl.Identity();

        int i_out = i_base + tid % block_stride;
        int _J = _J_offset;

        cumo_na_indexer_set_dim(&out_indexer, i_out);

        for (int8_t i_out_dim = 0; i_out_dim < out_indexer.ndim; ++i_out_dim) {
            in_indexer.index[i_out_dim] = out_indexer.index[i_out_dim];
        }

        for (long long i_reduce = i_out + _j_offset; i_reduce < in_indexer.total_size; i_reduce += _j_stride, _J += _J_stride) {
            cumo_na_indexer_set_dim(&reduce_indexer, i_reduce);

            for (int8_t i_reduce_dim = 0; i_reduce_dim < reduce_indexer.ndim; ++i_reduce_dim) {
                in_indexer.index[out_indexer.ndim + i_reduce_dim] = reduce_indexer.index[i_reduce_dim];
            }

            char* in_ptr = cumo_na_iarray_at_dim(&in_iarray, &in_indexer);
            impl.Reduce(impl.MapIn(*reinterpret_cast<TypeIn*>(in_ptr), i_reduce), accum);
        }

        if (block_stride < 512) {
            sdata[tid] = accum;
            __syncthreads();
            if (block_stride <= 256) {
                _REDUCE(256);
                __syncthreads();
                if (block_stride <= 128) {
                    _REDUCE(128);
                    __syncthreads();
                    if (block_stride <= 64) {
                        _REDUCE(64);
                        __syncthreads();
                        if (block_stride <= 32) {
                            _REDUCE(32);
                            if (block_stride <= 16) {
                                _REDUCE(16);
                                if (block_stride <= 8) {
                                    _REDUCE(8);
                                    if (block_stride <= 4) {
                                        _REDUCE(4);
                                        if (block_stride <= 2) {
                                            _REDUCE(2);
                                            if (block_stride <= 1) {
                                                _REDUCE(1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            accum = sdata[tid];
            __syncthreads();
        }
        if (_J_offset == 0 && i_out < out_indexer.total_size) {
            char* out_ptr = cumo_na_iarray_at_dim(&out_iarray, &out_indexer);
            *reinterpret_cast<TypeOut*>(out_ptr) = impl.MapOut(accum);
        }
    }
}

#undef _REDUCE

static constexpr size_t max_block_size = 512;

template <typename TypeIn, typename TypeOut, typename TypeReduce, typename ReductionImpl>
void cumo_reduce(na_reduction_arg_t arg, ReductionImpl&& impl) {
    na_indexer_t& in_indexer = arg.in_indexer;
    na_indexer_t& out_indexer = arg.out_indexer;

    int64_t clp2_count = round_up_to_power_of_2(int64_t(in_indexer.total_size / out_indexer.total_size - 1));
    size_t block_stride = std::max(int64_t(1), int64_t(max_block_size) / clp2_count);
    
    size_t total_size = (out_indexer.total_size + block_stride - 1) / block_stride * max_block_size;
    size_t grid_size = std::min(0x7fffffffUL, (total_size + max_block_size - 1) / max_block_size);
    size_t block_size = std::min(max_block_size, total_size);

    size_t shared_mem_size = sizeof(TypeReduce) * block_size;
    reduction_kernel<TypeIn,TypeOut,TypeReduce,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>( block_stride, impl);
}

#endif // CUMO_REDUCE_KERNEL_H
