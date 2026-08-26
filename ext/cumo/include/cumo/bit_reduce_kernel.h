#ifndef CUMO_BIT_REDUCE_KERNEL_H
#define CUMO_BIT_REDUCE_KERNEL_H

#include <algorithm>
#include <cstdint>

#include "cumo/indexer.h"
#include "cumo/reduce_kernel.h"

namespace cumo_bit_detail {

// A Bit reduction addresses its operands the way a numeric one does -- the
// offsets are arithmetic on step[] and shape[] and never touch the pointer --
// so cumo_reduce_addr_t describes it too, with the input side read as bits.
//
// What has no numeric equivalent is the unit a thread folds. When the reduce
// axis runs bit by bit through memory, one thread takes a whole word of
// CUMO_NB elements at a time instead of one element, which is where a bit
// reduction gets to be thirty-two times cheaper than the loop it replaces.
static inline bool reduce_by_words(const cumo_detail::cumo_reduce_addr_t& ad) {
    return ad.in_reduce_flat && ad.in_reduce_step == 1;
}

static inline int64_t reduce_unit_count(const cumo_detail::cumo_reduce_addr_t& ad, int64_t reduce_total_size) {
    return reduce_by_words(ad)
        ? (reduce_total_size + (int64_t)CUMO_NB - 1) / (int64_t)CUMO_NB
        : reduce_total_size;
}

// A unit is CUMO_NB elements wide when the axis folds by words, so the chunk a
// split has to keep worth its while is that many times shorter.
static inline int64_t reduce_split_count(int64_t unit_total_size, int64_t out_block_num, bool words) {
    int64_t min_chunk = words ? cumo_detail::min_split_chunk / (int64_t)CUMO_NB : cumo_detail::min_split_chunk;
    if (out_block_num >= cumo_detail::min_grid_size) return 1;
    if (unit_total_size < min_chunk * 2) return 1;
    int64_t want = (cumo_detail::min_grid_size + out_block_num - 1) / out_block_num;
    int64_t fits = unit_total_size / min_chunk;
    int64_t n = std::min(std::min(want, fits), cumo_detail::max_split);
    return n < 2 ? 1 : n;
}

// The offset helpers take the arg by template so that the two reduction args
// -- one answering a count, one answering a Bit -- share them.
template <typename TArg>
__device__ static inline ssize_t bit_in_out_offset(const TArg& arg, const cumo_detail::cumo_reduce_addr_t& ad, int64_t i_out) {
    if (ad.in_out_flat) return i_out * ad.in_out_step;
    if (ad.split < 0) return 0;
    return cumo_detail::axes_offset(arg.in, arg.in_indexer, 0, ad.split, i_out);
}

template <typename TArg>
__device__ static inline ssize_t bit_in_offset(const TArg& arg, const cumo_detail::cumo_reduce_addr_t& ad, ssize_t in_out_off, int64_t i_reduce, int64_t i_in) {
    if (ad.in_reduce_flat) return in_out_off + i_reduce * ad.in_reduce_step;
    if (ad.split < 0) return cumo_detail::axes_offset(arg.in, arg.in_indexer, 0, arg.in_indexer.ndim, i_in);
    return in_out_off + cumo_detail::axes_offset(arg.in, arg.in_indexer, ad.split, arg.in_indexer.ndim, i_reduce);
}

template <typename TArg>
__device__ static inline ssize_t bit_out_offset(const TArg& arg, const cumo_detail::cumo_reduce_addr_t& ad, int64_t i_out) {
    if (ad.out_flat) return i_out * ad.out_step;
    return cumo_detail::axes_offset(arg.out, arg.out_indexer, 0, arg.out_indexer.ndim, i_out);
}

struct BitCountImpl {
    __device__ uint64_t Identity(int64_t /*index*/) { return 0; }
    __device__ uint64_t MapIn(uint64_t in, int64_t /*index*/) { return in; }
    __device__ void Reduce(uint64_t next, uint64_t& accum) { accum += next; }
    __device__ uint64_t MapOut(uint64_t accum) { return accum; }
};

// Counts one output element's slice [begin, end) of the reduce axis, taking
// every reduce_block_size-th unit from reduce_offset so that the threads
// sharing an output split the axis between them. invert answers the zeros,
// which is what count_false is.
template <typename TArg>
__device__ static inline uint64_t bit_count_axis(
        const TArg& arg, const cumo_detail::cumo_reduce_addr_t& ad, bool words, int invert,
        ssize_t in_out_off, int64_t i_in, int64_t reduce_total_size,
        int64_t begin, int64_t end, int64_t reduce_offset, int64_t reduce_block_size) {
    const ssize_t nb = (ssize_t)CUMO_NB;
    uint64_t accum = 0;

    if (words) {
        ssize_t p = (ssize_t)arg.in.pos + in_out_off;
        const CUMO_BIT_DIGIT* a = arg.in.ptr + (size_t)(p / nb);
        ssize_t o = p % nb;
        uint64_t nw = (uint64_t)((o + reduce_total_size + nb - 1) / nb);

        for (int64_t c = begin + reduce_offset; c < end; c += reduce_block_size) {
            int64_t base = c * (int64_t)CUMO_NB;
            uint64_t kend = (reduce_total_size - base < (int64_t)CUMO_NB) ? (uint64_t)(reduce_total_size - base) : CUMO_NB;
            CUMO_BIT_DIGIT z = cumo_bit_gather_word(a, o, nw, (uint64_t)c);
            if (invert) z = ~z;
            accum += (uint64_t)__popc(z & CUMO_SLB(kend));
        }
    } else {
        for (int64_t i_reduce = begin + reduce_offset; i_reduce < end; i_reduce += reduce_block_size, i_in += reduce_block_size) {
            size_t pos = (size_t)((ssize_t)arg.in.pos + bit_in_offset(arg, ad, in_out_off, i_reduce, i_in));
            CUMO_BIT_DIGIT x;
            CUMO_LOAD_BIT(arg.in.ptr, pos, x);
            accum += (uint64_t)(invert ? (x == 0) : (x != 0));
        }
    }
    return accum;
}

__global__ static void bit_count_reduction_kernel(
        cumo_na_bit_reduction_arg_t arg, cumo_detail::cumo_reduce_addr_t ad, bool words, int invert,
        int out_block_size, int reduce_block_size, int64_t unit_total_size) {
    extern __shared__ __align__(8) char sdata_raw[];
    uint64_t* sdata = reinterpret_cast<uint64_t*>(sdata_raw);
    unsigned int tid = threadIdx.x;
    BitCountImpl impl;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset = tid / out_block_size;
    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off = bit_in_out_offset(arg, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        uint64_t accum = bit_count_axis(arg, ad, words, invert, in_out_off, i_in, reduce_total_size,
                                        0, unit_total_size, reduce_offset, reduce_block_size);

        accum = cumo_detail::reduce_in_block(accum, sdata, tid, out_block_size, impl);
        if (reduce_offset == 0) {
            *reinterpret_cast<uint64_t*>(arg.out.ptr + bit_out_offset(arg, ad, i_out)) = accum;
        }
    }
}

// First pass of a split count, for the shapes that would otherwise leave the
// grid a handful of blocks however long the reduce axis is.
template <typename TArg>
__global__ static void bit_count_partial_kernel(
        TArg arg, cumo_detail::cumo_reduce_addr_t ad, bool words, int invert,
        uint64_t* partial, int64_t n_split, int64_t chunk,
        int out_block_size, int reduce_block_size, int64_t unit_total_size) {
    extern __shared__ __align__(8) char sdata_raw[];
    uint64_t* sdata = reinterpret_cast<uint64_t*>(sdata_raw);
    unsigned int tid = threadIdx.x;
    BitCountImpl impl;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;
    int64_t partial_total_size = out_total_size * n_split;

    int64_t reduce_offset = tid / out_block_size;
    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i = out_base + out_offset; i < partial_total_size; i += out_stride) {
        int64_t i_out = i % out_total_size;
        int64_t i_split = i / out_total_size;
        int64_t begin = i_split * chunk;
        int64_t end = begin + chunk;
        if (end > unit_total_size) end = unit_total_size;
        ssize_t in_out_off = bit_in_out_offset(arg, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + begin + reduce_offset;

        uint64_t accum = bit_count_axis(arg, ad, words, invert, in_out_off, i_in, reduce_total_size,
                                        begin, end, reduce_offset, reduce_block_size);

        accum = cumo_detail::reduce_in_block(accum, sdata, tid, out_block_size, impl);
        if (reduce_offset == 0) {
            partial[i_out * n_split + i_split] = accum;
        }
    }
}


// How one launch is laid out over the outputs and the reduce axis, plus the
// second pass the shapes with too few outputs need. Worked out on the host so
// that both reductions below share it.
struct bit_reduce_plan {
    cumo_detail::cumo_reduce_addr_t ad;
    bool words;
    int64_t out_total_size;
    int64_t reduce_total_size;
    int64_t unit_total_size;
    int64_t out_block_size;
    int64_t reduce_block_size;
    int64_t out_block_num;
    int64_t n_split;
    int64_t chunk;
    int64_t split_out_block_size;
    int64_t split_reduce_block_size;
    int64_t partial_block_num;
};

template <typename TArg>
static inline bit_reduce_plan make_bit_reduce_plan(const TArg& arg) {
    bit_reduce_plan p;

    p.out_total_size = arg.out_indexer.total_size;
    p.reduce_total_size = arg.in_indexer.total_size / p.out_total_size;
    p.ad = cumo_detail::make_reduce_addr(arg, p.reduce_total_size);
    p.words = reduce_by_words(p.ad);
    p.unit_total_size = reduce_unit_count(p.ad, p.reduce_total_size);

    cumo_detail::reduce_block_split(p.ad, p.unit_total_size, &p.out_block_size, &p.reduce_block_size);
    p.out_block_num = (p.out_total_size + p.out_block_size - 1) / p.out_block_size;

    p.n_split = reduce_split_count(p.unit_total_size, p.out_block_num, p.words);
    p.chunk = 0;
    p.split_out_block_size = 0;
    p.split_reduce_block_size = 0;
    p.partial_block_num = 0;
    if (p.n_split > 1) {
        p.chunk = (p.unit_total_size + p.n_split - 1) / p.n_split;
        cumo_detail::reduce_block_split(p.ad, p.chunk, &p.split_out_block_size, &p.split_reduce_block_size);
        p.partial_block_num = (p.out_total_size * p.n_split + p.split_out_block_size - 1) / p.split_out_block_size;
        // Splitting buys nothing when the narrower block it leaves takes the
        // block count back down to where it started.
        if (p.partial_block_num <= p.out_block_num) p.n_split = 1;
    }
    return p;
}

// all? and any? are the count of set bits against the length of the axis and
// against zero, so they reduce the same way and differ only here.
__device__ static inline CUMO_BIT_DIGIT bit_pred_of_count(uint64_t count, int64_t reduce_total_size, int all) {
    return (CUMO_BIT_DIGIT)(all ? (count == (uint64_t)reduce_total_size) : (count != 0));
}

__global__ static void bit_pred_reduction_kernel(
        cumo_na_bit_pred_reduction_arg_t arg, cumo_detail::cumo_reduce_addr_t ad, bool words, int all,
        int out_block_size, int reduce_block_size, int64_t unit_total_size) {
    extern __shared__ __align__(8) char sdata_raw[];
    uint64_t* sdata = reinterpret_cast<uint64_t*>(sdata_raw);
    unsigned int tid = threadIdx.x;
    BitCountImpl impl;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset = tid / out_block_size;
    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off = bit_in_out_offset(arg, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        uint64_t accum = bit_count_axis(arg, ad, words, 0, in_out_off, i_in, reduce_total_size,
                                        0, unit_total_size, reduce_offset, reduce_block_size);

        accum = cumo_detail::reduce_in_block(accum, sdata, tid, out_block_size, impl);
        if (reduce_offset == 0) {
            size_t pos = (size_t)((ssize_t)arg.out.pos + bit_out_offset(arg, ad, i_out));
            CUMO_STORE_BIT(arg.out.ptr, pos, bit_pred_of_count(accum, reduce_total_size, all));
        }
    }
}

// Second pass of a split all? or any?. There are few outputs whenever the axis
// was worth splitting, so one thread per output walks its chunks.
__global__ static void bit_pred_combine_kernel(
        cumo_na_bit_pred_reduction_arg_t arg, cumo_detail::cumo_reduce_addr_t ad, int all,
        const uint64_t* partial, int64_t n_split) {
    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    for (int64_t i_out = blockIdx.x * blockDim.x + threadIdx.x; i_out < out_total_size;
         i_out += (int64_t)blockDim.x * gridDim.x) {
        uint64_t count = 0;
        for (int64_t i_split = 0; i_split < n_split; ++i_split) {
            count += partial[i_out * n_split + i_split];
        }
        size_t pos = (size_t)((ssize_t)arg.out.pos + bit_out_offset(arg, ad, i_out));
        CUMO_STORE_BIT(arg.out.ptr, pos, bit_pred_of_count(count, reduce_total_size, all));
    }
}

}  // cumo_bit_detail

// Counts the set (or, with invert, the clear) bits of every output element in
// one launch. Replaces a loop that ran the whole reduction once per output
// element, which cost a kernel launch per row.
static inline void cumo_bit_count_reduce(cumo_na_bit_reduction_arg_t arg, int invert) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    cumo_bit_detail::bit_reduce_plan p = cumo_bit_detail::make_bit_reduce_plan(arg);
    int64_t block_size = cumo_detail::max_block_size;
    int64_t shared_mem_size = sizeof(uint64_t) * block_size;

    if (p.n_split < 2) {
        int64_t grid_size = std::min(cumo_detail::max_grid_size, p.out_block_num);
        cumo_bit_detail::bit_count_reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
            arg, p.ad, p.words, invert, (int)p.out_block_size, (int)p.reduce_block_size, p.unit_total_size);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }

    int64_t partial_total_size = p.out_total_size * p.n_split;
    uint64_t* partial = reinterpret_cast<uint64_t*>(cumo_cuda_runtime_malloc(sizeof(uint64_t) * partial_total_size));

    int64_t grid_size = std::min(cumo_detail::max_grid_size, p.partial_block_num);
    cumo_bit_detail::bit_count_partial_kernel<<<grid_size, block_size, shared_mem_size>>>(
        arg, p.ad, p.words, invert, partial, p.n_split, p.chunk,
        (int)p.split_out_block_size, (int)p.split_reduce_block_size, p.unit_total_size);
    cumo_cuda_runtime_check_kernel_launch();

    cumo_na_reduction_arg_t combine;
    combine.in.ptr = reinterpret_cast<char*>(partial);
    combine.in.step[0] = sizeof(uint64_t);
    combine.in_indexer.ndim = 1;
    combine.in_indexer.shape[0] = partial_total_size;
    combine.in_indexer.total_size = partial_total_size;
    combine.out = arg.out;
    combine.out_indexer = arg.out_indexer;
    cumo_reduce<uint64_t, uint64_t>(combine, cumo_bit_detail::BitCountImpl{});

    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// The same for all? (all non-zero) and any? (any non-zero), whose result is a
// Bit rather than a count.
static inline void cumo_bit_pred_reduce(cumo_na_bit_pred_reduction_arg_t arg, int all) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    cumo_bit_detail::bit_reduce_plan p = cumo_bit_detail::make_bit_reduce_plan(arg);
    int64_t block_size = cumo_detail::max_block_size;
    int64_t shared_mem_size = sizeof(uint64_t) * block_size;

    if (p.n_split < 2) {
        int64_t grid_size = std::min(cumo_detail::max_grid_size, p.out_block_num);
        cumo_bit_detail::bit_pred_reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
            arg, p.ad, p.words, all, (int)p.out_block_size, (int)p.reduce_block_size, p.unit_total_size);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }

    int64_t partial_total_size = p.out_total_size * p.n_split;
    uint64_t* partial = reinterpret_cast<uint64_t*>(cumo_cuda_runtime_malloc(sizeof(uint64_t) * partial_total_size));

    int64_t grid_size = std::min(cumo_detail::max_grid_size, p.partial_block_num);
    cumo_bit_detail::bit_count_partial_kernel<<<grid_size, block_size, shared_mem_size>>>(
        arg, p.ad, p.words, 0, partial, p.n_split, p.chunk,
        (int)p.split_out_block_size, (int)p.split_reduce_block_size, p.unit_total_size);
    cumo_cuda_runtime_check_kernel_launch();

    int64_t combine_grid = (p.out_total_size + block_size - 1) / block_size;
    if (combine_grid > cumo_detail::max_grid_size) combine_grid = cumo_detail::max_grid_size;
    cumo_bit_detail::bit_pred_combine_kernel<<<combine_grid, block_size>>>(arg, p.ad, all, partial, p.n_split);
    cumo_cuda_runtime_check_kernel_launch();

    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

#endif // CUMO_BIT_REDUCE_KERNEL_H
