#ifndef CUMO_REDUCE_KERNEL_H
#define CUMO_REDUCE_KERNEL_H

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "cumo/indexer.h"

// A kernel parameter that is read through a runtime index, the way the steps
// of an operand are, is copied to local memory by the compiler unless it is
// declared __grid_constant__, and the copy is paid by every thread: the second
// operand of a zip reduction cost a 104-byte stack frame that put mulsum along
// a middle axis at three times the sum of the same bytes. The qualifier needs
// CUDA 11.7 and sm_70; below those the parameter is merely const.
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) || (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 7)))
#define CUMO_GRID_CONSTANT const
#else
#define CUMO_GRID_CONSTANT const __grid_constant__
#endif

namespace cumo_detail {

static constexpr int64_t max_block_size = 512;
static constexpr int64_t max_grid_size = 0x7fffffff;
static constexpr int64_t warp_size = 32;

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

static inline ssize_t step_magnitude(ssize_t step) { return step < 0 ? -step : step; }

// How a reduction addresses its operands, worked out on the host.
//
// A group of axes is flat when the offset of its i-th element is i * step,
// which is what lets a loop walk it with one multiply. Getting the offset from
// cumo_na_indexer_set_dim and cumo_na_iarray_at_dim instead costs far more than
// the reduction does: those write into the indexer, and the indexer carries
// shape[] and index[] for CUMO_NA_MAX_DIMENSION, so the copy the kernel writes
// into cannot stay in registers and every element pays a local memory round
// trip. Measured on an RTX 5070 Ti, a 2048x2048 sum along the last axis takes
// 1.89 ms that way and 0.02 ms with the offsets below.
//
// split is the first reduce dimension. It is -1 when the reduce axis does not
// end up on a dimension boundary, which leaves the whole flat index to
// decompose; the flat forms never need it.
typedef struct {
    int split;
    bool in_out_flat;
    bool in_reduce_flat;
    bool out_flat;
    bool out2_flat;
    bool out_inner;         // the out axis, not the reduce axis, runs along memory
    ssize_t in_out_step;    // bytes, or bits for a Bit input
    ssize_t in_reduce_step; // bytes, or bits for a Bit input
    ssize_t out_step;       // bytes
    ssize_t out2_step;      // bytes
} cumo_reduce_addr_t;

// True when the offset of the i-th element over dims [begin, end) is i * step.
// Templated on the iarray so that a Bit reduction, whose steps are bits rather
// than bytes, gets the same arithmetic.
template <typename TIarray>
static inline bool axes_are_flat(const TIarray& iarray, const cumo_na_indexer_t& indexer, int begin, int end, ssize_t* step) {
    if (begin >= end) {
        *step = 0;
        return true;
    }
    ssize_t acc = iarray.step[end - 1];
    for (int j = end - 1; j > begin; --j) {
        acc *= static_cast<ssize_t>(indexer.shape[j]);
        if (iarray.step[j - 1] != acc) return false;
    }
    *step = iarray.step[end - 1];
    return true;
}

template <typename TArg>
static inline cumo_reduce_addr_t make_reduce_addr(const TArg& arg, int64_t reduce_total_size) {
    cumo_reduce_addr_t ad;
    int in_ndim = arg.in_indexer.ndim;
    ssize_t whole_step;

    if (axes_are_flat(arg.in, arg.in_indexer, 0, in_ndim, &whole_step)) {
        // One stride covers the input, so the reduce axis does not have to fall
        // on a dimension boundary. This is also how the scratch of a split
        // reduction comes back in.
        ad.split = -1;
        ad.in_out_flat = true;
        ad.in_reduce_flat = true;
        ad.in_reduce_step = whole_step;
        ad.in_out_step = whole_step * reduce_total_size;
    } else {
        int split = in_ndim;
        int64_t acc = 1;
        while (split > 0 && acc < reduce_total_size) {
            --split;
            acc *= static_cast<int64_t>(arg.in_indexer.shape[split]);
        }
        if (acc == reduce_total_size) {
            ad.split = split;
            ad.in_reduce_flat = axes_are_flat(arg.in, arg.in_indexer, split, in_ndim, &ad.in_reduce_step);
            ad.in_out_flat = axes_are_flat(arg.in, arg.in_indexer, 0, split, &ad.in_out_step);
        } else {
            ad.split = -1;
            ad.in_reduce_flat = false;
            ad.in_out_flat = false;
            ad.in_reduce_step = 0;
            ad.in_out_step = 0;
        }
    }

    ad.out_flat = axes_are_flat(arg.out, arg.out_indexer, 0, arg.out_indexer.ndim, &ad.out_step);
    ad.out2_flat = true;
    ad.out2_step = 0;

    ssize_t out_inner_step, reduce_inner_step;
    if (ad.in_out_flat && ad.in_reduce_flat) {
        out_inner_step = ad.in_out_step;
        reduce_inner_step = ad.in_reduce_step;
    } else {
        out_inner_step = ad.split > 0 ? arg.in.step[ad.split - 1] : 0;
        reduce_inner_step = (ad.split >= 0 && ad.split < in_ndim) ? arg.in.step[in_ndim - 1] : 0;
    }
    ad.out_inner = out_inner_step != 0 &&
        (reduce_inner_step == 0 || step_magnitude(out_inner_step) < step_magnitude(reduce_inner_step));

    return ad;
}

static inline void set_reduce_addr_out2(cumo_reduce_addr_t* ad, const cumo_na_reduction_arg_t& arg, const cumo_na_iarray_t& out2) {
    ad->out2_flat = axes_are_flat(out2, arg.out_indexer, 0, arg.out_indexer.ndim, &ad->out2_step);
}

// How a block's threads are shared between outputs. Whichever of the two axes
// runs along memory decides whether a block's reads coalesce, so the threads
// that sit next to each other in the block walk that axis: consecutive in
// i_out when the out axis is the contiguous one (a 2048x2048 sum along axis 0
// otherwise puts each of a block's 512 threads on a different row and reads
// one element per transaction), and consecutive in i_reduce when the reduce
// axis is. In that second, reduce-major form the threads of one output are a
// contiguous group of reduce_block_size, and the group is sized so that every
// thread has min_reduce_per_thread elements to read rather than one: a
// 4096x256 sum along the last axis with a 256-thread group per row reads one
// element per thread and then spends the time in the tree, at 92 GB/s, where
// a group of 16 reading 16 each gets 520.
static constexpr int64_t min_reduce_per_thread = 16;

static inline void reduce_block_split(const cumo_reduce_addr_t& ad, int64_t reduce_total_size, int64_t* out_block_size, int64_t* reduce_block_size) {
    int64_t n = std::max(int64_t{1}, reduce_total_size);
    int64_t rbs;
    if (ad.out_inner) {
        rbs = std::min(max_block_size / warp_size, round_up_to_power_of_2((n + min_reduce_per_thread - 1) / min_reduce_per_thread));
    } else {
        rbs = std::min(max_block_size, round_up_to_power_of_2((n + min_reduce_per_thread - 1) / min_reduce_per_thread));
    }
    *reduce_block_size = rbs;
    *out_block_size = max_block_size / rbs;
}

// Offset of the i-th element over dims [begin, end), in whatever unit the
// iarray's steps carry. Everything here is
// read out of the kernel parameter and nothing is written back, which is what
// keeps the indexer out of local memory.
//
// i is always below the product of the dims, so the outermost one takes what
// is left without a division.
template <typename TIarray>
__device__ static __forceinline__ ssize_t axes_offset(const TIarray& iarray, const cumo_na_indexer_t& indexer, int begin, int end, int64_t i) {
    if (end <= begin) return 0;
    ssize_t off = 0;
    for (int j = end; --j > begin;) {
        int64_t n = static_cast<int64_t>(indexer.shape[j]);
        off += static_cast<ssize_t>(i % n) * iarray.step[j];
        i /= n;
    }
    return off + static_cast<ssize_t>(i) * iarray.step[begin];
}

// The same split of i handed to two operands that share the indexer, which is
// what the two inputs of a zip reduction are.
template <typename TIarray>
__device__ static __forceinline__ void axes_offset_pair(const TIarray& a, const TIarray& b, const cumo_na_indexer_t& indexer, int begin, int end, int64_t i, ssize_t* off_a, ssize_t* off_b) {
    *off_a = 0;
    *off_b = 0;
    if (end <= begin) return;
    for (int j = end; --j > begin;) {
        int64_t n = static_cast<int64_t>(indexer.shape[j]);
        ssize_t r = static_cast<ssize_t>(i % n);
        *off_a += r * a.step[j];
        *off_b += r * b.step[j];
        i /= n;
    }
    *off_a += static_cast<ssize_t>(i) * a.step[begin];
    *off_b += static_cast<ssize_t>(i) * b.step[begin];
}

// The input side takes the iarray and the indexer rather than the whole
// reduction arg, so that mulsum can address its second operand with the same
// indexer and a step set of its own.
template <bool FLAT>
__device__ static __forceinline__ ssize_t reduce_in_out_offset(const cumo_na_iarray_t& in, const cumo_na_indexer_t& in_indexer, const cumo_reduce_addr_t& ad, int64_t i_out) {
    if (FLAT || ad.in_out_flat) return i_out * ad.in_out_step;
    if (ad.split < 0) return 0;
    return axes_offset(in, in_indexer, 0, ad.split, i_out);
}

// reduce_in_out_offset for the two operands of a zip reduction at once.
template <bool FLAT>
__device__ static __forceinline__ void reduce_in_out_offset_pair(const cumo_na_iarray_t& in, const cumo_na_iarray_t& in2, const cumo_na_indexer_t& in_indexer, const cumo_reduce_addr_t& ad, const cumo_reduce_addr_t& ad2, int64_t i_out, ssize_t* off, ssize_t* off2) {
    if (!FLAT && !ad.in_out_flat && !ad2.in_out_flat && ad.split >= 0 && ad.split == ad2.split) {
        axes_offset_pair(in, in2, in_indexer, 0, ad.split, i_out, off, off2);
        return;
    }
    *off = reduce_in_out_offset<FLAT>(in, in_indexer, ad, i_out);
    *off2 = reduce_in_out_offset<FLAT>(in2, in_indexer, ad2, i_out);
}

template <bool FLAT>
__device__ static __forceinline__ ssize_t reduce_in_offset(const cumo_na_iarray_t& in, const cumo_na_indexer_t& in_indexer, const cumo_reduce_addr_t& ad, ssize_t in_out_off, int64_t i_reduce, int64_t i_in) {
    if (FLAT || ad.in_reduce_flat) return in_out_off + i_reduce * ad.in_reduce_step;
    if (ad.split < 0) return axes_offset(in, in_indexer, 0, in_indexer.ndim, i_in);
    return in_out_off + axes_offset(in, in_indexer, ad.split, in_indexer.ndim, i_reduce);
}

template <bool FLAT>
__device__ static __forceinline__ ssize_t reduce_out_offset(const cumo_na_reduction_arg_t& arg, const cumo_reduce_addr_t& ad, int64_t i_out) {
    if (FLAT || ad.out_flat) return i_out * ad.out_step;
    return axes_offset(arg.out, arg.out_indexer, 0, arg.out_indexer.ndim, i_out);
}

template <bool FLAT>
__device__ static __forceinline__ ssize_t reduce_out2_offset(const cumo_na_reduction_arg_t& arg, const cumo_na_iarray_t& out2, const cumo_reduce_addr_t& ad, int64_t i_out) {
    if (FLAT || ad.out2_flat) return i_out * ad.out2_step;
    return axes_offset(out2, arg.out_indexer, 0, arg.out_indexer.ndim, i_out);
}

// Which output and which slice of its reduce axis a thread takes, in the
// layout reduce_block_split describes.
__device__ static __forceinline__ void reduce_thread_split(const cumo_reduce_addr_t& ad, unsigned int tid, int out_block_size, int reduce_block_size, int64_t* reduce_offset, int64_t* out_offset) {
    if (!ad.out_inner) {
        *reduce_offset = tid % reduce_block_size;
        *out_offset = tid / reduce_block_size;
    } else {
        *reduce_offset = tid / out_block_size;
        *out_offset = tid % out_block_size;
    }
}

// Combines the accumulators a block holds for one output element. In the
// out-major layout the threads that share out_offset sit out_block_size apart,
// so the tree only has to run down to out_block_size. In the reduce-major
// layout they are a contiguous, warp-aligned group of reduce_block_size, so a
// step whose partners are within a warp needs only __syncwarp.
template <typename TypeReduce, typename ReductionImpl>
__device__ static __forceinline__ TypeReduce reduce_in_block(TypeReduce accum, TypeReduce* sdata, unsigned int tid, int out_block_size, int reduce_block_size, bool reduce_major, ReductionImpl& impl) {
    if (reduce_block_size == 1) return accum;
    if (reduce_major) {
        int r = tid % reduce_block_size;
        sdata[tid] = accum;
        for (int stride = reduce_block_size / 2; stride > 0; stride >>= 1) {
            if (stride >= warp_size) __syncthreads(); else __syncwarp();
            if (r < stride) {
                impl.Reduce(sdata[tid + stride], sdata[tid]);
            }
        }
        accum = sdata[tid];
        if (reduce_block_size > warp_size) __syncthreads(); else __syncwarp();
        return accum;
    }
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
    return accum;
}

// Reduces one output element's slice of the reduce axis, taking every
// reduce_block_size-th element from begin so that the threads sharing an
// output split the axis between them. A flat reduce group walks by pointer:
// recomputing i_reduce * step every element costs a 64-bit multiply, and the
// contiguous shapes are already at memory bandwidth without it.
//
// ARG_INDEX picks what MapIn is handed. arg(min|max) wants the index along the
// reduction axis, which is what CuPy's reductions pass. Everything else wants
// the offset of the element from the start of the input, which is the index
// (min|max)_index answers with.
template <bool FLAT, bool ARG_INDEX, typename TypeIn, typename ReductionImpl>
__device__ static __forceinline__ auto reduce_axis(const cumo_na_iarray_t& in, const cumo_na_indexer_t& in_indexer, const cumo_reduce_addr_t& ad, ReductionImpl& impl,
        ssize_t in_out_off, int64_t i_in, int64_t begin, int64_t end, int64_t reduce_offset, int64_t reduce_block_size) -> decltype(impl.Identity(0)) {
    using TypeReduce = decltype(impl.Identity(0));

    char* in_base = in.ptr;
    TypeIn* in_head = reinterpret_cast<TypeIn*>(in_base);
    int64_t i_reduce = begin + reduce_offset;
    char* p = in_base + reduce_in_offset<FLAT>(in, in_indexer, ad, in_out_off, i_reduce, i_in);
    ssize_t advance = ad.in_reduce_step * reduce_block_size;

    TypeReduce accum = ARG_INDEX ? impl.Identity(reduce_offset)
                                 : impl.Identity(static_cast<int64_t>(reinterpret_cast<TypeIn*>(p) - in_head));

    for (; i_reduce < end; i_reduce += reduce_block_size, i_in += reduce_block_size) {
        TypeIn* in_ptr = reinterpret_cast<TypeIn*>(p);
        impl.Reduce(impl.MapIn(*in_ptr, ARG_INDEX ? i_reduce : static_cast<int64_t>(in_ptr - in_head)), accum);
        p = (FLAT || ad.in_reduce_flat)
            ? p + advance
            : in_base + reduce_in_offset<FLAT>(in, in_indexer, ad, in_out_off, i_reduce + reduce_block_size, i_in + reduce_block_size);
    }
    return accum;
}

// Reduces one output element's slice of the reduce axis over two inputs at
// once, which is what mulsum wants: the product it accumulates never exists as
// an array. The pair comes out of one broadcast, so arg.in_indexer addresses
// both and only the steps differ, which is what in2 and ad2 carry.
template <bool FLAT, typename TypeIn, typename ReductionImpl>
__device__ static __forceinline__ auto reduce_axis_zip(const cumo_na_reduction_arg_t& arg, const cumo_na_iarray_t& in2,
        const cumo_reduce_addr_t& ad, const cumo_reduce_addr_t& ad2, ReductionImpl& impl,
        ssize_t in_out_off, ssize_t in_out_off2, int64_t i_in, int64_t begin, int64_t end,
        int64_t reduce_offset, int64_t reduce_block_size) -> decltype(impl.Identity(0)) {
    using TypeReduce = decltype(impl.Identity(0));

    int64_t i_reduce = begin + reduce_offset;
    char* p = arg.in.ptr + reduce_in_offset<FLAT>(arg.in, arg.in_indexer, ad, in_out_off, i_reduce, i_in);
    char* q = in2.ptr + reduce_in_offset<FLAT>(in2, arg.in_indexer, ad2, in_out_off2, i_reduce, i_in);
    ssize_t advance = ad.in_reduce_step * reduce_block_size;
    ssize_t advance2 = ad2.in_reduce_step * reduce_block_size;

    TypeReduce accum = impl.Identity(0);

    for (; i_reduce < end; i_reduce += reduce_block_size, i_in += reduce_block_size) {
        impl.Reduce(impl.MapIn(*reinterpret_cast<TypeIn*>(p), *reinterpret_cast<TypeIn*>(q), i_reduce), accum);
        p = (FLAT || ad.in_reduce_flat)
            ? p + advance
            : arg.in.ptr + reduce_in_offset<FLAT>(arg.in, arg.in_indexer, ad, in_out_off, i_reduce + reduce_block_size, i_in + reduce_block_size);
        q = (FLAT || ad2.in_reduce_flat)
            ? q + advance2
            : in2.ptr + reduce_in_offset<FLAT>(in2, arg.in_indexer, ad2, in_out_off2, i_reduce + reduce_block_size, i_in + reduce_block_size);
    }
    return accum;
}

// Reference: cupy reduction kernel
// Note that reduction and out axis are inverse with cupy. Former axes are out axes, latters are reduce axes.

template <bool FLAT, typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off = reduce_in_out_offset<FLAT>(arg.in, arg.in_indexer, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        TypeReduce accum = reduce_axis<FLAT,false,TypeIn>(arg.in, arg.in_indexer, ad, impl, in_out_off, i_in, 0, reduce_total_size, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(arg.out.ptr + reduce_out_offset<FLAT>(arg, ad, i_out));
            *out_ptr = impl.MapOut(accum);
        }
    }
}

// Variant of reduction_kernel reading two inputs, for mulsum. See
// reduce_axis_zip above.
template <bool FLAT, typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_zip_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_na_iarray_t in2, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad2, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off, in_out_off2;
        reduce_in_out_offset_pair<FLAT>(arg.in, in2, arg.in_indexer, ad, ad2, i_out, &in_out_off, &in_out_off2);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        TypeReduce accum = reduce_axis_zip<FLAT,TypeIn>(arg, in2, ad, ad2, impl, in_out_off, in_out_off2, i_in, 0, reduce_total_size, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(arg.out.ptr + reduce_out_offset<FLAT>(arg, ad, i_out));
            *out_ptr = impl.MapOut(accum);
        }
    }
}

// Variant of reduction_kernel for arg-reductions (argmax/argmin), which report
// the index along the reduction axis rather than the index of an element.
template <bool FLAT, typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_arg_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off = reduce_in_out_offset<FLAT>(arg.in, arg.in_indexer, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        TypeReduce accum = reduce_axis<FLAT,true,TypeIn>(arg.in, arg.in_indexer, ad, impl, in_out_off, i_in, 0, reduce_total_size, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(arg.out.ptr + reduce_out_offset<FLAT>(arg, ad, i_out));
            *out_ptr = impl.MapOut(accum);
        }
    }
}

// Variant of reduction_kernel writing two results per output element, so that
// minmax reads the input once instead of reducing it twice. The impl stores
// through the pointers rather than returning, since there are two of them.
template <bool FLAT, typename TypeIn, typename TypeOut, typename ReductionImpl>
__global__ static void reduction_pair_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_na_iarray_t out2_iarray, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using TypeReduce = decltype(impl.Identity(0));

    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i_out = out_base + out_offset; i_out < out_total_size; i_out += out_stride) {
        ssize_t in_out_off = reduce_in_out_offset<FLAT>(arg.in, arg.in_indexer, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + reduce_offset;

        TypeReduce accum = reduce_axis<FLAT,false,TypeIn>(arg.in, arg.in_indexer, ad, impl, in_out_off, i_in, 0, reduce_total_size, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            TypeOut* out_ptr = reinterpret_cast<TypeOut*>(arg.out.ptr + reduce_out_offset<FLAT>(arg, ad, i_out));
            TypeOut* out2_ptr = reinterpret_cast<TypeOut*>(out2_iarray.ptr + reduce_out2_offset<FLAT>(arg, out2_iarray, ad, i_out));
            impl.MapOut(accum, out_ptr, out2_ptr);
        }
    }
}

// First pass of a split reduction: each block takes one chunk of one output's
// reduce axis and writes its accumulator to partial[i_out * n_split + i_split].
// MapOut is deliberately not applied — the combine pass needs the accumulator.
//
// The scratch is laid out by output so the combine pass can read it flat, but
// threads are handed out by output first, so a block still covers consecutive
// outputs when those are the ones running along memory.
template <bool FLAT, bool ARG, typename TypeIn, typename TypeReduce, typename ReductionImpl>
__global__ static void reduction_partial_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, TypeReduce* partial, int64_t n_split, int64_t chunk, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;
    int64_t partial_total_size = out_total_size * n_split;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i = out_base + out_offset; i < partial_total_size; i += out_stride) {
        int64_t i_out = i % out_total_size;
        int64_t i_split = i / out_total_size;
        int64_t begin = i_split * chunk;
        int64_t end = begin + chunk;
        if (end > reduce_total_size) end = reduce_total_size;
        ssize_t in_out_off = reduce_in_out_offset<FLAT>(arg.in, arg.in_indexer, ad, i_out);
        int64_t i_in = i_out * reduce_total_size + begin + reduce_offset;

        TypeReduce accum = reduce_axis<FLAT,ARG,TypeIn>(arg.in, arg.in_indexer, ad, impl, in_out_off, i_in, begin, end, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            partial[i_out * n_split + i_split] = accum;
        }
    }
}

// First pass of a split zip reduction. See reduction_partial_kernel above.
template <bool FLAT, typename TypeIn, typename TypeReduce, typename ReductionImpl>
__global__ static void reduction_zip_partial_kernel(CUMO_GRID_CONSTANT cumo_na_reduction_arg_t arg, CUMO_GRID_CONSTANT cumo_na_iarray_t in2, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad, CUMO_GRID_CONSTANT cumo_reduce_addr_t ad2, TypeReduce* partial, int64_t n_split, int64_t chunk, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    extern __shared__ __align__(8) char sdata_raw[];
    TypeReduce* sdata = reinterpret_cast<TypeReduce*>(sdata_raw);
    unsigned int tid = threadIdx.x;

    int64_t out_total_size = arg.out_indexer.total_size;
    int64_t reduce_total_size = arg.in_indexer.total_size / out_total_size;
    int64_t partial_total_size = out_total_size * n_split;

    int64_t reduce_offset, out_offset;
    reduce_thread_split(ad, tid, out_block_size, reduce_block_size, &reduce_offset, &out_offset);
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    for (int64_t i = out_base + out_offset; i < partial_total_size; i += out_stride) {
        int64_t i_out = i % out_total_size;
        int64_t i_split = i / out_total_size;
        int64_t begin = i_split * chunk;
        int64_t end = begin + chunk;
        if (end > reduce_total_size) end = reduce_total_size;
        ssize_t in_out_off, in_out_off2;
        reduce_in_out_offset_pair<FLAT>(arg.in, in2, arg.in_indexer, ad, ad2, i_out, &in_out_off, &in_out_off2);
        int64_t i_in = i_out * reduce_total_size + begin + reduce_offset;

        TypeReduce accum = reduce_axis_zip<FLAT,TypeIn>(arg, in2, ad, ad2, impl, in_out_off, in_out_off2, i_in, begin, end, reduce_offset, reduce_block_size);

        accum = reduce_in_block(accum, sdata, tid, out_block_size, reduce_block_size, !ad.out_inner, impl);
        if (reduce_offset == 0) {
            partial[i_out * n_split + i_split] = accum;
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
template <typename TypeIn, typename TypeReduce, typename ReductionImpl, bool ARG = false>
TypeReduce* reduce_partial_pass(cumo_na_reduction_arg_t arg, cumo_reduce_addr_t ad, int64_t n_split, int64_t reduce_total_size, cumo_na_reduction_arg_t* arg2, ReductionImpl& impl) {
    int64_t chunk = (reduce_total_size + n_split - 1) / n_split;
    int64_t partial_total_size = arg.out_indexer.total_size * n_split;
    TypeReduce* partial = reinterpret_cast<TypeReduce*>(cumo_cuda_runtime_malloc(sizeof(TypeReduce) * partial_total_size));

    int64_t out_block_size, reduce_block_size;
    reduce_block_split(ad, chunk, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (partial_total_size + out_block_size - 1) / out_block_size;
    int64_t grid_size = std::min(max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(TypeReduce) * max_block_size;

    if (ad.in_out_flat && ad.in_reduce_flat) {
        reduction_partial_kernel<true,ARG,TypeIn,TypeReduce,ReductionImpl><<<grid_size, max_block_size, shared_mem_size>>>(arg, ad, partial, n_split, chunk, out_block_size, reduce_block_size, impl);
    } else {
        reduction_partial_kernel<false,ARG,TypeIn,TypeReduce,ReductionImpl><<<grid_size, max_block_size, shared_mem_size>>>(arg, ad, partial, n_split, chunk, out_block_size, reduce_block_size, impl);
    }
    cumo_cuda_runtime_check_kernel_launch();

    arg2->in.ptr = reinterpret_cast<char*>(partial);
    arg2->in.step[0] = sizeof(TypeReduce);
    arg2->in_indexer.ndim = 1;
    arg2->in_indexer.shape[0] = partial_total_size;
    arg2->in_indexer.total_size = partial_total_size;
    return partial;
}

// True when both operands of a zip reduction walk by a single stride, which is
// what lets the offset helpers drop their decomposition.
static inline bool zip_axes_are_flat(const cumo_reduce_addr_t& ad, const cumo_reduce_addr_t& ad2) {
    return ad.in_out_flat && ad.in_reduce_flat && ad2.in_out_flat && ad2.in_reduce_flat;
}

// First pass of a split zip reduction. See reduce_partial_pass above.
template <typename TypeIn, typename TypeReduce, typename ReductionImpl>
TypeReduce* reduce_zip_partial_pass(cumo_na_reduction_arg_t arg, cumo_na_iarray_t in2, cumo_reduce_addr_t ad, cumo_reduce_addr_t ad2, int64_t n_split, int64_t reduce_total_size, cumo_na_reduction_arg_t* arg2, ReductionImpl& impl) {
    int64_t chunk = (reduce_total_size + n_split - 1) / n_split;
    int64_t partial_total_size = arg.out_indexer.total_size * n_split;
    TypeReduce* partial = reinterpret_cast<TypeReduce*>(cumo_cuda_runtime_malloc(sizeof(TypeReduce) * partial_total_size));

    int64_t out_block_size, reduce_block_size;
    reduce_block_split(ad, chunk, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (partial_total_size + out_block_size - 1) / out_block_size;
    int64_t grid_size = std::min(max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(TypeReduce) * max_block_size;

    if (zip_axes_are_flat(ad, ad2)) {
        reduction_zip_partial_kernel<true,TypeIn,TypeReduce,ReductionImpl><<<grid_size, max_block_size, shared_mem_size>>>(arg, in2, ad, ad2, partial, n_split, chunk, out_block_size, reduce_block_size, impl);
    } else {
        reduction_zip_partial_kernel<false,TypeIn,TypeReduce,ReductionImpl><<<grid_size, max_block_size, shared_mem_size>>>(arg, in2, ad, ad2, partial, n_split, chunk, out_block_size, reduce_block_size, impl);
    }
    cumo_cuda_runtime_check_kernel_launch();

    arg2->in.ptr = reinterpret_cast<char*>(partial);
    arg2->in.step[0] = sizeof(TypeReduce);
    arg2->in_indexer.ndim = 1;
    arg2->in_indexer.shape[0] = partial_total_size;
    arg2->in_indexer.total_size = partial_total_size;
    return partial;
}

}  // cumo_detail

template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    if (ad.in_out_flat && ad.in_reduce_flat && ad.out_flat) {
        cumo_detail::reduction_kernel<true,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, ad, out_block_size, reduce_block_size, impl);
    } else {
        cumo_detail::reduction_kernel<false,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, ad, out_block_size, reduce_block_size, impl);
    }
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

    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce<TypeIn, TypeOut, ReductionImpl>(arg, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t arg2 = arg;
    TypeReduce* partial = cumo_detail::reduce_partial_pass<TypeIn, TypeReduce, ReductionImpl>(arg, ad, n_split, reduce_total_size, &arg2, impl);
    cumo_reduce<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(arg2, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// Variant of cumo_reduce reading two inputs, for mulsum. in2 describes the same
// shape as arg.in, since the one in_indexer addresses both.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_zip(cumo_na_reduction_arg_t arg, cumo_na_iarray_t in2, ReductionImpl&& impl) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_na_reduction_arg_t arg2 = arg;
    arg2.in = in2;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);
    cumo_detail::cumo_reduce_addr_t ad2 = cumo_detail::make_reduce_addr(arg2, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    if (cumo_detail::zip_axes_are_flat(ad, ad2) && ad.out_flat) {
        cumo_detail::reduction_zip_kernel<true,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, in2, ad, ad2, out_block_size, reduce_block_size, impl);
    } else {
        cumo_detail::reduction_zip_kernel<false,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, in2, ad, ad2, out_block_size, reduce_block_size, impl);
    }
    cumo_cuda_runtime_check_kernel_launch();
}

// cumo_reduce_split for a zip reduction. The first pass reads both operands and
// the combine pass has only accumulators left, so it is the plain one.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_zip_split(cumo_na_reduction_arg_t arg, cumo_na_iarray_t in2, ReductionImpl&& impl) {
    using TypeReduce = decltype(impl.Identity(0));

    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_na_reduction_arg_t arg2 = arg;
    arg2.in = in2;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);
    cumo_detail::cumo_reduce_addr_t ad2 = cumo_detail::make_reduce_addr(arg2, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce_zip<TypeIn, TypeOut, ReductionImpl>(arg, in2, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t combine = arg;
    TypeReduce* partial = cumo_detail::reduce_zip_partial_pass<TypeIn, TypeReduce, ReductionImpl>(arg, in2, ad, ad2, n_split, reduce_total_size, &combine, impl);
    cumo_reduce<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(combine, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// Variant of cumo_reduce writing two results per output element, for minmax.
// See reduction_pair_kernel above. out2 has to describe the same shape as
// arg.out, since the one out_indexer addresses both.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_pair(cumo_na_reduction_arg_t arg, cumo_na_iarray_t out2, ReductionImpl&& impl) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);
    cumo_detail::set_reduce_addr_out2(&ad, arg, out2);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    if (ad.in_out_flat && ad.in_reduce_flat && ad.out_flat && ad.out2_flat) {
        cumo_detail::reduction_pair_kernel<true,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out2, ad, out_block_size, reduce_block_size, impl);
    } else {
        cumo_detail::reduction_pair_kernel<false,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, out2, ad, out_block_size, reduce_block_size, impl);
    }
    cumo_cuda_runtime_check_kernel_launch();
}

// cumo_reduce_split for the two-output form. Same constraint on Identity.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_pair_split(cumo_na_reduction_arg_t arg, cumo_na_iarray_t out2, ReductionImpl&& impl) {
    using TypeReduce = decltype(impl.Identity(0));

    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce_pair<TypeIn, TypeOut, ReductionImpl>(arg, out2, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t arg2 = arg;
    TypeReduce* partial = cumo_detail::reduce_partial_pass<TypeIn, TypeReduce, ReductionImpl>(arg, ad, n_split, reduce_total_size, &arg2, impl);
    cumo_reduce_pair<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(arg2, out2, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

// Variant of cumo_reduce for arg-reductions (argmax/argmin), which returns
// indices along the reduction axis. See reduction_arg_kernel above.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_arg(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t block_size = cumo_detail::max_block_size;
    int64_t grid_size = std::min(cumo_detail::max_grid_size, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity(0))) * block_size;

    if (ad.in_out_flat && ad.in_reduce_flat && ad.out_flat) {
        cumo_detail::reduction_arg_kernel<true,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, ad, out_block_size, reduce_block_size, impl);
    } else {
        cumo_detail::reduction_arg_kernel<false,TypeIn,TypeOut,ReductionImpl><<<grid_size, block_size, shared_mem_size>>>(arg, ad, out_block_size, reduce_block_size, impl);
    }
    cumo_cuda_runtime_check_kernel_launch();
}

// cumo_reduce_split for the arg form. The partials already carry indices along
// the reduce axis, so the second pass combines them with the plain kernel.
// Only for an impl whose Identity ignores its index argument.
template <typename TypeIn, typename TypeOut, typename ReductionImpl>
void cumo_reduce_arg_split(cumo_na_reduction_arg_t arg, ReductionImpl&& impl) {
    using TypeReduce = decltype(impl.Identity(0));

    if (arg.out_indexer.total_size == 0) {
        return;
    }

    int64_t reduce_total_size = arg.in_indexer.total_size / arg.out_indexer.total_size;
    cumo_detail::cumo_reduce_addr_t ad = cumo_detail::make_reduce_addr(arg, reduce_total_size);

    int64_t out_block_size, reduce_block_size;
    cumo_detail::reduce_block_split(ad, reduce_total_size, &out_block_size, &reduce_block_size);
    int64_t out_block_num = (arg.out_indexer.total_size + out_block_size - 1) / out_block_size;

    int64_t n_split = cumo_detail::reduce_split_count(reduce_total_size, out_block_num);
    if (n_split < 2) {
        cumo_reduce_arg<TypeIn, TypeOut, ReductionImpl>(arg, std::forward<ReductionImpl>(impl));
        return;
    }

    cumo_na_reduction_arg_t arg2 = arg;
    TypeReduce* partial = cumo_detail::reduce_partial_pass<TypeIn, TypeReduce, ReductionImpl, true>(arg, ad, n_split, reduce_total_size, &arg2, impl);
    cumo_reduce<TypeReduce, TypeOut, cumo_detail::reduce_combine<ReductionImpl>>(arg2, cumo_detail::reduce_combine<ReductionImpl>{impl});
    cumo_cuda_runtime_free(reinterpret_cast<char*>(partial));
}

#endif // CUMO_REDUCE_KERNEL_H
