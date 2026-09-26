#ifndef CUMO_ROW_KERNEL_H
#define CUMO_ROW_KERNEL_H

#include "cumo/reduce_kernel.h"

// Reducing each row of an array to a few numbers and then walking the row again
// to write it out is the shape layer norm and softmax share. One block takes one
// row, so what the reduction found stays in shared memory between the two passes
// instead of going back to global memory as an array of its own: written out of
// mean, minus, divide and the rest, a layer norm costs nine kernels and eight
// intermediate arrays.
//
// Impl is a reduction impl as reduce_kernel.h means it, whose MapOut answers
// whatever the second pass needs. Apply turns one element into one element,
// taking its position along the row so that a per-column operand can be read.
//
// Two things Impl has to hold to, neither of which the compiler can check.
// It must not read the index its Identity and MapIn are handed: the kernel
// below always passes 0, while the split path passes the element's real offset,
// so an impl that reads it answers differently depending on which path a shape
// takes. And it must be default constructible, since cumo_reduce_split's
// reduce_combine names its accumulator as decltype(Impl().Identity(0)).

namespace cumo_detail {

template <typename TypeIn, typename TypeOut, typename Stats, typename Impl, typename Apply>
__global__ void row_reduce_apply_kernel(
        const TypeIn* x, TypeOut* y, Stats* stats_out, uint64_t rows, uint64_t cols, Impl impl, Apply apply)
{
    typedef decltype(impl.Identity(0)) Accum;
    static_assert(alignof(Accum) <= 8,
                  "the shared memory below is declared __align__(8), as every other "
                  "dynamic one in cumo is, so a wider accumulator would land unaligned");

    extern __shared__ __align__(8) char cumo_row_smem[];
    Accum* sdata = reinterpret_cast<Accum*>(cumo_row_smem);
    unsigned int tid = threadIdx.x;

    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const TypeIn* xr = x + row * cols;
        TypeOut* yr = y + row * cols;
        Accum accum = impl.Identity(0);

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            impl.Reduce(impl.MapIn(xr[i], 0), accum);
        }
        // The tree leaves the row's total in sdata[0] behind a barrier of its
        // own, and the second pass needs it in every thread, so the broadcast is
        // a read of that slot rather than the value handed back.
        reduce_in_block(accum, sdata, tid, 1, blockDim.x, true, impl);
        auto stats = impl.MapOut(sdata[0]);

        // What the row was reduced to, for a caller that wants it back. One
        // thread writes it, and the pass below reads only registers, so no
        // barrier is owed between the two.
        if (stats_out != NULL && tid == 0) {
            stats_out[row] = stats;
        }
        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            yr[i] = apply(xr[i], i, stats);
        }
        // Nobody may still be in the pass above when the next row rewrites
        // sdata. Removing this leaves every value test passing and is only
        // caught by compute-sanitizer --tool racecheck.
        __syncthreads();
    }
}

// The second half of the split path. blockIdx.y names the row, so finding one
// costs no division, and this path is only taken where rows is small.
template <typename TypeIn, typename TypeOut, typename Stats, typename Apply>
__global__ void row_apply_kernel(
        const TypeIn* x, TypeOut* y, const Stats* stats, uint64_t cols, Apply apply)
{
    uint64_t row = blockIdx.y;
    const TypeIn* xr = x + row * cols;
    TypeOut* yr = y + row * cols;
    Stats st = stats[row];

    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cols;
            i += (uint64_t)blockDim.x * gridDim.x) {
        yr[i] = apply(xr[i], i, st);
    }
}

}  // namespace cumo_detail

// Reduces each row of a contiguous rows x cols array with impl and writes the
// row back through apply. Both arrays are laid out the same way and neither may
// be the other.
//
// pstats, when it is not NULL, takes what each row reduced to. Its elements are
// the accumulator type, decltype(impl.MapOut(...)), which is not the element
// type for a class whose accumulator is wider, so the buffer behind it has to
// be sized in those. Nothing is written there for a row of no length.
template <typename TypeIn, typename TypeOut, typename Impl, typename Apply>
void cumo_row_reduce_apply_out(
        char* px, char* py, char* pstats, uint64_t rows, uint64_t cols, Impl impl, Apply apply)
{
    typedef decltype(impl.Identity(0)) Accum;
    typedef decltype(impl.MapOut(impl.Identity(0))) Stats;
    // A block that takes several rows amortizes its own launch, so the grid is
    // held below the row count on purpose. This is a tuning number and not the
    // ceiling for this axis, which is CUMO_MAX_GRID_DIM: a million rows of 8
    // take 1286.0us at a block per row and 1177.5 here, and two million of 4
    // take 2357.5 against 2094.4.
    static const uint64_t max_row_blocks = 65536;
    int64_t want;
    unsigned int block_dim, grid_dim;
    size_t shared_mem_size;

    if (rows == 0 || cols == 0) {
        return;
    }

    // A row gets one block, so a row too long for one block to walk leaves the
    // rest of the device idle unless there are other rows to fill it, and one
    // such row is slower than the nine kernels a layer norm otherwise costs.
    // Past that the reduction goes through the split
    // machinery, which walks the row with the whole device, and a second kernel
    // applies what it found.
    if (rows < (uint64_t)cumo_detail::min_grid_size &&
            cols > (uint64_t)(cumo_detail::max_block_size * cumo_detail::min_reduce_per_thread)) {
        cumo_na_reduction_arg_t arg;
        // The reduction writes the row totals wherever it is pointed, so a
        // caller that wants them back is handed the buffer rather than a copy.
        Stats* stats = pstats != NULL
            ? (Stats*)pstats
            : (Stats*)cumo_cuda_runtime_malloc(rows * sizeof(Stats));
        // rows is below min_grid_size to be here, which is well inside the y
        // limit, but that is a threshold from reduce_kernel.h and not a promise
        // about this axis, so the clamp is written out rather than assumed.
        dim3 apply_grid((unsigned int)cumo_get_grid_dim(cols),
                        (unsigned int)(rows < CUMO_MAX_GRID_DIM_Y ? rows : CUMO_MAX_GRID_DIM_Y));
        unsigned int apply_block = (unsigned int)cumo_get_block_dim(cols);

        memset(&arg, 0, sizeof(arg));
        arg.in.ptr = px;
        arg.in.step[0] = (ssize_t)(cols * sizeof(TypeIn));
        arg.in.step[1] = (ssize_t)sizeof(TypeIn);
        arg.in_indexer.ndim = 2;
        arg.in_indexer.total_size = rows * cols;
        arg.in_indexer.shape[0] = rows;
        arg.in_indexer.shape[1] = cols;
        arg.out.ptr = (char*)stats;
        arg.out.step[0] = (ssize_t)sizeof(Stats);
        arg.out_indexer.ndim = 1;
        arg.out_indexer.total_size = rows;
        arg.out_indexer.shape[0] = rows;

        // held is what the failure path frees, so it may only ever name scratch.
        // Handing it a buffer a live Ruby array owns would return that to the
        // pool and leave the array's own free to come back to it.
        cumo_reduce_split<TypeIn, Stats, Impl>(arg, Impl(impl),
                                               pstats != NULL ? NULL : (char*)stats);
        cumo_detail::row_apply_kernel<TypeIn, TypeOut, Stats, Apply><<<apply_grid, apply_block, 0, cumo_cuda_stream()>>>(
                (const TypeIn*)px, (TypeOut*)py, stats, cols, apply);
        if (pstats != NULL) {
            cumo_cuda_runtime_check_kernel_launch();
        } else {
            cumo_check_launch_holding(stats);
            cumo_cuda_runtime_free((char*)stats);
        }
        return;
    }

    // How many threads to spend on a row depends on how many rows there are to
    // go around. Past enough rows to fill the device, giving each thread a run
    // of the row rather than an element or two keeps the tree short. A single
    // row has no other block to overlap with and wants every thread it can use,
    // where the same run a thread is slower.
    if (rows < (uint64_t)cumo_detail::min_grid_size) {
        want = cumo_detail::round_up_to_power_of_2((int64_t)cols);
    } else {
        want = cumo_detail::round_up_to_power_of_2(
                ((int64_t)cols + cumo_detail::min_reduce_per_thread - 1) / cumo_detail::min_reduce_per_thread);
    }
    // reduce_in_block walks a power-of-two tree, and at one thread it returns
    // without writing sdata at all, which would leave the broadcast reading
    // whatever shared memory held. A whole warp is the smallest block here.
    if (want < cumo_detail::warp_size) want = cumo_detail::warp_size;
    if (want > cumo_detail::max_block_size) want = cumo_detail::max_block_size;
    block_dim = (unsigned int)want;
    grid_dim = (unsigned int)(rows < max_row_blocks ? rows : max_row_blocks);
    shared_mem_size = block_dim * sizeof(Accum);

    cumo_detail::row_reduce_apply_kernel<TypeIn, TypeOut, Stats, Impl, Apply><<<grid_dim, block_dim, shared_mem_size, cumo_cuda_stream()>>>(
            (const TypeIn*)px, (TypeOut*)py, (Stats*)pstats, rows, cols, impl, apply);
    cumo_cuda_runtime_check_kernel_launch();
}

// The shape layer_norm, rms_norm and softmax take: one array in, one of the
// same type out, and nothing kept from the reduction.
template <typename TypeIn, typename Impl, typename Apply>
void cumo_row_reduce_apply(
        char* px, char* py, uint64_t rows, uint64_t cols, Impl impl, Apply apply)
{
    cumo_row_reduce_apply_out<TypeIn, TypeIn, Impl, Apply>(px, py, NULL, rows, cols, impl, apply);
}

#endif // CUMO_ROW_KERNEL_H
