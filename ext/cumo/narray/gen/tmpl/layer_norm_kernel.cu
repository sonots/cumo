<%
  acc = is_half ? 'float' : 'dtype'
  to_acc = is_half ? 'cumo_half2float' : ''
  from_acc = is_half ? 'cumo_float2half' : ''
%>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct <%="cumo_#{c_iter}_stats"%> {
    <%=acc%> mean;
    <%=acc%> rstd;
};

// What both paths reduce a row to. Layer norm divides the sum of squared
// deviations by n, where cumo's var divides by n - 1, so the two cannot share
// a MapOut however much of the accumulation they share.
struct <%="cumo_#{c_iter}_stats_impl"%> : cumo_<%=type_name%>_moments_impl {
    <%=acc%> eps;
    __device__ <%="cumo_#{c_iter}_stats"%> MapOut(Moments accum) {
        return {accum.mean, <%=acc%>(1) / sqrt(accum.m2 / accum.n + eps)};
    }
};

__device__ static inline dtype <%="cumo_#{c_iter}_apply"%>(
        dtype x, dtype g, dtype b, <%="cumo_#{c_iter}_stats"%> st)
{
    <%=acc%> v = (<%=to_acc%>(x) - st.mean) * st.rstd;
    return <%=from_acc%>(v * <%=to_acc%>(g) + <%=to_acc%>(b));
}

// One block takes one row, so the mean and the deviation it works out stay in
// shared memory between the two passes instead of going back to global memory
// as arrays of their own. Written out of mean, minus, divide and the rest, the
// same answer costs nine kernels and eight intermediate arrays.
__global__ void <%="cumo_#{c_iter}_kernel"%>(
        char *px, char *pg, char *pb, char *py,
        uint64_t rows, uint64_t cols, <%=acc%> eps)
{
    typedef cumo_<%=type_name%>_moments_impl::Moments Moments;

    extern __shared__ __align__(8) char <%="cumo_#{c_iter}_smem"%>[];
    Moments *sdata = reinterpret_cast<Moments*>(<%="cumo_#{c_iter}_smem"%>);
    <%="cumo_#{c_iter}_stats_impl"%> impl;
    unsigned int tid = threadIdx.x;
    const dtype *g = (const dtype*)pg;
    const dtype *b = (const dtype*)pb;

    impl.eps = eps;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const dtype *x = (const dtype*)px + row * cols;
        dtype *y = (dtype*)py + row * cols;
        Moments accum = impl.Identity(0);
        <%="cumo_#{c_iter}_stats"%> st;

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            impl.Reduce(impl.MapIn(x[i], 0), accum);
        }
        // The tree leaves the row's total in sdata[0] behind a barrier of its
        // own, and the second pass needs it in every thread, so the broadcast
        // is a read of that slot rather than the value handed back.
        cumo_detail::reduce_in_block(accum, sdata, tid, 1, blockDim.x, true, impl);
        st = impl.MapOut(sdata[0]);

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            y[i] = <%="cumo_#{c_iter}_apply"%>(x[i], g[i], b[i], st);
        }
        // Nobody may still be in the pass above when the next row rewrites sdata.
        __syncthreads();
    }
}

// The second half of the split path. blockIdx.y names the row, so finding one
// costs no division, and this path is only taken where rows is small.
__global__ void <%="cumo_#{c_iter}_apply_kernel"%>(
        char *px, char *pg, char *pb, char *py,
        <%="cumo_#{c_iter}_stats"%> *stats, uint64_t cols)
{
    uint64_t row = blockIdx.y;
    const dtype *x = (const dtype*)px + row * cols;
    dtype *y = (dtype*)py + row * cols;
    const dtype *g = (const dtype*)pg;
    const dtype *b = (const dtype*)pb;
    <%="cumo_#{c_iter}_stats"%> st = stats[row];

    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cols;
            i += (uint64_t)blockDim.x * gridDim.x) {
        y[i] = <%="cumo_#{c_iter}_apply"%>(x[i], g[i], b[i], st);
    }
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{c_iter}_kernel_launch"%>(
        char *px, char *pg, char *pb, char *py,
        uint64_t rows, uint64_t cols, double eps)
{
    typedef cumo_<%=type_name%>_moments_impl::Moments Moments;
    typedef <%="cumo_#{c_iter}_stats"%> stats_t;
    static const uint64_t max_row_blocks = 65536;
    int64_t want;
    unsigned int block_dim, grid_dim;
    size_t shared_mem_size;

    // A row gets one block, so a row too long for one block to walk leaves the
    // rest of the device idle unless there are other rows to fill it: one row
    // of a million takes 695.8us that way, against 69.6 for the nine kernels
    // this replaces. Past that the reduction goes through the split machinery,
    // which walks the row with the whole device, and a second kernel applies
    // what it found.
    if (rows < (uint64_t)cumo_detail::min_grid_size &&
            cols > (uint64_t)(cumo_detail::max_block_size * cumo_detail::min_reduce_per_thread)) {
        cumo_na_reduction_arg_t arg;
        stats_t *stats = (stats_t*)cumo_cuda_runtime_malloc(rows * sizeof(stats_t));
        dim3 apply_grid((unsigned int)cumo_get_grid_dim(cols), (unsigned int)rows);
        unsigned int apply_block = (unsigned int)cumo_get_block_dim(cols);

        memset(&arg, 0, sizeof(arg));
        arg.in.ptr = px;
        arg.in.step[0] = (ssize_t)(cols * sizeof(dtype));
        arg.in.step[1] = (ssize_t)sizeof(dtype);
        arg.in_indexer.ndim = 2;
        arg.in_indexer.total_size = rows * cols;
        arg.in_indexer.shape[0] = rows;
        arg.in_indexer.shape[1] = cols;
        arg.out.ptr = (char*)stats;
        arg.out.step[0] = (ssize_t)sizeof(stats_t);
        arg.out_indexer.ndim = 1;
        arg.out_indexer.total_size = rows;
        arg.out_indexer.shape[0] = rows;

        cumo_reduce_split<dtype, stats_t, <%="cumo_#{c_iter}_stats_impl"%>>(
                arg, <%="cumo_#{c_iter}_stats_impl"%>{{}, (<%=acc%>)eps});
        <%="cumo_#{c_iter}_apply_kernel"%><<<apply_grid, apply_block>>>(px, pg, pb, py, stats, cols);
        // Released before the launch is asked about, since that raises and a
        // raise here is a longjmp past anything below it.
        cumo_cuda_runtime_free((char*)stats);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }

    // How many threads to spend on a row depends on how many rows there are to
    // go around. Past enough rows to fill the device, giving each thread a run
    // of the row rather than an element or two keeps the tree short: 4096 rows
    // of 768 take 84.7us at a thread per element and 31.4 at sixteen. A single
    // row has no other block to overlap with and wants every thread it can use,
    // where the same sixteen cost 5.6us against 2.8.
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
    // A block that takes several rows amortizes its own launch, so the grid is
    // held below the row count on purpose. This is a tuning number and not the
    // ceiling for this axis, which is CUMO_MAX_GRID_DIM: a million rows of 8
    // take 1286.0us at a block per row and 1177.5 here, and two million of 4
    // take 2357.5 against 2094.4.
    grid_dim = (unsigned int)(rows < max_row_blocks ? rows : max_row_blocks);
    shared_mem_size = block_dim * sizeof(Moments);

    <%="cumo_#{c_iter}_kernel"%><<<grid_dim, block_dim, shared_mem_size>>>(
            px, pg, pb, py, rows, cols, (<%=acc%>)eps);
    cumo_cuda_runtime_check_kernel_launch();
}
