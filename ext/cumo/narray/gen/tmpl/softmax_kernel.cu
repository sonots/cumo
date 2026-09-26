<%
  acc = acc_type.empty? ? 'dtype' : acc_type
%>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct <%="cumo_#{c_iter}_max_impl"%> {
    __device__ <%=acc%> Identity(int64_t /*index*/) { return -INFINITY; }
    __device__ <%=acc%> MapIn(dtype in, int64_t /*index*/) { return <%=to_acc%>(in); }
    __device__ void Reduce(<%=acc%> next, <%=acc%>& accum) { if (next > accum) accum = next; }
    __device__ <%=acc%> MapOut(<%=acc%> accum) { return accum; }
};

// Answers the reciprocal so that the pass below multiplies rather than divides.
struct <%="cumo_#{c_iter}_rden_impl"%> {
    __device__ <%=acc%> Identity(int64_t /*index*/) { return 0; }
    __device__ <%=acc%> MapIn(dtype in, int64_t /*index*/) { return <%=to_acc%>(in); }
    __device__ void Reduce(<%=acc%> next, <%=acc%>& accum) { accum += next; }
    __device__ <%=acc%> MapOut(<%=acc%> accum) { return <%=acc%>(1) / accum; }
};

struct <%="cumo_#{c_iter}_shift_apply"%> {
    __device__ dtype operator()(dtype x, uint64_t /*col*/, <%=acc%> max) const {
        return <%=from_acc%>(exp(<%=to_acc%>(x) - max));
    }
};

struct <%="cumo_#{c_iter}_scale_apply"%> {
    __device__ dtype operator()(dtype x, uint64_t /*col*/, <%=acc%> rden) const {
        return <%=from_acc%>(<%=to_acc%>(x) * rden);
    }
};

// Three passes over the row. Storing exp(x - max) is what lets the sum and the
// divide share it, and the one place with room is the output, so the second
// pass writes there and reduces what it wrote. One exponential an element,
// where taking the maximum and the sum together costs two.
__global__ void <%="cumo_#{c_iter}_kernel"%>(
        const dtype* x, dtype* y, uint64_t rows, uint64_t cols)
{
    extern __shared__ __align__(8) char <%="cumo_#{c_iter}_smem"%>[];
    <%=acc%>* sdata = reinterpret_cast<<%=acc%>*>(<%="cumo_#{c_iter}_smem"%>);
    <%="cumo_#{c_iter}_max_impl"%> max_impl;
    <%="cumo_#{c_iter}_rden_impl"%> rden_impl;
    unsigned int tid = threadIdx.x;

    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const dtype* xr = x + row * cols;
        dtype* yr = y + row * cols;
        <%=acc%> m = max_impl.Identity(0);
        <%=acc%> s = 0;
        <%=acc%> rden;

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            max_impl.Reduce(max_impl.MapIn(xr[i], 0), m);
        }
        cumo_detail::reduce_in_block(m, sdata, tid, 1, blockDim.x, true, max_impl);
        m = sdata[0];
        __syncthreads();

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            <%=acc%> t = exp(<%=to_acc%>(xr[i]) - m);
            yr[i] = <%=from_acc%>(t);
            s += t;
        }
        cumo_detail::reduce_in_block(s, sdata, tid, 1, blockDim.x, true, rden_impl);
        rden = <%=acc%>(1) / sdata[0];
        __syncthreads();

        for (uint64_t i = tid; i < cols; i += blockDim.x) {
            yr[i] = <%=from_acc%>(<%=to_acc%>(yr[i]) * rden);
        }
        __syncthreads();
    }
}

struct <%="cumo_#{c_iter}_fmax_op"%> {
    __device__ <%=acc%> operator()(<%=acc%> a, <%=acc%> b) const { return fmax(a, b); }
};

struct <%="cumo_#{c_iter}_add_op"%> {
    __device__ <%=acc%> operator()(<%=acc%> a, <%=acc%> b) const { return a + b; }
};

// Warp shuffles first, then one value a warp through shared memory. blockDim.x
// is a whole number of warps here.
template <typename Op>
__device__ <%=acc%> <%="cumo_#{c_iter}_block_reduce"%>(<%=acc%> v, <%=acc%>* sh, Op op, <%=acc%> identity)
{
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;

    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (warps == 1) return v;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < warps ? sh[lane] : identity;
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    // Nobody may read sh again until every warp has taken its value from it.
    __syncthreads();
    return v;
}

// The same row held in registers, VPT elements a thread, so that it is read once
// and written once where the kernel above reads it twice, writes it twice and
// reads it back. Indices are 32-bit and the maximum is an fmax rather than a
// branch: either one alone costs this kernel a third of its speed. The launcher
// keeps rows under 2^31 so that the row step cannot wrap. fmax drops a NaN as
// max_impl does, and the row goes NaN anyway through its exponential.
template <int VPT>
__global__ void __launch_bounds__(cumo_detail::max_block_size) <%="cumo_#{c_iter}_reg_kernel"%>(
        const dtype* x, dtype* y, uint32_t rows, uint32_t cols)
{
    __shared__ <%=acc%> sh[32];
    <%="cumo_#{c_iter}_fmax_op"%> max_op;
    <%="cumo_#{c_iter}_add_op"%> sum_op;

    for (uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const dtype* xr = x + (size_t)row * cols;
        dtype* yr = y + (size_t)row * cols;
        <%=acc%> v[VPT];
        <%=acc%> m = -INFINITY;
        <%=acc%> s = 0;
        <%=acc%> rden;

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < cols) {
                v[j] = <%=to_acc%>(xr[i]);
                m = fmax(m, v[j]);
            }
        }
        m = <%="cumo_#{c_iter}_block_reduce"%>(m, sh, max_op, <%=acc%>(-INFINITY));

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < cols) {
                v[j] = exp(v[j] - m);
                s += v[j];
            }
        }
        rden = <%=acc%>(1) / <%="cumo_#{c_iter}_block_reduce"%>(s, sh, sum_op, <%=acc%>(0));

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < cols) {
                yr[i] = <%=from_acc%>(v[j] * rden);
            }
        }
    }
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *py, uint64_t rows, uint64_t cols)
{
    static const uint64_t max_row_blocks = 65536;
    int64_t want;
    unsigned int block_dim, grid_dim;

    if (rows == 0 || cols == 0) return;

    // A row too long for one block to walk leaves the rest of the device idle
    // unless there are other rows to fill it, so the reduction goes through the
    // split machinery instead. The passes stay the three above, run as separate
    // launches: taking the maximum and the sum in one would cost a second
    // exponential an element, which is what a DFloat row cannot afford.
    if (rows < (uint64_t)cumo_detail::min_grid_size &&
            cols > (uint64_t)(cumo_detail::max_block_size * cumo_detail::min_reduce_per_thread)) {
        cumo_na_reduction_arg_t arg;
        <%=acc%>* stats = (<%=acc%>*)cumo_cuda_runtime_malloc(rows * sizeof(<%=acc%>));
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
        arg.out.step[0] = (ssize_t)sizeof(<%=acc%>);
        arg.out_indexer.ndim = 1;
        arg.out_indexer.total_size = rows;
        arg.out_indexer.shape[0] = rows;

        cumo_reduce_split<dtype, <%=acc%>, <%="cumo_#{c_iter}_max_impl"%>>(
                arg, <%="cumo_#{c_iter}_max_impl"%>{}, (char*)stats);
        cumo_detail::row_apply_kernel<dtype, dtype, <%=acc%>, <%="cumo_#{c_iter}_shift_apply"%>>
                <<<apply_grid, apply_block, 0, cumo_cuda_stream()>>>(
                    (const dtype*)px, (dtype*)py, stats, cols, <%="cumo_#{c_iter}_shift_apply"%>{});
        cumo_check_launch_holding(stats);

        // The sum is of what the pass above wrote, so the second reduction reads
        // the output rather than the input.
        arg.in.ptr = py;
        cumo_reduce_split<dtype, <%=acc%>, <%="cumo_#{c_iter}_rden_impl"%>>(
                arg, <%="cumo_#{c_iter}_rden_impl"%>{}, (char*)stats);
        cumo_detail::row_apply_kernel<dtype, dtype, <%=acc%>, <%="cumo_#{c_iter}_scale_apply"%>>
                <<<apply_grid, apply_block, 0, cumo_cuda_stream()>>>(
                    (const dtype*)py, (dtype*)py, stats, cols, <%="cumo_#{c_iter}_scale_apply"%>{});
        cumo_check_launch_holding(stats);
        cumo_cuda_runtime_free((char*)stats);
        return;
    }

    if (rows < (uint64_t)cumo_detail::min_grid_size) {
        want = cumo_detail::round_up_to_power_of_2((int64_t)cols);
    } else {
        want = cumo_detail::round_up_to_power_of_2(
                ((int64_t)cols + cumo_detail::min_reduce_per_thread - 1) / cumo_detail::min_reduce_per_thread);
    }
    if (want < cumo_detail::warp_size) want = cumo_detail::warp_size;
    if (want > cumo_detail::max_block_size) want = cumo_detail::max_block_size;
    block_dim = (unsigned int)want;
    grid_dim = (unsigned int)(rows < max_row_blocks ? rows : max_row_blocks);

    {
        uint64_t vpt = (cols + block_dim - 1) / block_dim;
        cudaStream_t stream = cumo_cuda_stream();

        // Sixteen a thread is where holding the row stops paying: 300 rows of
        // 16384 at thirty-two came out either side of the kernel above.
#define CUMO_SOFTMAX_REG_LAUNCH(n)                                                              \
        if (vpt <= (n)) {                                                                       \
            <%="cumo_#{c_iter}_reg_kernel"%><n><<<grid_dim, block_dim, 0, stream>>>(             \
                    (const dtype*)px, (dtype*)py, (uint32_t)rows, (uint32_t)cols);              \
            cumo_cuda_runtime_check_kernel_launch();                                            \
            return;                                                                             \
        }
        if (rows <= INT32_MAX) {
            CUMO_SOFTMAX_REG_LAUNCH(1)
            CUMO_SOFTMAX_REG_LAUNCH(2)
            CUMO_SOFTMAX_REG_LAUNCH(4)
            CUMO_SOFTMAX_REG_LAUNCH(8)
            CUMO_SOFTMAX_REG_LAUNCH(16)
        }
#undef CUMO_SOFTMAX_REG_LAUNCH
    }

    <%="cumo_#{c_iter}_kernel"%><<<grid_dim, block_dim, block_dim * sizeof(<%=acc%>), cumo_cuda_stream()>>>(
            (const dtype*)px, (dtype*)py, rows, cols);
    cumo_cuda_runtime_check_kernel_launch();
}
