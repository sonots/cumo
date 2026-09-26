<%
  acc = acc_type.empty? ? 'dtype' : acc_type
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

// Layer norm divides the sum of squared deviations by n, where cumo's var
// divides by n - 1, so the two cannot share a MapOut however much of the
// accumulation they share.
struct <%="cumo_#{c_iter}_impl"%> : cumo_<%=type_name%>_moments_impl {
    <%=acc%> eps;
    __device__ <%="cumo_#{c_iter}_stats"%> MapOut(Moments accum) {
        return {accum.mean, <%=acc%>(1) / sqrt(accum.m2 / accum.n + eps)};
    }
};

struct <%="cumo_#{c_iter}_apply"%> {
    const dtype* g;
    const dtype* b;
    __device__ dtype operator()(dtype x, uint64_t col, <%="cumo_#{c_iter}_stats"%> st) const {
        <%=acc%> v = (<%=to_acc%>(x) - st.mean) * st.rstd;
        return <%=from_acc%>(v * <%=to_acc%>(g[col]) + <%=to_acc%>(b[col]));
    }
};

// Holding the row lets the variance be taken about the mean in a second pass,
// where updating both element by element costs a division each. Dividing each
// term before summing costs precision, so it is kept for a sum that overflows.
template <int VPT>
__global__ void __launch_bounds__(cumo_detail::max_block_size) <%="cumo_#{c_iter}_held_kernel"%>(
        const dtype* x, dtype* y, const dtype* g, const dtype* b,
        uint32_t rows, uint32_t cols, <%=acc%> eps)
{
    typedef cumo_detail::row_vec<dtype> vec_t;
    const int width = sizeof(vec_t) / sizeof(dtype);
    __shared__ <%=acc%> sh[32];
    uint32_t nvec = cols / width;
    const vec_t* gv = (const vec_t*)g;
    const vec_t* bv = (const vec_t*)b;

    for (uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const vec_t* xr = (const vec_t*)(x + (size_t)row * cols);
        vec_t* yr = (vec_t*)(y + (size_t)row * cols);
        vec_t v[VPT];
        <%=acc%> s = 0, q = 0, mean, rstd;

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < nvec) {
                v[j] = xr[i];
#pragma unroll
                for (int k = 0; k < width; k++) s += <%=to_acc%>(v[j].v[k]);
            }
        }
        mean = cumo_detail::block_allreduce(s, sh, cumo_detail::row_add_op(), <%=acc%>(0));
        if (isinf(mean)) {
            <%=acc%> rcols = <%=acc%>(1) / <%=acc%>(cols);
            s = 0;
#pragma unroll
            for (int j = 0; j < VPT; j++) {
                uint32_t i = threadIdx.x + j * blockDim.x;
                if (i < nvec) {
#pragma unroll
                    for (int k = 0; k < width; k++) s += <%=to_acc%>(v[j].v[k]) * rcols;
                }
            }
            mean = cumo_detail::block_allreduce(s, sh, cumo_detail::row_add_op(), <%=acc%>(0));
        } else {
            mean /= <%=acc%>(cols);
        }

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < nvec) {
#pragma unroll
                for (int k = 0; k < width; k++) {
                    <%=acc%> d = <%=to_acc%>(v[j].v[k]) - mean;
                    q += d * d;
                }
            }
        }
        rstd = <%=acc%>(1) / sqrt(cumo_detail::block_allreduce(q, sh, cumo_detail::row_add_op(), <%=acc%>(0)) / <%=acc%>(cols) + eps);

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < nvec) {
                vec_t gg = gv[i], bb = bv[i], out;
#pragma unroll
                for (int k = 0; k < width; k++) {
                    <%=acc%> t = (<%=to_acc%>(v[j].v[k]) - mean) * rstd;
                    out.v[k] = <%=from_acc%>(t * <%=to_acc%>(gg.v[k]) + <%=to_acc%>(bb.v[k]));
                }
                yr[i] = out;
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

void <%="cumo_#{c_iter}_kernel_launch"%>(
        char *px, char *pg, char *pb, char *py,
        uint64_t rows, uint64_t cols, double eps)
{
    <%="cumo_#{c_iter}_impl"%> impl;
    <%="cumo_#{c_iter}_apply"%> apply;

    const void* ptrs[4] = {px, pg, pb, py};
    unsigned int block_dim;
    int vpt = cumo_row_held_shape<dtype>(rows, cols, ptrs, 4, &block_dim);

    if (vpt > 0) {
        unsigned int grid_dim = (unsigned int)(rows < cumo_detail::max_row_blocks ? rows : cumo_detail::max_row_blocks);
#define CUMO_LAYER_NORM_HELD(n)                                                                   \
        case n:                                                                                   \
            <%="cumo_#{c_iter}_held_kernel"%><n><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>( \
                    (const dtype*)px, (dtype*)py, (const dtype*)pg, (const dtype*)pb,              \
                    (uint32_t)rows, (uint32_t)cols, (<%=acc%>)eps);                                \
            cumo_cuda_runtime_check_kernel_launch();                                              \
            return;
        switch (vpt) {
        CUMO_LAYER_NORM_HELD(1)
        CUMO_LAYER_NORM_HELD(2)
        CUMO_LAYER_NORM_HELD(4)
        CUMO_LAYER_NORM_HELD(8)
        }
#undef CUMO_LAYER_NORM_HELD
    }

    impl.eps = (<%=acc%>)eps;
    apply.g = (const dtype*)pg;
    apply.b = (const dtype*)pb;
    cumo_row_reduce_apply<dtype>(px, py, rows, cols, impl, apply);
}
