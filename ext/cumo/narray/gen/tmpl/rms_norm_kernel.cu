<%
  acc = acc_type.empty? ? 'dtype' : acc_type
%>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

// The accumulation is the sum of squares cumo's rms already carries, so only
// what the row does with it differs: rms answers the root mean square and this
// answers its reciprocal, which is what multiplies the row. Holding eps inside
// the square root is what keeps a row of zeros finite.
//
// MapOut below hides the base's, and answers a different type from it, so what
// row_kernel.h names Stats is decided by the static type an impl is passed as.
// Everything here passes the derived one, and sliced to the base this would
// still compile and would size the stats buffer as the element type instead.
// A static_assert over decltype(impl.MapOut(...)) says that, and cannot live
// here: CUDA 11.8 reads the decltype in a non-template host function as a call
// to a __device__ function and refuses it, where the same spelling inside
// row_kernel.h's template is fine.
struct <%="cumo_#{c_iter}_impl"%> : cumo_<%=type_name%>_rms_impl {
    <%=acc%> eps;
    __device__ <%=acc%> MapOut(<%=acc%> accum) {
        return <%=acc%>(1) / sqrt(accum / n + eps);
    }
};

struct <%="cumo_#{c_iter}_apply"%> {
    const dtype* g;
    __device__ dtype operator()(dtype x, uint64_t col, <%=acc%> rrms) const {
        return <%=from_acc%>(<%=to_acc%>(x) * rrms * <%=to_acc%>(g[col]));
    }
};

// The row read once, sixteen bytes at a time, and held in registers.
template <int VPT>
__global__ void __launch_bounds__(cumo_detail::max_block_size) <%="cumo_#{c_iter}_held_kernel"%>(
        const dtype* x, dtype* y, const dtype* g, uint32_t rows, uint32_t cols, <%=acc%> eps)
{
    typedef cumo_detail::row_vec<dtype> vec_t;
    const int width = sizeof(vec_t) / sizeof(dtype);
    __shared__ <%=acc%> sh[32];
    uint32_t nvec = cols / width;
    const vec_t* gv = (const vec_t*)g;

    for (uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const vec_t* xr = (const vec_t*)(x + (size_t)row * cols);
        vec_t* yr = (vec_t*)(y + (size_t)row * cols);
        vec_t v[VPT];
        <%=acc%> s = 0, rrms;

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < nvec) {
                v[j] = xr[i];
#pragma unroll
                for (int k = 0; k < width; k++) {
                    <%=acc%> t = <%=to_acc%>(v[j].v[k]);
                    s += t * t;
                }
            }
        }
        rrms = <%=acc%>(1) / sqrt(cumo_detail::block_allreduce(s, sh, cumo_detail::row_add_op(), <%=acc%>(0)) / <%=acc%>(cols) + eps);

#pragma unroll
        for (int j = 0; j < VPT; j++) {
            uint32_t i = threadIdx.x + j * blockDim.x;
            if (i < nvec) {
                vec_t gg = gv[i], out;
#pragma unroll
                for (int k = 0; k < width; k++) {
                    out.v[k] = <%=from_acc%>(<%=to_acc%>(v[j].v[k]) * rrms * <%=to_acc%>(gg.v[k]));
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
        char *px, char *pg, char *py,
        uint64_t rows, uint64_t cols, double eps)
{
    <%="cumo_#{c_iter}_impl"%> impl;
    <%="cumo_#{c_iter}_apply"%> apply;

    const void* ptrs[3] = {px, pg, py};
    unsigned int block_dim;
    int vpt = cumo_row_held_shape<dtype>(rows, cols, ptrs, 3, &block_dim);

    if (vpt > 0) {
        unsigned int grid_dim = (unsigned int)(rows < 65536 ? rows : 65536);
#define CUMO_RMS_NORM_HELD(n)                                                                     \
        case n:                                                                                   \
            <%="cumo_#{c_iter}_held_kernel"%><n><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>( \
                    (const dtype*)px, (dtype*)py, (const dtype*)pg,                                \
                    (uint32_t)rows, (uint32_t)cols, (<%=acc%>)eps);                                \
            break;
        switch (vpt) {
        CUMO_RMS_NORM_HELD(1)
        CUMO_RMS_NORM_HELD(2)
        CUMO_RMS_NORM_HELD(4)
        CUMO_RMS_NORM_HELD(8)
        }
#undef CUMO_RMS_NORM_HELD
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }

    impl.n = (<%=acc%>)cols;
    impl.eps = (<%=acc%>)eps;
    apply.g = (const dtype*)pg;
    cumo_row_reduce_apply<dtype>(px, py, rows, cols, impl, apply);
}
