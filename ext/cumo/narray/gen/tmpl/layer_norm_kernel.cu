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

    impl.eps = (<%=acc%>)eps;
    apply.g = (const dtype*)pg;
    apply.b = (const dtype*)pb;
    cumo_row_reduce_apply<dtype>(px, py, rows, cols, impl, apply);
}
