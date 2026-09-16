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

    impl.n = (<%=acc%>)cols;
    impl.eps = (<%=acc%>)eps;
    apply.g = (const dtype*)pg;
    cumo_row_reduce_apply<dtype>(px, py, rows, cols, impl, apply);
}
