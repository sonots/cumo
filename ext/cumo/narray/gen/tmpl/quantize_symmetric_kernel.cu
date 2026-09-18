<%
  acc = acc_type.empty? ? 'dtype' : acc_type
%>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

// The row is reduced to the largest magnitude in it, which is the only thing
// the scale depends on, and the pass back over the row divides by that scale.
// A row of zeros has no scale to divide by: it answers a scale of zero and a
// row of zeros, which is the value the quantized row stands for either way.
// A NaN is carried rather than skipped, so that a row holding one answers a
// scale of NaN instead of the zero a row of zeros answers. Neither a NaN nor an
// infinity has a quantized value to stand for, and telling the two apart is
// what lets the row below say so.
struct <%="cumo_#{c_iter}_impl"%> {
    __device__ <%=acc%> Identity(int64_t) { return <%=acc%>(0); }
    __device__ <%=acc%> MapIn(dtype x, int64_t) {
        <%=acc%> v = <%=to_acc%>(x);
        return v < <%=acc%>(0) ? -v : v;
    }
    __device__ void Reduce(<%=acc%> next, <%=acc%>& accum) {
        accum = (next != next || next > accum) ? next : accum;
    }
    __device__ <%=acc%> MapOut(<%=acc%> accum) { return accum / <%=acc%>(127); }
};

struct <%="cumo_#{c_iter}_apply"%> {
    __device__ int8_t operator()(dtype x, uint64_t, <%=acc%> scale) const {
        <%=acc%> q;

        // A scale that is not a finite positive number leaves no value to round
        // to: zero is the row of zeros, and a NaN or an infinity got there from
        // an element that 8 bits cannot stand for either.
        if (!(scale > <%=acc%>(0)) || !(scale < INFINITY)) {
            return (int8_t)0;
        }
        // Ties away from zero, the way Cumo::NArray#round takes them. round()
        // is that rule already, where floor(|q| + 0.5) costs a second one of
        // its own: in double the pair is what makes the row slower than the
        // six kernels it replaces.
        q = round(<%=to_acc%>(x) / scale);
        // The largest magnitude in the row maps exactly onto the limit, but a
        // rounding on the way there could still answer one past it.
        if (q > <%=acc%>(127)) q = <%=acc%>(127);
        if (q < -<%=acc%>(127)) q = -<%=acc%>(127);
        return (int8_t)q;
    }
};

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{c_iter}_kernel_launch"%>(
        char *px, char *pq, char *ps, uint64_t rows, uint64_t cols)
{
    cumo_row_reduce_apply_out<dtype, int8_t, <%="cumo_#{c_iter}_impl"%>, <%="cumo_#{c_iter}_apply"%>>(
            px, pq, ps, rows, cols, <%="cumo_#{c_iter}_impl"%>{}, <%="cumo_#{c_iter}_apply"%>{});
}
