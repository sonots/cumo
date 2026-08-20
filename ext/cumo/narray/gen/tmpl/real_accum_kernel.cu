#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

// Reduce also combines two accumulators, so next has to be the accumulator type:
// the integer types widen to 64 bits and taking dtype there truncates every
// partial the shared-memory tree merges.
struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_zero; }
    __device__ <%=dtype%> MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(<%=dtype%> next, <%=dtype%>& accum) { accum += next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_one; }
    __device__ <%=dtype%> MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(<%=dtype%> next, <%=dtype%>& accum) { accum *= next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_min_impl {
<% if is_float %>
    // A NaN loses the comparison, so it is skipped unless the accumulator is
    // still the identity: the reduction then answers NaN only when every
    // element was NaN, and equal elements keep the earlier one as numo does.
    __device__ dtype Identity(int64_t /*index*/) { return (dtype)nan(""); }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { if (next < accum || !not_nan(accum)) { accum = next; } }
<% else %>
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MAX; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? next : accum; }
<% end %>
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_max_impl {
<% if is_float %>
    __device__ dtype Identity(int64_t /*index*/) { return (dtype)nan(""); }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { if (accum < next || !not_nan(accum)) { accum = next; } }
<% else %>
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MIN; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? accum : next; }
<% end %>
    __device__ dtype MapOut(dtype accum) { return accum; }
};

// min and max fused, so that minmax reads the input once. The rules per
// component are the two above verbatim: minmax must not answer differently
// from min and max.
struct cumo_<%=type_name%>_minmax_impl {
    struct MinAndMax {
        dtype min;
        dtype max;
    };
<% if is_float %>
    __device__ MinAndMax Identity(int64_t /*index*/) { return {(dtype)nan(""), (dtype)nan("")}; }
    __device__ MinAndMax MapIn(dtype in, int64_t /*index*/) { return {in, in}; }
    __device__ void Reduce(MinAndMax next, MinAndMax& accum) {
        if (next.min < accum.min || !not_nan(accum.min)) { accum.min = next.min; }
        if (accum.max < next.max || !not_nan(accum.max)) { accum.max = next.max; }
    }
<% else %>
    __device__ MinAndMax Identity(int64_t /*index*/) { return {DATA_MAX, DATA_MIN}; }
    __device__ MinAndMax MapIn(dtype in, int64_t /*index*/) { return {in, in}; }
    __device__ void Reduce(MinAndMax next, MinAndMax& accum) {
        accum.min = next.min < accum.min ? next.min : accum.min;
        accum.max = next.max < accum.max ? accum.max : next.max;
    }
<% end %>
    __device__ void MapOut(MinAndMax accum, dtype* out_min, dtype* out_max) {
        *out_min = accum.min;
        *out_max = accum.max;
    }
};

struct cumo_<%=type_name%>_ptp_impl {
    struct MinAndMax {
        dtype min;
        dtype max;
    };
    __device__ MinAndMax Identity(int64_t /*index*/) { return {DATA_MAX, DATA_MIN}; }
    __device__ MinAndMax MapIn(dtype in, int64_t /*index*/) { return {in, in}; }
    __device__ void Reduce(MinAndMax next, MinAndMax& accum) {
        if (next.min < accum.min) { accum.min = next.min; }
        if (accum.max < next.max) { accum.max = next.max; }
    }
    __device__ dtype MapOut(MinAndMax accum) {
    <% if is_float %>
        // A NaN loses both comparisons above, so every element being NaN leaves
        // the identity untouched. An empty reduction raises before it gets here.
        if (accum.max < accum.min) { return (dtype)nan(""); }
    <% end %>
        return m_sub(accum.max, accum.min);
    }
};

<% if is_float %>
// The nan-aware forms of the reductions above. sum, prod and their kin skip a
// NaN, so it maps to the identity and the tree never sees it. min and max go
// the other way: numo answers NaN as soon as one element is NaN, and a NaN
// absorbs in Reduce because it loses every comparison.
struct cumo_<%=type_name%>_sum_nan_impl {
    __device__ dtype Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return not_nan(in) ? in : m_zero; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_add(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_nan_impl {
    __device__ dtype Identity(int64_t /*index*/) { return m_one; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return not_nan(in) ? in : m_one; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_mul(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_min_nan_impl {
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MAX; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { if (!not_nan(next) || next < accum) { accum = next; } }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_max_nan_impl {
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MIN; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { if (!not_nan(next) || accum < next) { accum = next; } }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_minmax_nan_impl {
    struct MinAndMax {
        dtype min;
        dtype max;
    };
    __device__ MinAndMax Identity(int64_t /*index*/) { return {DATA_MAX, DATA_MIN}; }
    __device__ MinAndMax MapIn(dtype in, int64_t /*index*/) { return {in, in}; }
    __device__ void Reduce(MinAndMax next, MinAndMax& accum) {
        if (!not_nan(next.min) || next.min < accum.min) { accum.min = next.min; }
        if (!not_nan(next.max) || accum.max < next.max) { accum.max = next.max; }
    }
    __device__ void MapOut(MinAndMax accum, dtype* out_min, dtype* out_max) {
        *out_min = accum.min;
        *out_max = accum.max;
    }
};

struct cumo_<%=type_name%>_ptp_nan_impl : cumo_<%=type_name%>_minmax_nan_impl {
    __device__ dtype MapOut(MinAndMax accum) { return m_sub(accum.max, accum.min); }
};
<% end %>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, <%=dtype%>, cumo_<%=type_name%>_prod_impl>(*arg, cumo_<%=type_name%>_prod_impl{});
}

void cumo_<%=type_name%>_min_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_min_impl>(*arg, cumo_<%=type_name%>_min_impl{});
}

void cumo_<%=type_name%>_max_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_max_impl>(*arg, cumo_<%=type_name%>_max_impl{});
}

void cumo_<%=type_name%>_ptp_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_ptp_impl>(*arg, cumo_<%=type_name%>_ptp_impl{});
}

void cumo_<%=type_name%>_minmax_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2)
{
    cumo_reduce_pair_split<dtype, dtype, cumo_<%=type_name%>_minmax_impl>(*arg, *out2, cumo_<%=type_name%>_minmax_impl{});
}
<% if is_float %>

void cumo_<%=type_name%>_sum_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_sum_nan_impl>(*arg, cumo_<%=type_name%>_sum_nan_impl{});
}

void cumo_<%=type_name%>_prod_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_prod_nan_impl>(*arg, cumo_<%=type_name%>_prod_nan_impl{});
}

void cumo_<%=type_name%>_min_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_min_nan_impl>(*arg, cumo_<%=type_name%>_min_nan_impl{});
}

void cumo_<%=type_name%>_max_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_max_nan_impl>(*arg, cumo_<%=type_name%>_max_nan_impl{});
}

void cumo_<%=type_name%>_ptp_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_ptp_nan_impl>(*arg, cumo_<%=type_name%>_ptp_nan_impl{});
}

void cumo_<%=type_name%>_minmax_nan_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2)
{
    cumo_reduce_pair_split<dtype, dtype, cumo_<%=type_name%>_minmax_nan_impl>(*arg, *out2, cumo_<%=type_name%>_minmax_nan_impl{});
}
<% end %>
