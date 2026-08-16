#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum += next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_one; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum *= next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_min_impl {
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MAX; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? next : accum; }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_max_impl {
    __device__ dtype Identity(int64_t /*index*/) { return DATA_MIN; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? accum : next; }
    __device__ dtype MapOut(dtype accum) { return accum; }
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

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_prod_impl>(*arg, cumo_<%=type_name%>_prod_impl{});
}

void cumo_<%=type_name%>_min_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, dtype, cumo_<%=type_name%>_min_impl>(*arg, cumo_<%=type_name%>_min_impl{});
}

void cumo_<%=type_name%>_max_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, dtype, cumo_<%=type_name%>_max_impl>(*arg, cumo_<%=type_name%>_max_impl{});
}

void cumo_<%=type_name%>_ptp_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, dtype, cumo_<%=type_name%>_ptp_impl>(*arg, cumo_<%=type_name%>_ptp_impl{});
}
