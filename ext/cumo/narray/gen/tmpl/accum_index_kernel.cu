<% unless defined?($cumo_narray_gen_tmpl_accum_index_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_index_kernel_included = 1 %>
<% unless type_name == 'robject' %>

<%   [64,32].each do |i| %>
#define idx_t int<%=i%>_t

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_min_index_int<%=i%>_impl {
    struct MinAndArgMin {
        dtype min;
        idx_t argmin;
    };
    __device__ MinAndArgMin Identity() { return {DATA_MAX, 0}; }
    __device__ MinAndArgMin MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(MinAndArgMin next, MinAndArgMin& accum) {
        if (accum.min > next.min) {
            accum = next;
        }
    }
    __device__ idx_t MapOut(MinAndArgMin accum) { return accum.argmin; }
};

struct cumo_<%=type_name%>_max_index_int<%=i%>_impl {
    struct MaxAndArgMax {
        dtype max;
        idx_t argmax;
    };
    __device__ MaxAndArgMax Identity() { return {DATA_MIN, 0}; }
    __device__ MaxAndArgMax MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
        if (accum.max < next.max) {
            accum = next;
        }
    }
    __device__ idx_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
};

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_min_index_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, idx_t, cumo_<%=type_name%>_min_index_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_min_index_int<%=i%>_impl{});
}

void cumo_<%=type_name%>_max_index_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, idx_t, cumo_<%=type_name%>_max_index_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_max_index_int<%=i%>_impl{});
}

#undef idx_t
<% end %>

<% end %>
<% end %>
