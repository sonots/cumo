<% unless defined?($cumo_narray_gen_tmpl_accum_arg_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_arg_kernel_included = 1 %>
<% unless type_name == 'robject' %>

<%   [64,32].each do |i| %>
#define idx_t int<%=i%>_t

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_argmin_int<%=i%>_impl {
    struct ValueAndIndex {
        dtype value;
        idx_t index;
    };
    // The identity has to lose to any element, including a NaN, and it
    // carries the largest index so that a tie goes to the earlier one.
    // NaN is the identity for the float types, the way it is for min and
    // max, so that every element being NaN answers the first index as
    // numo does.
    __device__ ValueAndIndex Identity(idx_t /*index*/) { return {<% if is_float %>(dtype)nan("")<% else %>DATA_MAX<% end %>, INT<%=i%>_MAX}; }
    __device__ ValueAndIndex MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(ValueAndIndex next, ValueAndIndex& accum) {
<% if is_float %>
        bool accum_nan = !not_nan(accum.value);
        bool next_nan = !not_nan(next.value);
        if (accum_nan || next_nan) {
            if (accum_nan && (!next_nan || next.index < accum.index)) { accum = next; }
            return;
        }
<% end %>
        if (accum.value > next.value || (accum.value == next.value && next.index < accum.index)) { accum = next; }
    }
    __device__ idx_t MapOut(ValueAndIndex accum) { return accum.index; }
};

<% if is_float %>
// numo answers the index of the first NaN as soon as one element is NaN,
// so a NaN wins outright here and the earliest of them wins among
// themselves. The identity cannot be NaN for that reason.
struct cumo_<%=type_name%>_argmin_nan_int<%=i%>_impl {
    struct ValueAndIndex {
        dtype value;
        idx_t index;
    };
    __device__ ValueAndIndex Identity(idx_t /*index*/) { return {(dtype)INFINITY, INT<%=i%>_MAX}; }
    __device__ ValueAndIndex MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(ValueAndIndex next, ValueAndIndex& accum) {
        bool accum_nan = !not_nan(accum.value);
        bool next_nan = !not_nan(next.value);
        if (accum_nan || next_nan) {
            if (next_nan && (!accum_nan || next.index < accum.index)) { accum = next; }
            return;
        }
        if (accum.value > next.value || (accum.value == next.value && next.index < accum.index)) { accum = next; }
    }
    __device__ idx_t MapOut(ValueAndIndex accum) { return accum.index; }
};
<% end %>

struct cumo_<%=type_name%>_argmax_int<%=i%>_impl {
    struct ValueAndIndex {
        dtype value;
        idx_t index;
    };
    // The identity has to lose to any element, including a NaN, and it
    // carries the largest index so that a tie goes to the earlier one.
    // NaN is the identity for the float types, the way it is for min and
    // max, so that every element being NaN answers the first index as
    // numo does.
    __device__ ValueAndIndex Identity(idx_t /*index*/) { return {<% if is_float %>(dtype)nan("")<% else %>DATA_MIN<% end %>, INT<%=i%>_MAX}; }
    __device__ ValueAndIndex MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(ValueAndIndex next, ValueAndIndex& accum) {
<% if is_float %>
        bool accum_nan = !not_nan(accum.value);
        bool next_nan = !not_nan(next.value);
        if (accum_nan || next_nan) {
            if (accum_nan && (!next_nan || next.index < accum.index)) { accum = next; }
            return;
        }
<% end %>
        if (accum.value < next.value || (accum.value == next.value && next.index < accum.index)) { accum = next; }
    }
    __device__ idx_t MapOut(ValueAndIndex accum) { return accum.index; }
};

<% if is_float %>
// numo answers the index of the first NaN as soon as one element is NaN,
// so a NaN wins outright here and the earliest of them wins among
// themselves. The identity cannot be NaN for that reason.
struct cumo_<%=type_name%>_argmax_nan_int<%=i%>_impl {
    struct ValueAndIndex {
        dtype value;
        idx_t index;
    };
    __device__ ValueAndIndex Identity(idx_t /*index*/) { return {-(dtype)INFINITY, INT<%=i%>_MAX}; }
    __device__ ValueAndIndex MapIn(dtype in, idx_t index) { return {in, index}; }
    __device__ void Reduce(ValueAndIndex next, ValueAndIndex& accum) {
        bool accum_nan = !not_nan(accum.value);
        bool next_nan = !not_nan(next.value);
        if (accum_nan || next_nan) {
            if (next_nan && (!accum_nan || next.index < accum.index)) { accum = next; }
            return;
        }
        if (accum.value < next.value || (accum.value == next.value && next.index < accum.index)) { accum = next; }
    }
    __device__ idx_t MapOut(ValueAndIndex accum) { return accum.index; }
};
<% end %>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_argmin_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_arg_split<dtype, idx_t, cumo_<%=type_name%>_argmin_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_argmin_int<%=i%>_impl{});
}
<% if is_float %>

void cumo_<%=type_name%>_argmin_nan_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_arg_split<dtype, idx_t, cumo_<%=type_name%>_argmin_nan_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_argmin_nan_int<%=i%>_impl{});
}
<% end %>

void cumo_<%=type_name%>_argmax_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_arg_split<dtype, idx_t, cumo_<%=type_name%>_argmax_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_argmax_int<%=i%>_impl{});
}
<% if is_float %>

void cumo_<%=type_name%>_argmax_nan_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_arg_split<dtype, idx_t, cumo_<%=type_name%>_argmax_nan_int<%=i%>_impl>(*arg, cumo_<%=type_name%>_argmax_nan_int<%=i%>_impl{});
}
<% end %>

#undef idx_t
<% end %>

<% end %>
<% end %>
