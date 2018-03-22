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

template<typename Iterator>
__global__ void cumo_<%=type_name%>_min_index_int<%=i%>_kernel(Iterator begin, Iterator end, char *i_ptr, ssize_t i_step, char *o_ptr)
{
    Iterator iter = thrust::min_element(thrust::cuda::par, begin, end);
    size_t idx = (size_t)(iter - begin);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}

template<typename Iterator>
__global__ void cumo_<%=type_name%>_max_index_int<%=i%>_kernel(Iterator begin, Iterator end, char *i_ptr, ssize_t i_step, char *o_ptr)
{
    Iterator iter = thrust::max_element(thrust::cuda::par, begin, end);
    size_t idx = (size_t)(iter - begin);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_min_index_int<%=i%>_kernel_launch(uint64_t n, char *d_ptr, ssize_t d_step, char *i_ptr, ssize_t i_step, char* o_ptr)
{
    ssize_t d_step_idx = d_step / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)d_ptr);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)d_ptr) + n * d_step_idx);
    cumo_<%=type_name%>_min_index_int<%=i%>_kernel<<<1,1>>>(data_begin, data_end, i_ptr, i_step, o_ptr);
}

void cumo_<%=type_name%>_max_index_int<%=i%>_kernel_launch(uint64_t n, char *d_ptr, ssize_t d_step, char *i_ptr, ssize_t i_step, char* o_ptr)
{
    ssize_t d_step_idx = d_step / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)d_ptr);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)d_ptr) + n * d_step_idx);
    cumo_<%=type_name%>_max_index_int<%=i%>_kernel<<<1,1>>>(data_begin, data_end, i_ptr, i_step, o_ptr);
}
#undef idx_t
<% end %>

<% end %>
<% end %>
