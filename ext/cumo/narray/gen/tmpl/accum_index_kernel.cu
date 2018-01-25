<% unless defined?($cumo_narray_gen_tmpl_accum_index_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_index_kernel_included = 1 %>
<% unless type_name == 'robject' %>

size_t <%=type_name%>_min_index_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    if (stride_idx == 1) {
        thrust::device_ptr<dtype> elem = thrust::min_element(data_begin, data_end);
        return (size_t)(elem.get() - data_begin.get());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        thrust_strided_range<thrust::device_vector<dtype>::iterator>::iterator iter = thrust::min_element(range.begin(), range.end());
        return (size_t)(iter - range.begin());
    }
}

size_t <%=type_name%>_max_index_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    if (stride_idx == 1) {
        thrust::device_ptr<dtype> elem = thrust::max_element(data_begin, data_end);
        return (size_t)(elem.get() - data_begin.get());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        thrust_strided_range<thrust::device_vector<dtype>::iterator>::iterator iter = thrust::max_element(range.begin(), range.end());
        return (size_t)(iter - range.begin());
    }
}

<% end %>
<% end %>
