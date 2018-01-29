dtype <%=type_name%>_sum_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_zero;
    if (stride_idx == 1) {
        cumo_debug_breakpoint();
        return thrust::reduce(data_begin, data_end, init, thrust::plus<dtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust::plus<dtype>());
    }
}

dtype <%=type_name%>_prod_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_one;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust::multiplies<dtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust::multiplies<dtype>());
    }
}

dtype <%=type_name%>_min_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = DATA_MAX;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust::minimum<dtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust::minimum<dtype>());
    }
}

dtype <%=type_name%>_max_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = DATA_MIN;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust::maximum<dtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust::maximum<dtype>());
    }
}

void <%=type_name%>_minmax_kernel_launch(uint64_t n, char *p, ssize_t stride, dtype* amin, dtype* amax);
dtype <%=type_name%>_ptp_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    dtype min,max;
    <%=type_name%>_minmax_kernel_launch(n,p,stride,&min,&max);
    return m_sub(max,min);
}
