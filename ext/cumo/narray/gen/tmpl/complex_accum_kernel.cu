dtype <%=type_name%>_sum_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_zero;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust_plus());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust_plus());
    }
}

dtype <%=type_name%>_prod_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_one;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust_multiplies());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust_multiplies());
    }
}

dtype <%=type_name%>_mean_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    dtype sum = <%=type_name%>_sum_kernel_launch(n, p, stride);
    return c_div_r(sum, n);
}

rtype <%=type_name%>_var_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    thrust_complex_variance_unary_op<dtype, rtype>  unary_op;
    thrust_complex_variance_binary_op<dtype, rtype> binary_op;
    thrust_complex_variance_data<dtype, rtype> init = {};
    thrust_complex_variance_data<dtype, rtype> result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, unary_op, init, binary_op);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), unary_op, init, binary_op);
    }
    return result.variance();
}

rtype <%=type_name%>_stddev_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    return r_sqrt(<%=type_name%>_var_kernel_launch(n, p, stride));
}

rtype <%=type_name%>_rms_kernel_launch(uint64_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    rtype init = 0;
    rtype result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, thrust_square(), init, thrust::plus<rtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), thrust_square(), init, thrust::plus<rtype>());
    }
    return r_sqrt(result/n);
}
